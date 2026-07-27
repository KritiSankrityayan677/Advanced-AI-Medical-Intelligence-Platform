"""
main.py
FastAPI application exposing the Medical AI Platform over HTTP.

Run with:
    uvicorn app.api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

import io
import os
import base64
import tempfile
import gc
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from groq import APITimeoutError, APIConnectionError

from app.core.config import (
    ARCHITECTURE,
    LLM_MODEL,
    MAX_CONCURRENT_INFERENCE,
    INFERENCE_QUEUE_TIMEOUT_SECONDS,
)
from app.models.model_loader import DEVICE, get_model
from app.models.predictor import predict
from app.xai.gradcam import generate_heatmap
from app.llm.report_generator import generate_report
from app.api.schemas import (
    HealthResponse,
    PredictionResponse,
    ReportSections,
    FullReportResponse,
)

from typing import List
from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.database import Base, engine, get_db
from app.db import crud

from app.api.schemas import (
    HealthResponse,
    PredictionResponse,
    ReportSections,
    FullReportResponse,
    PredictionRecord,
    HistoryResponse,
)

# Gate on the heavy inference pipelines (predict + Grad-CAM [+ report]) so
# requests run one at a time instead of thrashing for the same limited CPU.
# See app/core/config.py for the rationale.
_inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Load the model at boot, not on the first real request, so no user
    # request ever pays the cold-load cost.
    get_model()
    yield


# ── Create the app ─────────────────────────────────────
app = FastAPI(
    title="Medical AI Platform API",
    description="Brain tumor classification with Grad-CAM explainability "
                "and AI-generated preliminary reports.",
    version="1.0.0",
    lifespan=_lifespan,
)

# Create database tables on first startup (safe to call every time)
Base.metadata.create_all(bind=engine)

# Allow browser-based frontends (Phase 6) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _acquire_inference_slot():
    """Wait for a free inference slot, but never wait indefinitely — fail
    fast with a clear 'busy' response instead of queueing silently until
    some external proxy kills the connection with no explanation."""
    try:
        await asyncio.wait_for(
            _inference_semaphore.acquire(), timeout=INFERENCE_QUEUE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server is busy processing another request. Please try again shortly.",
        )


# ── Helpers ────────────────────────────────────────────
def _validate_image(file: UploadFile):
    """Reject anything that isn't an image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")


def _save_bytes_to_temp(data: bytes, filename: str) -> str:
    """Write uploaded bytes to a temp file and return its path."""
    suffix = os.path.splitext(filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def _encode_image_base64(pil_image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Endpoints ──────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Medical AI Platform API. See /docs for usage."}


@app.get("/health", response_model=HealthResponse)
def health():
    """Simple liveness check."""
    return HealthResponse(
        status="ok",
        architecture=ARCHITECTURE,
        device=str(DEVICE),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a brain MRI and get the predicted class, probabilities, and Grad-CAM heatmap."""
    _validate_image(file)
    data = await file.read()
    tmp_path = _save_bytes_to_temp(data, file.filename)
    acquired = False
    try:
        await _acquire_inference_slot()
        acquired = True
        result = await run_in_threadpool(predict, tmp_path)
        gc.collect()
        heatmap_img, _, _ = await run_in_threadpool(generate_heatmap, tmp_path)
        gc.collect()
        heatmap_b64 = await run_in_threadpool(_encode_image_base64, heatmap_img)
        gc.collect()
    finally:
        if acquired:
            _inference_semaphore.release()
        os.remove(tmp_path)

    # Persist the prediction
    crud.create_prediction(
        db=db,
        filename=file.filename,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        all_probabilities=result["all_probabilities"],
        model_architecture=ARCHITECTURE,
    )

    return PredictionResponse(**result, heatmap_base64=heatmap_b64)


@app.post("/predict/report", response_model=FullReportResponse)
async def predict_report_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a brain MRI and get prediction, Grad-CAM heatmap, and a full report."""
    _validate_image(file)
    data = await file.read()
    tmp_path = _save_bytes_to_temp(data, file.filename)
    acquired = False
    try:
        await _acquire_inference_slot()
        acquired = True
        prediction = await run_in_threadpool(predict, tmp_path)
        gc.collect()
        heatmap_img, _, _ = await run_in_threadpool(generate_heatmap, tmp_path)
        gc.collect()
        try:
            report = await run_in_threadpool(generate_report, prediction)
        except APITimeoutError:
            raise HTTPException(
                status_code=504,
                detail="The report-generation service (Groq) timed out. Please try again.",
            )
        except APIConnectionError:
            raise HTTPException(
                status_code=502,
                detail="Could not reach the report-generation service (Groq). Please try again shortly.",
            )
        gc.collect()
        heatmap_b64 = await run_in_threadpool(_encode_image_base64, heatmap_img)
        gc.collect()
    finally:
        if acquired:
            _inference_semaphore.release()
        os.remove(tmp_path)

    # Persist the prediction along with the report
    crud.create_prediction(
        db=db,
        filename=file.filename,
        predicted_class=prediction["predicted_class"],
        confidence=prediction["confidence"],
        all_probabilities=prediction["all_probabilities"],
        model_architecture=ARCHITECTURE,
        report_text=report["raw_report"],
        report_model=report["model"],
    )

    return FullReportResponse(
        predicted_class=prediction["predicted_class"],
        confidence=prediction["confidence"],
        all_probabilities=prediction["all_probabilities"],
        report=ReportSections(**report["sections"]),
        raw_report=report["raw_report"],
        heatmap_base64=heatmap_b64,
        model=report["model"],
    )
    
@app.get("/history", response_model=HistoryResponse)
def history_endpoint(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Return the most recent predictions, newest first."""
    predictions = crud.get_predictions(db, limit=limit, offset=offset)
    total = crud.count_predictions(db)
    return HistoryResponse(
        total=total,
        count=len(predictions),
        predictions=predictions,
    )


@app.get("/history/{prediction_id}", response_model=PredictionRecord)
def history_by_id_endpoint(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    """Return one specific prediction by ID."""
    row = crud.get_prediction_by_id(db, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return row