"""
test_api.py
Tests for the FastAPI endpoints using FastAPI's in-memory TestClient.
Tests needing the model or LLM key skip gracefully when unavailable.
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.config import MODEL_WEIGHTS_PATH, GROQ_API_KEY

client = TestClient(app)


def _fake_image_bytes():
    """Create a small in-memory JPEG for testing uploads."""
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_returns_ok():
    """Health endpoint must return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_non_image():
    """Uploading a non-image must return HTTP 400."""
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.skipif(not MODEL_WEIGHTS_PATH.exists(),
                    reason="Model weights not available")
def test_predict_returns_valid_prediction():
    """Predict endpoint must return a valid prediction for an image."""
    response = client.post(
        "/predict",
        files={"file": ("scan.jpg", _fake_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "predicted_class" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["all_probabilities"]) == 4


@pytest.mark.skipif(not (MODEL_WEIGHTS_PATH.exists() and GROQ_API_KEY),
                    reason="Needs model weights and GROQ_API_KEY")
def test_report_endpoint_returns_full_response():
    """Report endpoint must return report sections and a heatmap."""
    response = client.post(
        "/predict/report",
        files={"file": ("scan.jpg", _fake_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "report" in body
    assert set(body["report"].keys()) == {
        "findings", "impression", "recommendation", "disclaimer"
    }
    assert len(body["heatmap_base64"]) > 0