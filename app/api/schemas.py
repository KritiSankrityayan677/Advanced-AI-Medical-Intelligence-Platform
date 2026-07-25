"""
schemas.py
Pydantic models defining the shape of API requests and responses.
FastAPI uses these for validation and auto-generated documentation.
"""

from typing import Dict
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status: str
    architecture: str
    device: str


class PredictionResponse(BaseModel):
    """Response for the /predict endpoint."""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]


class ReportSections(BaseModel):
    """The four structured sections of a generated report."""
    findings: str
    impression: str
    recommendation: str
    disclaimer: str


class FullReportResponse(BaseModel):
    """Response for the /predict/report endpoint."""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    report: ReportSections
    raw_report: str
    heatmap_base64: str
    model: str