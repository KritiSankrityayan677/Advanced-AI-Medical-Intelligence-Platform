"""
schemas.py
Pydantic models defining the shape of API requests and responses.
FastAPI uses these for validation and auto-generated documentation.
"""

from typing import Dict
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from typing import Dict, List, Optional
from datetime import datetime


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
    
class PredictionRecord(BaseModel):
    """One historical prediction row."""
    id: int
    timestamp: datetime
    filename: Optional[str]
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    report_text: Optional[str]
    report_model: Optional[str]
    model_architecture: str

    class Config:
        from_attributes = True  # allow reading from SQLAlchemy objects


class HistoryResponse(BaseModel):
    """Paginated history response."""
    total: int
    count: int
    predictions: List[PredictionRecord]