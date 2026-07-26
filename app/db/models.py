"""
models.py
SQLAlchemy ORM models — one class per database table.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON

from app.db.database import Base


class Prediction(Base):
    """One row per prediction made through the API."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Upload info
    filename = Column(String, nullable=True)

    # Prediction outputs
    predicted_class = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    all_probabilities = Column(JSON, nullable=False)

    # Optional LLM report (only present for /predict/report calls)
    report_text = Column(Text, nullable=True)
    report_model = Column(String, nullable=True)

    # MLOps: track which model version produced this
    model_architecture = Column(String, nullable=False)