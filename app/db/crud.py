"""
crud.py
Reusable database operations (Create, Read, Update, Delete).
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.db.models import Prediction


def create_prediction(
    db: Session,
    predicted_class: str,
    confidence: float,
    all_probabilities: dict,
    model_architecture: str,
    filename: Optional[str] = None,
    report_text: Optional[str] = None,
    report_model: Optional[str] = None,
) -> Prediction:
    """Insert a new prediction row and return the created object."""
    row = Prediction(
        filename=filename,
        predicted_class=predicted_class,
        confidence=confidence,
        all_probabilities=all_probabilities,
        report_text=report_text,
        report_model=report_model,
        model_architecture=model_architecture,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_predictions(
    db: Session,
    limit: int = 50,
    offset: int = 0,
) -> List[Prediction]:
    """Return the most recent predictions, newest first."""
    return (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_prediction_by_id(db: Session, prediction_id: int) -> Optional[Prediction]:
    """Return a specific prediction by ID, or None."""
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()


def count_predictions(db: Session) -> int:
    """Total number of predictions in the database."""
    return db.query(Prediction).count()