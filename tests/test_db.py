"""
test_db.py
Tests for the database layer and history endpoints.
Uses an in-memory SQLite database so tests are fast and isolated.
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.main import app
from app.db.database import Base, get_db
from app.db import crud
from app.core.config import MODEL_WEIGHTS_PATH


# ── In-memory test database ────────────────────────────
from sqlalchemy.pool import StaticPool

TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # share one connection across threads
)
TestSessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

def override_get_db():
    """Provide a test session instead of the real one."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Create fresh tables before each test, drop after."""
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


client = TestClient(app)


def _fake_image_bytes():
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ── CRUD unit tests (offline) ──────────────────────────
def test_create_prediction_persists_row():
    """create_prediction must insert and return a row with an ID."""
    db = TestSessionLocal()
    row = crud.create_prediction(
        db=db,
        predicted_class="glioma",
        confidence=0.9,
        all_probabilities={"glioma": 0.9, "meningioma": 0.05,
                           "notumor": 0.03, "pituitary": 0.02},
        model_architecture="densenet121",
        filename="test.jpg",
    )
    assert row.id is not None
    assert row.timestamp is not None
    assert row.predicted_class == "glioma"
    db.close()


def test_get_predictions_returns_newest_first():
    """get_predictions must order by timestamp descending."""
    db = TestSessionLocal()
    for cls in ["glioma", "meningioma", "notumor"]:
        crud.create_prediction(
            db=db, predicted_class=cls, confidence=0.8,
            all_probabilities={cls: 0.8},
            model_architecture="densenet121",
        )
    rows = crud.get_predictions(db, limit=10)
    assert len(rows) == 3
    # Most recent insert should come first
    assert rows[0].predicted_class == "notumor"
    db.close()


def test_count_predictions():
    """count_predictions must return the correct total."""
    db = TestSessionLocal()
    assert crud.count_predictions(db) == 0
    crud.create_prediction(
        db=db, predicted_class="glioma", confidence=0.9,
        all_probabilities={"glioma": 0.9},
        model_architecture="densenet121",
    )
    assert crud.count_predictions(db) == 1
    db.close()


# ── History endpoint tests (offline) ───────────────────
def test_history_empty_returns_zero_total():
    """/history on an empty DB must return total 0."""
    response = client.get("/history")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["predictions"] == []


def test_history_by_id_not_found_returns_404():
    """/history/999 on a nonexistent id must return 404."""
    response = client.get("/history/999")
    assert response.status_code == 404


# ── Full-stack test (needs model) ──────────────────────
@pytest.mark.skipif(not MODEL_WEIGHTS_PATH.exists(),
                    reason="Model weights not available")
def test_predict_endpoint_persists_history():
    """A successful predict call must create a history record."""
    # Predict
    response = client.post(
        "/predict",
        files={"file": ("scan.jpg", _fake_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200

    # History should now contain one record
    history = client.get("/history").json()
    assert history["total"] == 1
    assert history["predictions"][0]["predicted_class"] in {
        "glioma", "meningioma", "notumor", "pituitary"
    }