"""
database.py
Database connection setup using SQLAlchemy.
Uses SQLite for simplicity — the whole DB lives in a single file.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import PROJECT_ROOT


# The database file lives in the project root
DB_PATH = PROJECT_ROOT / "medical_ai.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# The engine is the low-level connection manager
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
)

# A session factory — each call to SessionLocal() gives a new session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# All ORM models will inherit from this base class
Base = declarative_base()


def get_db():
    """
    Yields a database session for a single request, then closes it.
    Used by FastAPI as a dependency.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()