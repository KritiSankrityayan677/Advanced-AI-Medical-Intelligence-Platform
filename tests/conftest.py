"""
conftest.py
Shared pytest fixtures for the Medical AI Platform test suite.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_SAMPLES_DIR = PROJECT_ROOT / "test_samples"


@pytest.fixture
def test_samples_dir() -> Path:
    """Path to the folder containing test images."""
    if not TEST_SAMPLES_DIR.exists():
        pytest.skip(f"Test samples not found at {TEST_SAMPLES_DIR}")
    return TEST_SAMPLES_DIR


@pytest.fixture
def sample_glioma_image(test_samples_dir) -> Path:
    """Path to a known glioma test image."""
    img = test_samples_dir / "glioma_1.jpg"
    if not img.exists():
        pytest.skip(f"Sample image not found: {img}")
    return img


@pytest.fixture
def sample_meningioma_image(test_samples_dir) -> Path:
    """Path to a known meningioma test image."""
    img = test_samples_dir / "meningioma_1.jpg"
    if not img.exists():
        pytest.skip(f"Sample image not found: {img}")
    return img


@pytest.fixture
def sample_notumor_image(test_samples_dir) -> Path:
    """Path to a known notumor test image."""
    img = test_samples_dir / "notumor_1.jpg"
    if not img.exists():
        pytest.skip(f"Sample image not found: {img}")
    return img


@pytest.fixture
def sample_pituitary_image(test_samples_dir) -> Path:
    """Path to a known pituitary test image."""
    img = test_samples_dir / "pituitary_1.jpg"
    if not img.exists():
        pytest.skip(f"Sample image not found: {img}")
    return img