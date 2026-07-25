"""
test_gradcam.py
Tests for the Grad-CAM explainer.
"""

from PIL import Image
from app.xai.gradcam import generate_heatmap
from app.core.config import CLASS_NAMES, INPUT_SIZE


def test_heatmap_returns_three_values(sample_glioma_image):
    """generate_heatmap must return exactly 3 values."""
    result = generate_heatmap(sample_glioma_image)
    assert len(result) == 3


def test_heatmap_is_pil_image(sample_glioma_image):
    """The returned heatmap must be a PIL Image."""
    heatmap, _, _ = generate_heatmap(sample_glioma_image)
    assert isinstance(heatmap, Image.Image)


def test_heatmap_has_correct_size(sample_glioma_image):
    """The heatmap must have the model's input size."""
    heatmap, _, _ = generate_heatmap(sample_glioma_image)
    assert heatmap.size == (INPUT_SIZE, INPUT_SIZE)


def test_heatmap_is_rgb(sample_glioma_image):
    """The heatmap must be a 3-channel RGB image."""
    heatmap, _, _ = generate_heatmap(sample_glioma_image)
    assert heatmap.mode == "RGB"


def test_predicted_class_is_valid(sample_glioma_image):
    """The returned class name must be one of the known classes."""
    _, cls, _ = generate_heatmap(sample_glioma_image)
    assert cls in CLASS_NAMES


def test_confidence_is_valid_probability(sample_glioma_image):
    """The returned confidence must be between 0 and 1."""
    _, _, conf = generate_heatmap(sample_glioma_image)
    assert 0.0 <= conf <= 1.0


def test_target_class_override_works(sample_glioma_image):
    """Explicit target_class must be respected."""
    for class_idx in range(len(CLASS_NAMES)):
        _, cls, _ = generate_heatmap(sample_glioma_image, target_class=class_idx)
        assert cls == CLASS_NAMES[class_idx]