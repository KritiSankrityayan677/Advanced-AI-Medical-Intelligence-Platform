"""
test_config.py
Tests for the configuration module.
These tests verify config values without needing the large model weights file,
so they run cleanly in CI environments.
"""

import pytest

from app.core.config import (
    MODEL_WEIGHTS_PATH,
    MODEL_METADATA_PATH,
    ARCHITECTURE,
    NUM_CLASSES,
    CLASS_NAMES,
    INPUT_SIZE,
    NORM_MEAN,
    NORM_STD,
    GRADCAM_TARGET_LAYER,
)


def test_metadata_file_exists():
    """Metadata JSON must exist (skipped if not present, e.g. on CI)."""
    if not MODEL_METADATA_PATH.exists():
        pytest.skip("Metadata not available in this environment")
    assert MODEL_METADATA_PATH.exists()


def test_model_weights_file_exists():
    """Model weights file must exist (skipped if not present, e.g. on CI)."""
    if not MODEL_WEIGHTS_PATH.exists():
        pytest.skip("Model weights not available in this environment")
    assert MODEL_WEIGHTS_PATH.exists()


def test_num_classes_is_four():
    """Model must have exactly 4 classes."""
    assert NUM_CLASSES == 4


def test_class_names_are_correct():
    """Class names must match the expected tumor types."""
    expected = {"glioma", "meningioma", "notumor", "pituitary"}
    assert set(CLASS_NAMES) == expected


def test_input_size_is_224():
    """Input size must be 224 (matching DenseNet-121 pretraining)."""
    assert INPUT_SIZE == 224


def test_normalization_values_are_imagenet():
    """Normalization must use ImageNet stats."""
    assert NORM_MEAN == [0.485, 0.456, 0.406]
    assert NORM_STD == [0.229, 0.224, 0.225]


def test_architecture_is_densenet():
    """Architecture must be densenet121."""
    assert ARCHITECTURE == "densenet121"


def test_gradcam_target_layer_defined():
    """Grad-CAM target layer must be a non-empty string."""
    assert isinstance(GRADCAM_TARGET_LAYER, str)
    assert len(GRADCAM_TARGET_LAYER) > 0