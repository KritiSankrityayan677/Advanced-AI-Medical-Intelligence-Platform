"""
test_predictor.py
Tests for the prediction module.
"""

import pytest
from app.models.predictor import predict
from app.core.config import CLASS_NAMES


def test_predict_returns_dict(sample_glioma_image):
    """predict() must return a dictionary."""
    result = predict(sample_glioma_image)
    assert isinstance(result, dict)


def test_predict_has_required_keys(sample_glioma_image):
    """The prediction dict must have all required keys."""
    result = predict(sample_glioma_image)
    assert "predicted_class" in result
    assert "confidence" in result
    assert "all_probabilities" in result


def test_predicted_class_is_valid(sample_glioma_image):
    """The predicted class must be one of the known class names."""
    result = predict(sample_glioma_image)
    assert result["predicted_class"] in CLASS_NAMES


def test_confidence_is_valid_probability(sample_glioma_image):
    """Confidence must be a probability between 0 and 1."""
    result = predict(sample_glioma_image)
    conf = result["confidence"]
    assert 0.0 <= conf <= 1.0


def test_all_probabilities_sum_to_one(sample_glioma_image):
    """All class probabilities must sum to approximately 1.0."""
    result = predict(sample_glioma_image)
    total = sum(result["all_probabilities"].values())
    assert abs(total - 1.0) < 0.01  # allow small floating-point tolerance


def test_all_probabilities_have_all_classes(sample_glioma_image):
    """The all_probabilities dict must contain every class."""
    result = predict(sample_glioma_image)
    assert set(result["all_probabilities"].keys()) == set(CLASS_NAMES)


def test_confidence_matches_top_probability(sample_glioma_image):
    """The confidence value must equal the top class's probability."""
    result = predict(sample_glioma_image)
    top_class = result["predicted_class"]
    top_prob = result["all_probabilities"][top_class]
    assert abs(result["confidence"] - top_prob) < 0.01


def test_all_classes_produce_predictions(
    sample_glioma_image,
    sample_meningioma_image,
    sample_notumor_image,
    sample_pituitary_image,
):
    """Every sample from every class must produce a valid prediction."""
    for img in [
        sample_glioma_image,
        sample_meningioma_image,
        sample_notumor_image,
        sample_pituitary_image,
    ]:
        result = predict(img)
        assert result["predicted_class"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0