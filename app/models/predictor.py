"""
predictor.py
Takes an image and returns a prediction with probabilities.
This is the main inference API used by the REST API, notebooks, and tests.
"""

from pathlib import Path
from typing import Union, Dict

import torch
from PIL import Image
from torchvision import transforms

from app.core.config import (
    CLASS_NAMES,
    INPUT_SIZE,
    NORM_MEAN,
    NORM_STD,
)
from app.models.model_loader import get_model, DEVICE


# Preprocessing pipeline — must match Phase 1's eval_tf exactly
_preprocess = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def _load_and_preprocess(image_path: Union[str, Path]) -> torch.Tensor:
    """
    Load an image from disk, preprocess it, and return a batch tensor
    of shape (1, 3, 224, 224) ready for the model.
    """
    image = Image.open(image_path)
    tensor = _preprocess(image)
    batch = tensor.unsqueeze(0)  # add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    return batch.to(DEVICE)


def predict(image_path: Union[str, Path]) -> Dict:
    """
    Run the model on an image and return the prediction.

    Args:
        image_path: Path to an image file (jpg, png, etc.)

    Returns:
        Dict with keys:
            predicted_class: str  — the top-1 class name
            confidence: float     — probability of the top-1 class (0.0-1.0)
            all_probabilities: Dict[str, float] — probability per class
    """
    # 1. Load the cached model (or load if first call)
    model = get_model()

    # 2. Load and preprocess the image
    batch = _load_and_preprocess(image_path)

    # 3. Forward pass — no gradients needed for inference
    with torch.no_grad():
        logits = model(batch)                          # shape (1, 4) — raw scores
        probabilities = torch.softmax(logits, dim=1)   # shape (1, 4) — sum to 1
        probabilities = probabilities.squeeze(0)       # shape (4,)  — remove batch dim

    # 4. Convert to plain Python types for easy handling
    probabilities = probabilities.cpu().tolist()       # tensor → list of floats

    # 5. Find the winner
    top_idx = int(torch.tensor(probabilities).argmax())
    predicted_class = CLASS_NAMES[top_idx]
    confidence = float(probabilities[top_idx])

    # 6. Build the per-class probability dictionary
    all_probs = {
        class_name: round(float(prob), 4)
        for class_name, prob in zip(CLASS_NAMES, probabilities)
    }

    # 7. Return a clean result dictionary
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.models.predictor <path/to/image.jpg>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(f"\nPrediction for: {sys.argv[1]}")
    print(f"  Class      : {result['predicted_class']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"\nAll probabilities:")
    for cls, prob in result["all_probabilities"].items():
        print(f"  {cls:12s}: {prob:.2%}")