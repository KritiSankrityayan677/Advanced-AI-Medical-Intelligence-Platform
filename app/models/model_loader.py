"""
model_loader.py
Loads the trained DenseNet-121 model and prepares it for inference.
Caches the loaded model so subsequent calls are instant.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121

from app.core.config import (
    MODEL_WEIGHTS_PATH,
    ARCHITECTURE,
    NUM_CLASSES,
)

# Determine which device to use (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache for the loaded model (None until first call)
_model_cache = None


def _build_model() -> nn.Module:
    """
    Construct the DenseNet-121 architecture with a 4-class head.
    Loads pretrained weights from disk into the architecture.
    """
    # 1. Create the architecture (no pretrained ImageNet weights needed)
    model = densenet121(weights=None)

    # 2. Replace the ImageNet 1000-class head with our 4-class head
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)

    # 3. Load our trained weights from disk
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # 4. Move to device and set to evaluation mode
    model = model.to(DEVICE)
    model.eval()

    return model


def get_model() -> nn.Module:
    """
    Returns the loaded model. Loads it on first call, then caches.
    All other modules should call this instead of loading directly.
    """
    global _model_cache
    if _model_cache is None:
        print(f"Loading model from {MODEL_WEIGHTS_PATH}...")
        _model_cache = _build_model()
        print(f"Model loaded and ready on {DEVICE}")
    return _model_cache


if __name__ == "__main__":
    model = get_model()
    print(f"\nModel type: {type(model).__name__}")
    print(f"Classifier: {model.classifier}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")