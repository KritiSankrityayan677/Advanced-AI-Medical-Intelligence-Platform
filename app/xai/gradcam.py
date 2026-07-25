"""
gradcam.py
Generates Grad-CAM heatmaps that visualize what regions of an input
image the model attended to when making its prediction.
"""

from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.core.config import (
    INPUT_SIZE,
    NORM_MEAN,
    NORM_STD,
    CLASS_NAMES,
    GRADCAM_TARGET_LAYER,
)
from app.models.model_loader import get_model, DEVICE


# Preprocessing pipeline — matches Phase 1's eval_tf
_preprocess = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Look up the Grad-CAM target layer by name from the config.
    For DenseNet-121, this is 'features.norm5' — the last conv+BN block.
    """
    parts = GRADCAM_TARGET_LAYER.split(".")
    layer = model
    for part in parts:
        layer = getattr(layer, part)
    return layer


def _tensor_to_original_rgb(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load the original image, resize to model input size, and return it
    as a NumPy array in [0, 1] range — this is the base for the overlay.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    rgb = np.array(image, dtype=np.float32) / 255.0
    return rgb


def generate_heatmap(
    image_path: Union[str, Path],
    target_class: int = None,
) -> Tuple[Image.Image, str, float]:
    """
    Generate a Grad-CAM heatmap overlay for the given image.

    Args:
        image_path: Path to the image file.
        target_class: Class index to explain. If None, uses the model's top prediction.

    Returns:
        Tuple of:
            heatmap_image: PIL.Image — original scan with heatmap overlay
            predicted_class: str — name of the class the heatmap explains
            confidence: float — probability of that class (0.0-1.0)
    """
    # 1. Load the model
    model = get_model()

    # 2. Preprocess the image into a tensor batch
    image = Image.open(image_path)
    input_tensor = _preprocess(image).unsqueeze(0).to(DEVICE)

    # 3. Get the predicted class if none specified
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

        if target_class is None:
            target_class = int(probabilities.argmax())

        confidence = float(probabilities[target_class])
        predicted_class = CLASS_NAMES[target_class]

    # 4. Set up Grad-CAM on the target layer
    target_layer = _get_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])

    # 5. Compute the heatmap for the target class
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]  # unwrap the batch dimension

    # 6. Load the original image as RGB for overlay
    rgb_image = _tensor_to_original_rgb(image_path)

    # 7. Blend the heatmap onto the original scan
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    # 8. Convert NumPy array back to a PIL Image for easy display/saving
    heatmap_image = Image.fromarray(overlay)

    return heatmap_image, predicted_class, confidence


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.xai.gradcam <path/to/image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    heatmap, predicted_class, confidence = generate_heatmap(image_path)

    output_path = "gradcam_output.png"
    heatmap.save(output_path)

    print(f"\nPredicted class : {predicted_class}")
    print(f"Confidence      : {confidence:.2%}")
    print(f"Heatmap saved to: {output_path}")