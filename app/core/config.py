"""
config.py
Central configuration for the Medical AI Platform.
Reads model_metadata.json and exposes settings to other modules.
"""
import os
import json
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Locate the project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Key file paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_WEIGHTS_PATH = MODELS_DIR / "brain_tumor_densenet121.pth"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Load metadata from JSON
if not MODEL_METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Model metadata not found at {MODEL_METADATA_PATH}. "
        "Did you complete Phase 1?"
    )

with open(MODEL_METADATA_PATH, "r") as f:
    METADATA = json.load(f)

# Expose key settings as constants for other modules to use
ARCHITECTURE = METADATA["architecture"]
NUM_CLASSES = METADATA["num_classes"]
CLASS_NAMES = METADATA["classes"]
INPUT_SIZE = METADATA["input_size"]
NORM_MEAN = METADATA["normalization"]["mean"]
NORM_STD = METADATA["normalization"]["std"]
GRADCAM_TARGET_LAYER = METADATA["gradcam_target_layer"]

# ── LLM settings (loaded from .env) ──────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def print_config():
    """Print all config values for debugging."""
    print("=" * 55)
    print("MEDICAL AI PLATFORM — CONFIGURATION")
    print("=" * 55)
    print(f"Project root         : {PROJECT_ROOT}")
    print(f"Model weights        : {MODEL_WEIGHTS_PATH}")
    print(f"Model metadata       : {MODEL_METADATA_PATH}")
    print(f"Architecture         : {ARCHITECTURE}")
    print(f"Number of classes    : {NUM_CLASSES}")
    print(f"Class names          : {CLASS_NAMES}")
    print(f"Input size           : {INPUT_SIZE}")
    print(f"Normalization mean   : {NORM_MEAN}")
    print(f"Normalization std    : {NORM_STD}")
    print(f"Grad-CAM target layer: {GRADCAM_TARGET_LAYER}")
    print("=" * 55)


if __name__ == "__main__":
    print_config()