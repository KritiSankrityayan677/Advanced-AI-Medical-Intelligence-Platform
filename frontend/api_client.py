"""
api_client.py
Thin wrapper around the Medical AI Platform REST API.
Streamlit UI code calls these functions instead of hitting HTTP directly.
"""

import os
import requests

# Default to localhost, but allow override via environment variable
#API_BASE_URL = os.getenv("MEDICAL_AI_API_URL", "http://127.0.0.1:8000")
API_BASE_URL = "https://neurovision-ai.up.railway.app"


def check_health() -> dict:
    """Check if the API is reachable and return its health status."""
    print(f"Checking health of API at {API_BASE_URL}")
    response = requests.get(f"{API_BASE_URL}/health", timeout=(10, 300))
    response.raise_for_status()
    return response.json()


def predict(image_bytes: bytes, filename: str = "scan.jpg") -> dict:
    """Send an image to /predict and return the prediction dict."""
    files = {"file": (filename, image_bytes, "image/jpeg")}
    print(f"Sending image to API at {API_BASE_URL}/predict")
    response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=(10, 400))
    response.raise_for_status()
    return response.json()


def predict_report(image_bytes: bytes, filename: str = "scan.jpg") -> dict:
    """Send an image to /predict/report and return the full response with report + heatmap."""
    files = {"file": (filename, image_bytes, "image/jpeg")}
    print(f"Sending image to API at {API_BASE_URL}/predict/report")
    response = requests.post(
        f"{API_BASE_URL}/predict/report",
        files=files,
        timeout=(10, 400),  # LLM call takes longer
    )
    response.raise_for_status()
    return response.json()


def get_history(limit: int = 50, offset: int = 0) -> dict:
    """Fetch prediction history from the API."""
    print(f"Fetching prediction history from API at {API_BASE_URL}/history")
    response = requests.get(
        f"{API_BASE_URL}/history",
        params={"limit": limit, "offset": offset},
        timeout=(10, 300),
    )
    response.raise_for_status()
    return response.json()


def get_prediction_by_id(prediction_id: int) -> dict:
    """Fetch a single historical prediction by ID."""
    response = requests.get(f"{API_BASE_URL}/history/{prediction_id}", timeout=(10, 300))
    response.raise_for_status()
    return response.json()