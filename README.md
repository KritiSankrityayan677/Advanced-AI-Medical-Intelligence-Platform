# NeuroVision AI

Brain MRI tumor classification platform with Explainable AI and LLM-generated radiology reports.

> ⚠️ Preliminary AI-assisted output only. Not a medical diagnosis — requires radiologist review.

## Features

- CNN-based classification of brain MRI scans into 4 classes: glioma, meningioma, pituitary tumor, no tumor detected
- Grad-CAM heatmap overlay for explainability
- LLM-generated structured radiology report (Groq API)
- FastAPI REST backend with SQLAlchemy-backed prediction history
- Streamlit frontend for upload, results, and history browsing

## Tech Stack

Streamlit · FastAPI · PyTorch/torchvision · grad-cam · Groq API · SQLAlchemy

## Getting Started

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Add your LLM API key to a `.env` file in `backend/`:
```
GROQ_API_KEY=your_key_here
```

## Project Structure

```
backend/    # FastAPI app, model, Grad-CAM, LLM integration, database
frontend/   # Streamlit UI
notebooks/  # Model training notebook
```

## Disclaimer

All outputs (classifications, confidence scores, Grad-CAM overlays, reports) are preliminary and require review by a qualified radiologist. This is a student project, not a certified medical device.

## License

Specify your license here.
