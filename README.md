# Medical AI Platform

Advanced AI Medical Intelligence Platform for brain tumor classification from MRI scans.

## Current Status
Phase 1 complete: trained baseline classifier.

## Phase 1 Results
- Architecture: DenseNet-121 (transfer learning from ImageNet)
- Dataset: Brain Tumor MRI Dataset (4 classes: glioma, meningioma, notumor, pituitary)
- Test accuracy: 94.7%
- Macro ROC-AUC: 0.989
- Per-class sensitivity: glioma 0.813, meningioma 0.990, notumor 0.998, pituitary 0.988

## Project Structure
- `notebooks/` — Jupyter notebooks for training and analysis
- `models/` — Trained model weights and metadata
- `app/` — Backend application (API, model loader, XAI, LLM, DB) — coming in future phases
- `frontend/` — Web interface — coming in future phases
- `tests/` — Automated tests — coming in future phases

## Roadmap
- [x] Phase 1 — Model training and evaluation
- [ ] Phase 2 — Explainable AI (Grad-CAM heatmaps)
- [ ] Phase 3 — LLM-based report generation
- [ ] Phase 4 — REST API with FastAPI
- [ ] Phase 5 — Database for prediction history
- [ ] Phase 6 — Web frontend
- [ ] Phase 7 — Docker deployment