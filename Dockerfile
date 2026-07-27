FROM python:3.12-slim

# System deps needed for opencv-headless and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Start the API
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]