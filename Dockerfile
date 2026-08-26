# Dockerfile for Hugging Face Spaces (24/7 Free Cloud Hosting, No Credit Card)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEPTHWIZARD_MODEL_ID=depth-anything/Depth-Anything-V2-Base-hf \
    DEPTHWIZARD_MAX_INPUT_SIZE=1024 \
    DEPTHWIZARD_MAX_DECODED_PIXELS=50000000 \
    DEPTHWIZARD_ARTIFACT_DIR=/tmp/artifacts \
    DEPTHWIZARD_CORS_ORIGINS=* \
    HF_HOME=/tmp/huggingface \
    PORT=7860

WORKDIR /app

# Install system dependencies (rasterio and pillow C libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create artifacts directory with open permissions
RUN mkdir -p /tmp/artifacts /tmp/huggingface && \
    chmod -R 777 /tmp/artifacts /tmp/huggingface

# Copy application source code
COPY backend/ /app/backend/
COPY ml/ /app/ml/

# Hugging Face Spaces expects traffic on port 7860
EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
