# Hugging Face Space (Docker SDK) — free CPU Basic hardware, no credit card.
#
# Deliberately no Gradio and no ZeroGPU. ZeroGPU only allocates a GPU inside
# @spaces.GPU functions driven by Gradio events, so a mounted FastAPI route
# never receives one; and Gradio's client schema walker crashes on this API's
# multipart UploadFile OpenAPI schema. Plain FastAPI removes both failure modes.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEPTHWIZARD_MODEL_ID=depth-anything/Depth-Anything-V2-Base-hf \
    DEPTHWIZARD_MAX_INPUT_SIZE=518 \
    DEPTHWIZARD_MAX_DECODED_PIXELS=50000000 \
    DEPTHWIZARD_MAX_UPLOAD_MB=32 \
    DEPTHWIZARD_ARTIFACT_DIR=/tmp/artifacts \
    DEPTHWIZARD_CORS_ORIGINS=* \
    HF_HOME=/opt/hf-cache \
    PORT=7860

# Spaces runs the container as UID 1000; match it so cached weights stay readable.
RUN useradd -m -u 1000 user

WORKDIR /app

# CPU-only Torch first: ~200 MB instead of the ~2.5 GB CUDA build, and this
# Space has no GPU. requirements.txt then sees torch>=2.2 already satisfied.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Bake the weights into the image. Downloading ~400 MB on every cold start is
# the main cause of Space startup timeouts and leaves the API dead if the Hub
# is briefly unreachable.
RUN mkdir -p /opt/hf-cache && python -c "from transformers import AutoImageProcessor as P, AutoModelForDepthEstimation as M; import os; m=os.environ['DEPTHWIZARD_MODEL_ID']; P.from_pretrained(m); M.from_pretrained(m)" && chown -R user:user /opt/hf-cache

COPY backend/ /app/backend/
COPY ml/ /app/ml/

RUN mkdir -p /tmp/artifacts && chown -R user:user /tmp/artifacts /app

USER user
EXPOSE 7860

# --proxy-headers + --forwarded-allow-ips are required behind the Spaces
# reverse proxy. Without them request.base_url is http://, so every artifact URL
# handed to the HTTPS frontend is mixed content and silently blocked.
CMD ["uvicorn", "backend.app:app", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--timeout-keep-alive", "300"]
