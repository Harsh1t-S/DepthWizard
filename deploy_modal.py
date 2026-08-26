"""Modal deployment for the DepthWizard GPU backend.

Usage
-----
1. Install Modal: ``pip install modal``
2. Authenticate once: ``modal setup``
3. Deploy: ``modal deploy deploy_modal.py``

The ASGI app is served at the URL printed by ``modal deploy``.
Point the frontend ``VITE_API_URL`` at that URL and rebuild / redeploy.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"

app = modal.App("depthwizard")

# Persistent volume to cache Hugging Face model weights across cold starts.
model_cache = modal.Volume.from_name("depthwizard-model-cache", create_if_missing=True)

# Container image with all backend dependencies pre-installed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115,<1",
        "uvicorn[standard]>=0.30,<1",
        "python-multipart>=0.0.9,<1",
        "numpy>=1.26,<3",
        "Pillow>=10.3,<13",
        "rasterio>=1.3.10,<2",
        "torch>=2.2",
        "transformers>=4.45,<5",
        "safetensors>=0.4.3",
    )
    .copy_local_dir("backend", "/app/backend")
    .copy_local_dir("ml", "/app/ml")
    .env(
        {
            "DEPTHWIZARD_MODEL_ID": MODEL_ID,
            "DEPTHWIZARD_ARTIFACT_DIR": "/tmp/artifacts",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
        }
    )
)


# ---------------------------------------------------------------------------
# Pre-download model weights into the volume on build to avoid cold-start lag.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
)
def warm_model_cache() -> None:
    """Download model weights into the persistent volume."""
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    AutoImageProcessor.from_pretrained(MODEL_ID)
    AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model_cache.commit()
    print(f"✓ Model {MODEL_ID} cached.")


# ---------------------------------------------------------------------------
# ASGI web endpoint — exposes the existing FastAPI app on a public HTTPS URL.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": model_cache},
    timeout=300,
    container_idle_timeout=120,
    allow_concurrent_inputs=4,
    scaledown_window=300,
)
@modal.asgi_app()
def web():
    """Return the existing FastAPI application unchanged."""
    import sys
    sys.path.insert(0, "/app")

    # Allow any origin when deployed — the frontend is on a different domain.
    import os
    os.environ.setdefault(
        "DEPTHWIZARD_CORS_ORIGINS",
        "*",
    )

    from backend.app import create_app

    return create_app()
