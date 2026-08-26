"""Hugging Face Spaces Entrypoint for DepthWizard (100% Free Gradio/Python SDK).

Mounts the FastAPI backend so all endpoints (/api/analyze, /api/demo, /api/health, /artifacts)
are served directly from this 24/7 free space without Docker.
"""

from __future__ import annotations

import os
import gradio as gr
from backend.app import create_app

# Configure backend environment for Hugging Face Spaces
os.environ.setdefault("DEPTHWIZARD_MODEL_ID", "depth-anything/Depth-Anything-V2-Base-hf")
os.environ.setdefault("DEPTHWIZARD_MAX_INPUT_SIZE", "1024")
os.environ.setdefault("DEPTHWIZARD_MAX_DECODED_PIXELS", "50000000")
os.environ.setdefault("DEPTHWIZARD_ARTIFACT_DIR", "/tmp/artifacts")
os.environ.setdefault("DEPTHWIZARD_CORS_ORIGINS", "*")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")

# Create the existing FastAPI application
fastapi_app = create_app()

# Simple dashboard view at root
with gr.Blocks(title="DepthWizard API") as demo:
    gr.Markdown("# 🛰️ DepthWizard Backend API")
    gr.Markdown(
        "This Hugging Face Space hosts the **DepthWizard FastAPI Service**.\n\n"
        "- **Status**: 🟢 Online 24/7\n"
        "- **Health Check**: [`/api/health`](/api/health)\n"
        "- **Swagger Docs**: [`/docs`](/docs)\n"
        "- **Frontend**: Connect your Vercel frontend by setting `VITE_API_URL` to this Space URL."
    )

# Mount FastAPI app so all /api/* routes are handled by our FastAPI backend
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
