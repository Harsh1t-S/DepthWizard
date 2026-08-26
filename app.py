"""Non-Docker entrypoint for the DepthWizard backend.

The Hugging Face Space runs the Dockerfile and starts ``backend.app:app``
directly, so this module is only a convenience for hosts that expect a
top-level ``app`` (``uvicorn app:app``).

Gradio is intentionally not used here. ZeroGPU only allocates a GPU inside
``@spaces.GPU`` functions invoked from Gradio events, so a FastAPI route
mounted beside a Gradio Blocks never receives one, and Gradio's client-side
schema walker raises ``TypeError: argument of type 'bool' is not iterable``
when it introspects this API's multipart ``UploadFile`` OpenAPI schema.
"""

from __future__ import annotations

import os

os.environ.setdefault("DEPTHWIZARD_MODEL_ID", "depth-anything/Depth-Anything-V2-Base-hf")
os.environ.setdefault("DEPTHWIZARD_MAX_INPUT_SIZE", "518")
os.environ.setdefault("DEPTHWIZARD_ARTIFACT_DIR", "/tmp/artifacts")
os.environ.setdefault("DEPTHWIZARD_CORS_ORIGINS", "*")

from backend.app import app  # noqa: E402  (env must be set before app import)

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
