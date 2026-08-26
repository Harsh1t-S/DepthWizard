# DepthWizard — Current State & Handover Document

**Repository:** https://github.com/Harsh1t-S/DepthWizard  
**Workspace:** `D:\Downloads\SIH26175`  
**Frontend Deployment:** `https://depth-wizard.vercel.app` (Vercel)  
**Backend Deployment:** `https://huggingface.co/spaces/MrT0nyStark/depthwizard-backend` (Hugging Face Space)  
**Direct API URL:** `https://mrt0nystark-depthwizard-backend.hf.space`  

---

## 1. What We Are Trying to Achieve

DepthWizard is an ISRO/SIH26175 prototype for monocular depth estimation on aerial and satellite imagery.

### Core System:
1. **Backend (Python / FastAPI / PyTorch)**:
   - Uses **Depth Anything V2** (`depth-anything/Depth-Anything-V2-Base-hf`).
   - Accepts RGB image / GeoTIFF upload via multipart form at `POST /api/analyze`.
   - Optional reference DEM / sparse GCP calibration to estimate metric DSM.
   - Computes relative depth grid, saves artifacts (`depth.png`, `depth.npy`, `original.png`), and serves them at `GET /artifacts/{job_id}/{filename}`.
   - Provides synthetic demo endpoint `GET /api/demo` and health check `GET /api/health`.

2. **Frontend (React 19 / Vite / Three.js / React Three Fiber)**:
   - Hosted 24/7 on **Vercel** (`depth-wizard.vercel.app`).
   - Connects to backend via `VITE_API_URL` environment variable.
   - Interactive 3D terrain viewer rendering a solid 3D diorama block with satellite RGB texture, bilinear subpixel sampling, multi-level Gaussian smoothing, height exaggeration controls, top-down preset, and elevation point inspection.

### The Hosting Goal & Constraints:
- **100% Free 24/7 Cloud Hosting** (accessible anywhere from any device, even when the local laptop is turned off).
- **Zero Payment Methods / Credit Cards**: The user does not want to enter credit cards (rules out paid Modal tier, paid Docker spaces, or paid GPU instances).

---

## 2. What Has Been Completed & Verified

1. **Frontend 3D Viewer Overhaul (`frontend/src/components/TerrainViewer.tsx`)**:
   - Fixed "janky lines" and stepped terracing using **subpixel bilinear sampling** of the depth grid.
   - Added **multi-level edge-preserving Gaussian smoothing filter** (`None`, `Subtle`, `Balanced`, `Smooth`).
   - Added **solid 3D diorama block base** (perimeter skirt walls + bottom cap).
   - Reduced vertical exaggeration factor from `1.35` down to `0.4`–`0.6` to kill the "tall pillar" bug.
   - Added top-down camera preset and auto-rotate cinematic controls.
   - Added visible texture loading states (loading spinner, loaded, failed with retry button).
   - Frontend passes `tsc -b && vite build` cleanly.

2. **Local Backend (`backend/app.py`, `ml/depth_anything.py`)**:
   - 19/19 Python unit tests pass (`pytest tests/`).

3. **Vercel Frontend**:
   - Successfully deployed to `https://depth-wizard.vercel.app`.
   - Configured with `VITE_API_URL = https://mrt0nystark-depthwizard-backend.hf.space`.

---

## 3. The Current Problem We Are Facing

### The Problem:
When the Vercel frontend calls the Hugging Face Space backend (`POST /api/analyze` or `GET /api/health`), it gets `503 Service Unavailable` or `"Your space is in error"`.

### Root Cause Analysis on Hugging Face Spaces:
1. The Space `MrT0nyStark/depthwizard-backend` was created using the **Gradio SDK** with **ZeroGPU (Free)** hardware.
2. In Hugging Face Gradio Spaces, Hugging Face expects an `app.py` that runs Gradio. We mounted our FastAPI app using `gr.mount_gradio_app(fastapi_app, demo, path="/")`.
3. **Issue 1 — ZeroGPU Startup Requirement**:
   - ZeroGPU requires `import spaces` on line 1 of `app.py` before any other module (Torch, Gradio, OS).
   - ZeroGPU scans the module during container startup for a function decorated with `@spaces.GPU`. If it doesn't detect one connected to a Gradio event, it throws `No @spaces.GPU function detected during startup` and halts.
4. **Issue 2 — OpenAPI / Pydantic Schema Parsing Conflict in Gradio**:
   - When Gradio starts, its internal route handler calls `get_api_info()` and `json_schema_to_python_type` on all mounted routes.
   - FastAPI's multipart upload endpoint (`/api/analyze` with `UploadFile`) outputs OpenAPI schemas with boolean attributes (`additionalProperties: true`).
   - `gradio_client.utils.json_schema_to_python_type` crashes on boolean schemas with `TypeError: argument of type 'bool' is not iterable` in `gradio_client/utils.py:863`.
   - This unhandled 500 error during the initial ASGI health check causes the Uvicorn/Gradio server to exit immediately (`Application startup complete -> Shutting down`).

---

## 4. Current File Implementations

### `app.py` (Current Hugging Face Entrypoint)
```python
import spaces  # Line 1 for ZeroGPU
import os
import io
import numpy as np
from PIL import Image
import gradio as gr

os.environ.setdefault("DEPTHWIZARD_MODEL_ID", "depth-anything/Depth-Anything-V2-Base-hf")
os.environ.setdefault("DEPTHWIZARD_MAX_INPUT_SIZE", "1024")
os.environ.setdefault("DEPTHWIZARD_MAX_DECODED_PIXELS", "50000000")
os.environ.setdefault("DEPTHWIZARD_ARTIFACT_DIR", "/tmp/artifacts")
os.environ.setdefault("DEPTHWIZARD_CORS_ORIGINS", "*")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")

from backend.app import create_app
from backend.model import get_depth_estimator

@spaces.GPU(duration=120)
def predict_depth_zerogpu(image: Image.Image) -> Image.Image:
    if image is None:
        return None
    estimator = get_depth_estimator()
    depth, _, _ = estimator.predict(image, quality_mode="fast")
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(depth_norm)

fastapi_app = create_app()

with gr.Blocks(title="DepthWizard Backend API", show_api=False) as demo:
    gr.Markdown("# 🛰️ DepthWizard Backend API (ZeroGPU Active)")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Test Image")
            btn = gr.Button("Test Depth on ZeroGPU", variant="primary")
        with gr.Column():
            output_img = gr.Image(type="pil", label="Depth Map Output")
    btn.click(fn=predict_depth_zerogpu, inputs=input_img, outputs=output_img, show_api=False)

app = gr.mount_gradio_app(fastapi_app, demo, path="/", show_api=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

### `README.md` (Hugging Face YAML Frontmatter on Space repo)
```yaml
---
title: DepthWizard Backend
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---
```

---

## 5. Potential Solutions for the Next Assistant to Evaluate

1. **Option A: Pure FastAPI on Hugging Face (Custom / Gradio bypass)**:
   - In Gradio 4/5, if `gr.mount_gradio_app` continues to fail schema inspection on `/openapi.json`, set `fastapi_app.openapi_url = None` or override the OpenAPI generator so Gradio cannot introspect multipart schemas.
   - Or mount FastAPI as a pure Starlette `Mount("/api", app=fastapi_app)` on `demo.app`.

2. **Option B: Switch Space Hardware to Free CPU Basic**:
   - In HF Space Settings, if hardware is switched from ZeroGPU to standard CPU (2 vCPU · 16 GB RAM), `spaces` is no longer required and standard `app.py` runs without ZeroGPU decorators or constraints.

3. **Option C: Alternative Free 24/7 Hosting for FastAPI**:
   - **Render.com** (Free Web Service tier, no credit card required, connects directly to GitHub repo, runs `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`).
   - **Koyeb** (Free Eco tier, no credit card required).

---

## 6. Helpful Commands

- Run unit tests locally: `python -m pytest tests/`
- Build frontend locally: `cd frontend && npm run build`
- Push clean commit to HF Space:
  ```powershell
  git checkout --orphan hf-clean
  git reset
  git add README.md app.py requirements.txt backend/ ml/
  git commit -m "deploy update"
  git push hf hf-clean:main --force
  git checkout -f main
  git branch -D hf-clean
  ```
