# DepthWizard Handover

Repository: https://github.com/Harsh1t-S/DepthWizard  
Workspace: `D:\Downloads\SIH26175`  
Branch: `main`  
Latest pushed commit: `9d9072e` — `feat: improve aerial depth quality and benchmarking`

## Current state

DepthWizard is an SIH26175/ISRO prototype that supports:

- RGB/GeoTIFF upload
- Depth Anything V2 inference with CUDA/CPU fallback
- Relative depth output
- Optional aligned DSM evaluation
- Affine calibration with direction selection
- Pearson correlation, MAE and RMSE
- Depth previews and downloadable artifacts
- React/Three.js interactive terrain viewer
- Fast and quality inference modes
- FastAPI backend and Vite frontend

Previous verification:

- Python tests: 19 passed
- Frontend production build passed
- Local frontend: `http://127.0.0.1:5173`
- Local backend: `http://127.0.0.1:8000`
- API documentation: `http://127.0.0.1:8000/docs`

Run commands should be confirmed from the README/package files before use.

## Scientific constraints

Never claim arbitrary RGB-only results are absolute elevation.

- RGB only: label as **Relative Depth / Relative DSM**
- Metric DSM requires DEM, DSM, GCPs or another elevation reference
- Full-ground-truth affine fitting is only **benchmark calibration / feasibility evaluation**
- The current reconstruction is a 2.5D heightfield, not true photogrammetric 3D

## Immediate viewer problem

The current 3D model looks like tall rounded pillars and does not visually resemble the uploaded image.

Diagnosis:

1. The screenshot is showing the blue/teal/yellow height palette, not the RGB texture.
2. Texture failures are silently swallowed in `frontend/src/components/TerrainViewer.tsx`.
3. Backend depth is percentile-normalized to `[0,1]`, then the frontend min/max-normalizes it again.
4. Frontend converts the full range to `1.35` scene units and applies default height exaggeration `1.35`, giving approximately `1.82` vertical units across a `5.6`-unit-wide surface.
5. This greatly exaggerates normal depth errors and creates tower-like structures.
6. The generic monocular model has a strong global perspective/row trend on overhead imagery.
7. The mesh is approximately `192×192`, smoothing small rooftops.
8. A heightfield cannot reconstruct building façades, overhangs or genuinely vertical walls.

There does not appear to be an image/depth orientation bug. The UV mapping is probably correct.

## Highest-priority fixes

Edit `frontend/src/components/TerrainViewer.tsx`:

1. Add visible texture states: loading, loaded and failed.
2. Add a retry button when texture loading fails.
3. Rename material choices clearly:
   - `RGB texture`
   - `Height colors`
4. Do not silently fall back to height colors while suggesting texture is active.
5. Reduce default height exaggeration to approximately `0.35–0.5`.
6. Add a top-down camera preset.
7. Increase mesh resolution to `256×256` if performance remains smooth.
8. Avoid frontend re-normalization when backend mesh heights are already normalized.
9. Optionally add a clearly labeled **Local relief visualization** that removes low-frequency/global depth trends for display only. Preserve and export the raw prediction unchanged.

Test the uploaded image again after these fixes.

## Model-quality direction

The current default is:

`depth-anything/Depth-Anything-V2-Base-hf`

Quality mode uses enhanced/tiled inference. This improves detail but cannot solve the domain mismatch between ordinary monocular photographs and nadir satellite imagery.

The most important accuracy experiment is:

- Evaluate the current model on several aligned aerial RGB/DSM tiles.
- Compare raw depth, inverted depth and locally detrended depth.
- Explore an aerial/nDSM-specialized model or fine-tuning strategy.
- Use SRTM/coarse DEM or sparse GCP samples for deployable metric calibration.
- Keep benchmark calibration against full DSM clearly separate.

Do not fabricate accuracy claims.

## Hosting research

Recommended immediate architecture:

- Frontend: Vercel
- GPU backend: Modal

### 1. Modal — recommended for quickest deployment

Why:

- Existing FastAPI application can be exposed using `@modal.asgi_app`
- Supports public HTTPS endpoints
- GPU containers scale to zero
- Starter plan currently advertises about `$30/month` of compute credit
- T4 is approximately `$0.59/hour`, excluding CPU/memory
- Requires relatively little adaptation

Implementation outline:

- Add a small Modal entrypoint
- Build/install backend dependencies in a Modal image
- Request a T4 GPU
- Cache Hugging Face model weights in a Modal Volume
- Return the existing FastAPI application with `@modal.asgi_app`
- Run `modal deploy`
- Point `VITE_API_URL` at the resulting Modal URL
- Confirm CORS allows the deployed frontend

Watch the approximately 150-second normal web-request window and model cold starts.

Official references:

- https://modal.com/pricing
- https://modal.com/docs/guide/webhooks
- https://modal.com/docs/guide/webhook-urls
- https://modal.com/docs/guide/webhook-timeouts

### 2. Hugging Face Docker Space

Good hackathon alternative:

- Docker Spaces support FastAPI and custom containers
- T4 small is approximately `$0.40/hour`
- Community GPU grants can be requested
- Paid hardware is billed while starting/running
- Use port `7860`
- Configure sleep when not demonstrating

References:

- https://huggingface.co/docs/hub/spaces-overview
- https://huggingface.co/docs/hub/spaces-gpus
- https://huggingface.co/docs/hub/spaces-sdks-docker

### 3. Cloud Run with L4

Better production option but more setup:

- Scales to zero
- L4 GPU support
- Requires container registry, IAM, GPU quota and external artifact storage
- Approximate minimum active cost is around `$0.90/hour` after GPU, minimum CPU and memory
- New Google Cloud accounts may receive trial credit

References:

- https://docs.cloud.google.com/run/docs/configuring/services/gpu
- https://cloud.google.com/run/pricing
- https://docs.cloud.google.com/run/quotas

### 4. RunPod Serverless

- Competitive raw GPU pricing
- Flex workers scale to zero
- More adaptation because serverless normally expects a worker-handler contract instead of an unchanged FastAPI server
- Regular Pods are easier but stay billable while running

Reference:

- https://docs.runpod.io/serverless/pricing

Do not use Vercel, Railway or Render for the CUDA inference process. Vercel is appropriate for the frontend; the others can host lightweight CPU services but are not the preferred GPU inference target.

## Next work order

1. Pull `main` and inspect current status without discarding user changes.
2. Fix texture visibility and excessive vertical scaling.
3. Run frontend build/typecheck.
4. Test one real uploaded image through the complete local flow.
5. Add Modal deployment files with minimal changes.
6. Deploy the backend after the user authenticates Modal locally.
7. Deploy the frontend to Vercel and set its backend URL.
8. Verify upload → inference → depth → textured 3D over the public URLs.
9. Commit and push the completed changes.

Keep the implementation economical and focused on P0 functionality. Do not add authentication, billing, queues or unrelated infrastructure.