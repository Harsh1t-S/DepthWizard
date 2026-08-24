# DepthWizard — SIH26175

DepthWizard is a local, GPU-aware demonstration and evaluation workbench for
monocular depth on aerial imagery. A React/Vite interface calls a FastAPI
service, which runs Depth Anything V2 and writes inspectable artifacts for each
job. An optional, aligned digital surface model (DSM) enables a benchmark-only
calibration and error report.

> **Scientific scope:** the model predicts **relative monocular depth**, not
> metric elevation. Relative depth can order scene structure, but a single RGB
> image does not determine absolute scale or vertical datum. DepthWizard only
> produces a metric-looking DSM after fitting an affine mapping to supplied
> ground truth. That full-ground-truth fit is useful for feasibility analysis;
> it is non-deployable and must not be reported as accuracy on unseen data.

No accuracy result is bundled or claimed by this repository.

## Architecture

```text
Browser (React + Vite, :5173)
              |
              | JSON / multipart HTTP
              v
FastAPI service (:8000) ---- GET /api/demo (precomputed synthetic fixture)
              |
              +---- Depth Anything V2 Small ---- relative depth
              |
              +---- optional aligned DSM ---- affine fit + benchmark metrics
              |
              `---- configured artifact root/<job_id> + /artifacts/... URLs
```

- `frontend/` contains the Vite user interface.
- `backend/` contains the API and job-artifact handling.
- `ml/` contains inference, calibration, metrics, and rendering logic.
- `scripts/` contains command-line feasibility and evaluation entry points.
- `data/` is intentionally untracked apart from placement guidance.

## Prerequisites

The supported setup is Windows 10/11 with PowerShell, Python **3.11 or 3.12**,
and Node.js **20.19+ or 22.12+** with npm (required by the pinned Vite 7 toolchain).
Python 3.14 is not recommended because compatible PyTorch wheels may be
unavailable. Live inference can run on CPU, but an NVIDIA RTX GPU (including an
RTX 4050) is substantially more practical.

For GPU use, install a current NVIDIA driver and a CUDA-enabled PyTorch build
compatible with that driver. A separate CUDA Toolkit is normally unnecessary
for a prebuilt PyTorch wheel. DepthWizard detects CUDA through PyTorch and falls
back to CPU when CUDA is unavailable; it does not silently fabricate a result.

Check the environment after installing dependencies:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Install

From the repository root:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Set-Location frontend
npm install
Set-Location ..
```

If PyTorch reports `CUDA: False` on an NVIDIA system, install the appropriate
CUDA build of PyTorch for the installed driver, then repeat the check above.

## Model cache and offline operation

Live inference defaults to
`depth-anything/Depth-Anything-V2-Small-hf`. On the first live request,
Transformers downloads the model from Hugging Face, so network access and cache
space are required. Configure the process before starting the backend:

```powershell
$env:DEPTHWIZARD_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
$env:DEPTHWIZARD_MAX_INPUT_SIZE = "1024"
$env:DEPTHWIZARD_MAX_DECODED_PIXELS = "50000000"
$env:DEPTHWIZARD_ARTIFACT_DIR = "outputs/jobs"
$env:HF_HOME = "$PWD\.cache\huggingface"
```

For offline use, first populate the cache with one successful online inference,
then start a new PowerShell session with the same cache path and:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

Offline live inference cannot initialize if the selected model revision is not
already cached. `GET /api/demo` remains a precomputed synthetic demonstration;
it is not proof that the live model is installed. See `.env.example` for the
available settings. Settings are process environment variables; export them in
PowerShell before launching the service.

## Start

Use two PowerShell windows from the repository root:

```powershell
.\start-backend.ps1
```

```powershell
.\start-frontend.ps1
```

Then open <http://localhost:5173>. The API is at
<http://localhost:8000>; interactive FastAPI documentation is at
<http://localhost:8000/docs>. The equivalent direct commands are:

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
Set-Location frontend; npm run dev
```

The frontend uses `VITE_API_URL`, defaulting to `http://localhost:8000`.
Set it before `npm run dev` when the API runs elsewhere.

## Demo and live workflow

1. Start both services and select **Load Demo Scene**. This calls
   `GET /api/demo` and loads a **precomputed synthetic demonstration fixture**.
   It is explicitly labelled and is not live model output or a benchmark result.
2. Upload an image and analyze it to run live relative-depth inference. The first
   run may pause while model weights download.
3. Optionally supply a spatially aligned DSM to run affine calibration and
   calculate benchmark metrics. This uses the full valid ground truth for the
   fit, so the resulting numbers are diagnostic only.
4. Inspect/download the artifact URLs returned by the API. Each live response
   records its model, device, mode, timing, notices, and geospatial metadata.

The live endpoint never substitutes the synthetic demo for failed or unavailable
model inference.

## Command-line feasibility and evaluation

Run the commands from the repository root with the virtual environment active.
Use explicit output directories to keep runs separate.

Relative-depth feasibility (no metric accuracy claim):

```powershell
python scripts/test_depth.py data/sample/IMAGE.tif --output-dir outputs/feasibility --max-input-size 1024
```

Evaluation with a pixel-aligned ground-truth DSM:

```powershell
python scripts/evaluate_depth.py data/sample/IMAGE.tif data/sample/GROUND_TRUTH_DSM.tif --output-dir outputs/evaluation --max-input-size 1024
```

The feasibility command answers whether the pipeline can produce coherent
relative depth for an image. It cannot measure metric error without ground
truth. The evaluation command uses ground-truth correlation to choose the
relative-depth orientation, fits `height = a * oriented_depth + b` on valid DSM
pixels, and reports error after that same-scene fit. This removes orientation,
scale, and shift using ground truth and therefore does **not** demonstrate
zero-shot metric-depth deployment. For a defensible study, choose calibration
parameters on separate calibration tiles and evaluate on held-out tiles; the
provided full-GT command does not implement that protocol.

## ISPRS Potsdam and Vaihingen data

Obtain the datasets from their official source and comply with their terms.
They are not redistributed here. A convenient local layout is:

```text
data/
  isprs/
    potsdam/
      images/
      dsm/
    vaihingen/
      images/
      dsm/
  sample/
    README.md
```

This layout is organizational only: the CLI accepts explicit file paths and
does not automatically discover, pair, split, or batch-score ISPRS tiles. Pair
an image with the correct DSM tile, verify pixel alignment, resolution, CRS,
vertical units, and no-data values, and keep calibration and evaluation tiles
separate. Do not compare numbers across Potsdam/Vaihingen or preprocessing
variants without documenting those choices.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service/model/device status |
| `GET` | `/api/demo` | Precomputed synthetic fixture; no live inference |
| `POST` | `/api/analyze` | Multipart live inference; required `image`, optional `ground_truth_dsm` |
| `GET` | `/artifacts/{job_id}/{filename}` | Generated artifact download |

Example calls:

```powershell
curl.exe http://localhost:8000/api/health
curl.exe -F "image=@data/sample/IMAGE.tif" http://localhost:8000/api/analyze
curl.exe -F "image=@data/sample/IMAGE.tif" -F "ground_truth_dsm=@data/sample/GROUND_TRUTH_DSM.tif" http://localhost:8000/api/analyze
```

Live responses include `job_id`, `demo`, `precomputed`, `model`, `device`,
`mode`, `input`, `processing_time_seconds`, `geospatial`, `metrics`,
`calibration`, `depth_grid`, `urls`, `artifacts`, and `notices`.

## Outputs and GeoTIFF behavior

A run can write `original.png`, `depth.png`, and `depth.npy`. Runs with ground
truth can additionally write `ground_truth.png`, `error.png`, and
`metrics.json`. API artifacts are grouped by job ID; CLI files go to the
explicit `--output-dir`. Unless `DEPTHWIZARD_ARTIFACT_DIR` overrides it, the API
artifact root is `backend/artifacts/`.

When both calibration ground truth and a source image with a valid GeoTIFF CRS
and affine transform are available, the pipeline can also write
`calibrated_dsm.tif` using the source georeferencing. PNG previews and
`depth.npy` are visualization/numeric artifacts, not georeferenced metric
products. A calibrated GeoTIFF inherits horizontal metadata but its vertical
meaning is only as valid as the supplied DSM, alignment, units, and affine fit.

If both rasters carry usable georeferencing but their grids differ, DepthWizard
reprojects/resamples the DSM onto the image grid and records a notice. With
matching dimensions but incomplete georeferencing, it assumes pixel-for-pixel
alignment. As a last resort for differently sized ungeoreferenced rasters, it
performs a numeric resize and reports that fact; such results need especially
careful interpretation.

## Limitations

- Monocular relative depth is ambiguous in absolute scale and shift and is not
  a replacement for stereo, LiDAR, photogrammetry, or surveyed elevation.
- A full-GT affine fit is data leakage for deployment evaluation. Its metrics
  are same-scene, post-calibration diagnostics only.
- Domain shift, haze, shadows, seasonal change, roofs/trees, image channel
  composition, resolution, tiling, and resizing can materially affect output.
- Image/DSM misregistration, CRS or vertical-datum mismatch, and mishandled
  no-data pixels can dominate reported errors.
- GPU memory use rises with input size. Lower `--max-input-size` or
  `DEPTHWIZARD_MAX_INPUT_SIZE` if an RTX 4050 runs out of memory; CPU fallback is
  slower.
- Decoded inputs default to a 50,000,000-pixel safety limit, configurable with
  `DEPTHWIZARD_MAX_DECODED_PIXELS`. This is a decoded-pixel limit, not a file-size
  limit.
- The demo is synthetic and precomputed. It must not be presented as measured
  model accuracy or as a live-inference result.
