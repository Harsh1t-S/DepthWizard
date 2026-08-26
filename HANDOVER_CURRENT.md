# DepthWizard — Current State & Handover

**Repository:** https://github.com/Harsh1t-S/DepthWizard
**Workspace:** `D:\Downloads\SIH26175`
**Branch:** `main`
**Last updated:** 2026-08-26

---

## 1. The problem statement (recovered — read this first)

**SIH26175 — "DepthWizard: Single-View Height Estimation and 3D Flythrough"**
Organisation: Indian Space Research Organisation (ISRO), Department of Space.
Theme: Miscellaneous. Source: https://sih.gov.in/sih2026PS

### What is required

Build an end-to-end pipeline turning single-view optical RGB remote-sensing
images into high-precision elevation maps, supporting **both**:

- **Non-georeferenced** RGB (PNG/JPG) → **Relative** Digital Surface Model (rDSM).
- **Georeferenced** RGB (GeoTIFF) → **Absolute** DSM with metric height values.

Use a pre-trained monocular depth backbone. For georeferenced imagery, a
low-resolution DEM (e.g. SRTM 30 m) or a limited set of GCPs may map
scale-agnostic depth to absolute elevation. Then project the original optical
image onto a generated 3D terrain mesh and integrate with a rendering engine
(Unity, Three.js, or Babylon.js).

> "The interface should support **seamless first-person navigation** and analysis
> of structural heights and slopes from arbitrary aerial perspectives."

> "Build an immersive, preferably interactive, rendering pipeline that converts
> the optical texture and derived depth map into a navigable 3D environment
> **deployable as a standalone application**."

### Evaluation criteria — 50 / 50

| Weight | Criterion |
| --- | --- |
| 50% | **DSM estimation accuracy** — RMSE, MAE, correlation against LiDAR or reference data, including **performance stability across urban, sparse, hilly, and forested landscapes** |
| 50% | **Visualization** — projection accuracy, visual fidelity, **navigability of the 3D flythrough**, interface intuitiveness, software stability, and **successful standalone deployment** |

### Expected deliverable

A fully integrated software suite with complete source code and technical
documentation, deployable as a unified module containing:

- **Elevation estimation module** — accepts PNG/JPG/TIFF, outputs a high-fidelity
  DSM in a standard geospatial format.
- **Interactive visualization platform** — users upload imagery, visualize
  reconstructed terrain, and **validate estimated height values against
  reference datasets**.

### Dataset notes

- Reference repository: https://github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026
  (as of 2026-08-25 it contained only a README; recheck before every test with
  `python scripts/check_official_sac_data.py`).
- Any openly available high-resolution remote-sensing dataset may be used for
  development. SRTM 30 m is explicitly permitted as a calibration reference.
- **Final evaluation uses ISRO RGB-band optical satellite imagery.**

### Two consequences that should drive planning

1. **First-person navigation is a hard requirement**, not a stretch goal. The
   current viewer is orbit-only.
2. **Standalone deployment is named in the scoring rubric; cloud hosting is
   not.** Effort spent on cloud demos does not earn marks.

---

## 2. Strategic decision taken

Cloud hosting was **deprioritised** on 2026-08-26 in favour of the standalone
desktop application, because that is what the rubric scores.

### Why Hugging Face hosting was abandoned

The account `MrT0nyStark` is a **free personal account** (`isPro: false`,
verified via `/api/whoami-v2`). Per the current Hugging Face documentation:

> "Gradio and Docker Spaces run on compute and require a paid plan to create:
> PRO for personal accounts... Free personal accounts in good standing can still
> host up to 2 Gradio Spaces running on ZeroGPU."

Therefore, on this account:

- **Docker Spaces: impossible.** Requires PRO.
- **CPU Basic on a compute Space: impossible.** Same restriction.
- **ZeroGPU + Gradio: possible** (max 2 Spaces) — the only free compute path.
- **Static Spaces: free**, but no server-side compute.

An access token does not change this. Plan tier is an account property; no
token scope can unlock Docker Spaces.

The Space `MrT0nyStark/depthwizard-backend` currently sits in `CONFIG_ERROR`
with `"ZeroGPU is only available on Gradio SDK"` because its README declares
`sdk: docker` while its hardware request is still `zero-a10g`. Resolving this
is optional and secondary. See §6.

### Why the earlier Gradio attempts kept failing

Worth recording so nobody retries the same approach:

1. **ZeroGPU never provided a GPU to the API.** `@spaces.GPU` allocates a GPU
   only for that decorated function when a Gradio event invokes it. FastAPI
   routes mounted alongside a Gradio Blocks never enter that path.
2. `gradio_client.utils.json_schema_to_python_type` raised
   `TypeError: argument of type 'bool' is not iterable` on the multipart
   `UploadFile` OpenAPI schema. Largely a bug in the pinned `sdk_version:
   4.44.1`; modern Gradio 6.x would likely avoid it.
3. `gr.Blocks(show_api=False)` is invalid — `show_api` belongs to `launch()` and
   `mount_gradio_app`, not `Blocks`.
4. The old entrypoint did `depth, _, _ = estimator.predict(...)`, but `predict`
   returns a **2-tuple** `(ndarray, PredictionInfo)`. It would have raised even
   after startup succeeded.

---

## 3. What was completed and verified this session

### Standalone application (the current architecture)

One process. FastAPI serves the API, the artifacts, **and** the built UI on a
single origin, opened in a native OS window.

| File | Change |
| --- | --- |
| `desktop.py` | **New.** Launcher: free port, background uvicorn, health poll, pywebview window, browser fallback, `--no-window` headless mode |
| `backend/app.py` | `frontend_bundle()` locator; SPA mounted at `/`; API-only landing page registered only when no bundle exists |
| `frontend/src/lib/api.ts` | `API_BASE_URL` now defaults to **empty (same-origin)**; `absoluteBase()` resolves artifact URLs against the page origin |
| `frontend/vite.config.ts` | Dev/preview proxies for `/api` and `/artifacts` → `127.0.0.1:8000`, so relative paths work in dev too |
| `packaging/depthwizard.spec` | **New.** PyInstaller onedir spec |
| `packaging/fetch_weights.py` | **New.** Stages weights into `packaging/hf_cache` for offline builds |
| `requirements-desktop.txt` | **New.** `pywebview`, `pyinstaller` |

`VITE_API_URL` still overrides the base, so a split deployment remains possible.

### Verified working

- `pytest tests/` — **19 passed**.
- `npm run build` — clean (`tsc -b && vite build`).
- Single origin serves: `/` (SPA), `/assets/*`, `/api/health`, `/api/demo`,
  `/artifacts/*`, `/docs` — all HTTP 200.
- **Real inference end to end**: `ID6_Banner_San_Diego_PHR1B_20150724.jpg`,
  1920×1080, `POST /api/analyze` → **HTTP 200 in 2.4 s on CUDA**. Artifacts
  written to `%LOCALAPPDATA%\DepthWizard\artifacts\<job_id>\`
  (`original.png`, `depth.png`, `depth.npy`, `metrics.json`) and served back.
- **3D viewer renders in-browser** on the single origin: WebGL canvas active,
  mesh 192×132, RGB texture loaded, **zero console errors**.
- Model weights staged: **372 MB** in `packaging/hf_cache`.

### Earlier backend fixes (still in place, still valuable)

- **Artifact URLs were `http://` behind a TLS proxy.** `_public_artifact_base()`
  used `request.base_url`, so an HTTPS frontend blocked every artifact as mixed
  content — a likely cause of the "Texture failed" reports. Fixed via uvicorn
  `--proxy-headers --forwarded-allow-ips` plus a `DEPTHWIZARD_PUBLIC_BASE_URL`
  override.
- **CORS advertised `*` together with `allow_credentials=True`**, which the Fetch
  spec forbids. Credentials now auto-disable under a wildcard.
- `GET /` returned 404 on API-only deployments.

---

## 4. How to run it

Source checkout:

```powershell
.\.venv\Scripts\Activate.ps1
cd frontend; npm install; npm run build; cd ..
python desktop.py
```

Headless (no window, fixed port):

```powershell
$env:DEPTHWIZARD_PORT="8777"; python desktop.py --no-window
```

Build the distributable:

```powershell
python -m pip install -r requirements-desktop.txt
python packaging/fetch_weights.py          # 372 MB, once
python -m PyInstaller packaging/depthwizard.spec --noconfirm
```

Output lands in `dist/DepthWizard/`.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `DEPTHWIZARD_PORT` | Fixed port instead of an ephemeral one |
| `DEPTHWIZARD_FRONTEND_DIR` | Override the UI bundle location |
| `DEPTHWIZARD_ARTIFACT_DIR` | Job output directory (desktop default: `%LOCALAPPDATA%\DepthWizard\artifacts`) |
| `DEPTHWIZARD_PUBLIC_BASE_URL` | Force the artifact URL origin behind a proxy |
| `DEPTHWIZARD_MAX_INPUT_SIZE` | Inference bound (default 1024; 518 is the model's native resolution) |
| `HF_HOME` | Weight cache; set automatically to the bundled cache in a packaged build |

---

## 5. Next work, in priority order

### P0 — Not yet done, blocking the packaged deliverable

1. **Run the PyInstaller build and fix what breaks.** The spec is written but
   **has never been executed**. Expect trouble with Torch DLLs, `rasterio`
   GDAL data files, and `transformers` dynamic imports. This is the single
   biggest remaining unknown.
2. **First-person navigation.** Explicitly required by the problem statement and
   part of the 50% visualization score. Add pointer-lock + WASD with the eye
   height clamped to the sampled surface, `frameloop="always"`, near plane 0.01.
   Note the honest limitation: a heightfield has no building façades, so a
   ground-level camera shows stretched roof texture on vertical faces. Mitigate
   by shading near-vertical faces with a separate procedural material.

### P1 — Directly tied to the 50% accuracy score

3. **Scene-statistic / semantic-prior calibration.** The problem statement
   permits "scene-level statistics, semantic priors" as calibration inputs, not
   only DEM/GCP. This is the only route to metric height on non-georeferenced
   images, which the repo currently cannot do at all. Shadow-length plus solar
   azimuth is the classic approach.
4. **Landscape-diversity validation.** The rubric names urban, sparse, hilly and
   forested. Build an evaluation matrix across all four; currently only urban
   (ISPRS Potsdam, DC) is covered.
5. **In-UI validation view.** "Validate estimated height values against
   reference datasets" is a named deliverable. Metrics currently surface only
   via CLI and the API payload.

### P2 — Depth and mesh quality (detailed analysis earlier in the session)

6. **Mesh grid throws away detail.** `make_mesh_grid` caps `target_long_edge` at
   192 and rejects anything above 256 (`backend/artifacts.py:77`, `:82`).
   Downsampling a 1024 px depth map to 192 turns every building edge into a
   ramp — this is the root cause of the "rounded pillars" complaint. The blocker
   is JSON transport; the fix is to emit a 16-bit grayscale PNG plus
   `{min, max}` and decode it in the browser.
7. **Percentile clipping destroys roofs.** `normalize_for_preview` clamps at the
   2nd/98th percentile (`backend/artifacts.py:40`). On aerial imagery the top 2%
   *is* the tall buildings. Use 0.5/99.5 for display and full min/max for
   geometry.
8. **No edge-aware filtering.** The frontend 3×3 Gaussian
   (`TerrainViewer.tsx:111`) blurs *across* depth discontinuities. Replace with a
   guided filter using the RGB image as guide — depth edges then snap to image
   edges. Pure NumPy, no new dependency.
9. **Discontinuity culling.** The index builder connects rooftop vertices to
   ground vertices, producing vertical rubber sheets. Gate triangles on
   `|Δh| < threshold`.
10. **Disparity is treated as height.** Depth Anything V2 outputs *inverse*
    depth, and `calibration.py` fits a linear affine to it. Relief is therefore
    nonlinearly compressed at the tall end. Fit `h = a/(d + c) + b`, or invert
    first. Affects both realism and reported metrics.
11. **Render realism.** `<Environment preset="city" />`, ACES tone mapping, and
    SSAO are the three largest "looks real" wins. The directional light's shadow
    frustum is also unconfigured (`TerrainViewer.tsx:806`) — default ortho box
    `d=5` against a 5.6-unit terrain leaves corners unshadowed.
12. **Normal map from full-resolution depth.** Recovers micro-relief far beyond
    mesh resolution. Cheap, large visual gain.
13. **Model upgrade.** `apple/DepthPro` has the sharpest boundaries of any open
    model; `Depth-Anything-V2-Large-hf` is a drop-in improvement.

### Explicitly out of scope

Ground-level photo input and 360° equirectangular panoramas. The problem
statement covers remote-sensing imagery only.

---

## 6. Optional: reviving the cloud demo

Only worth doing after P0/P1. The only free path is **Gradio + ZeroGPU**:

- Set `sdk: gradio` and a current `sdk_version` in the Space README frontmatter,
  and switch hardware away from `zero-a10g` only if moving off ZeroGPU.
- `import spaces` must precede any Torch import.
- Route inference through a module-level `@spaces.GPU(duration=...)` function
  using a global estimator — the canonical ZeroGPU pattern. A public
  `set_depth_estimator()` seam in `ml/depth_anything.py` would let the entrypoint
  install a GPU-wrapped estimator without touching private state.
- Pin modern Gradio to avoid the `json_schema_to_python_type` crash.
- ZeroGPU cannot be tested locally; every iteration is push, wait for build,
  read logs.

`scripts/deploy_hf.ps1` builds a clean orphan commit and force-pushes it, using
`deploy/hf-space/README.md` as the Space README. It restores the working branch
in a `finally` block. Change that README's frontmatter to `sdk: gradio` before
reusing it.

---

## 7. Security note

Two Hugging Face access tokens were shared in plaintext chat during this
session, and one is stored unencrypted in `.git/config` as part of the `hf`
remote URL. **Rotate both** at https://huggingface.co/settings/tokens, then
reset the remote without embedded credentials:

```powershell
git remote set-url hf https://huggingface.co/spaces/MrT0nyStark/depthwizard-backend
```

---

## 8. Scientific constraints (unchanged — do not relax)

- RGB only → label output **Relative Depth / Relative DSM**. Never claim
  absolute elevation.
- Metric DSM requires a DEM, DSM, GCPs, or another elevation reference.
- Fitting against the full ground-truth DSM is **benchmark calibration /
  feasibility evaluation only**, never deployable accuracy.
- The reconstruction is a **2.5D heightfield**, not photogrammetric 3D. It
  cannot represent façades, overhangs, or truly vertical walls.
- Do not fabricate accuracy claims.

---

## 9. Useful commands

```powershell
python -m pytest tests/                 # 19 tests
cd frontend; npm run build              # tsc -b && vite build
python desktop.py                       # standalone app
python scripts/check_official_sac_data.py   # recheck the ISRO reference repo
```

Note: this workspace has a `PYTHONPATH` entry (`D:\claude-tools\pypkgs`) that
shadows the venv's `pydantic` and breaks pytest collection. Clear it for the
command if collection errors appear.
