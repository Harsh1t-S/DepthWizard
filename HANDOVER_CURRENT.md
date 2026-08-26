# DepthWizard — Current State & Handover

**Repository:** https://github.com/Harsh1t-S/DepthWizard
**Workspace:** `D:\Downloads\SIH26175`
**Branch:** `main` — clean
**Last updated:** 2026-08-27

---

## 1. The problem statement

**SIH26175 — "DepthWizard: Single-View Height Estimation and 3D Flythrough"**
Indian Space Research Organisation (ISRO), Department of Space.
Source: https://sih.gov.in/sih2026PS

Turn single-view optical RGB remote-sensing images into high-precision
elevation maps, supporting both:

- **Non-georeferenced** RGB (PNG/JPG) → **Relative** DSM
- **Georeferenced** RGB (GeoTIFF) → **Absolute** DSM with metric heights

Calibration may use a low-resolution DEM (SRTM 30 m), GCPs, or
**"scene-level statistics, semantic priors"**. Project the optical image onto a
3D mesh and render with Unity, Three.js, or Babylon.js.

> "The interface should support **seamless first-person navigation**."

> "...a navigable 3D environment **deployable as a standalone application**."

### Scoring — 50 / 50

| Weight | Criterion |
| --- | --- |
| 50% | **DSM accuracy** — RMSE, MAE, correlation vs LiDAR, **stability across urban, sparse, hilly, forested** |
| 50% | **Visualization** — projection accuracy, visual fidelity, **flythrough navigability**, interface, stability, **standalone deployment** |

Final evaluation uses ISRO RGB optical satellite imagery.

---

## 2. Requirement status — all met

| Requirement | Status |
| --- | --- |
| Non-georeferenced RGB → relative DSM | Done |
| Georeferenced → **absolute metric DSM, no external file** | **Done** — shadow + solar geometry |
| Pre-trained monocular backbone | Done |
| Calibration via DEM / GCP | Done |
| Optical image projected onto 3D mesh | Done |
| Three.js rendering | Done |
| **Seamless first-person navigation** | Done — pointer-lock flythrough |
| **Standalone application** | Done — `DepthWizard.exe`, runs offline |
| DSM in standard geospatial format | Done — GeoTIFF with source CRS |
| Validate against reference datasets | Done — holdout panel in UI |

What remains is quality, not coverage.

---

## 3. Accuracy — measured, not claimed

Five DC landscapes, RGB orthophoto against 2020 LiDAR DSM.

### Deployable (shadow calibration; ground truth never used to calibrate)

| Scene | MAE (m) | RMSE (m) |
| --- | ---: | ---: |
| dc-mall | 6.40 | 8.08 |
| dc-waterfront | 7.13 | 9.04 |
| dc-residential | 7.37 | 9.34 |
| dc-urban | 8.30 | 12.01 |
| dc-rock-creek | 12.18 | 16.18 |

**MAE 6.40–12.18 m.** Benchmark fitting *against* the ground truth gives
6.38–12.19 m, so shadow calibration recovers essentially the same accuracy
without consulting the ground truth. Forested terrain is the weakest case by a
clear margin, roughly 2× worse than built-up.

Quality mode improves RMSE ~5.5% (12.008 → 11.451 on dc-urban) for 0.4 s. It is
now the interface default.

### Backbone comparison — closed, do not repeat

Six models, same scene, matched resolution:

| Model | MAE | RMSE |
| --- | ---: | ---: |
| **V2-Base (pipeline default)** | **6.574** | **8.915** |
| V2-Large | 6.388 | 8.596 |
| DA3MONO-LARGE | 6.668 | 9.086 |
| Distill-Any-Depth-Large | 6.730 | 9.029 |
| DA3-LARGE | 10.435 | 12.940 |
| DA3METRIC-LARGE | 10.595 | 13.276 |

Everything clusters at RMSE 8.9–9.1 except metric variants, ~45% worse — they
carry an absolute depth prior from ground-level scenes that does not transfer to
nadir imagery. **Swapping backbones is exhausted.** Full detail and reproduce
commands in `docs/MODEL_RESEARCH.md`.

Two traps recorded there: **`DA3-LARGE` is CC-BY-NC** (non-commercial — a real
risk for an ISRO submission), and installing `depth-anything-3` pulled 60+
packages and **downgraded numpy to 1.26, breaking rasterio's stated
`numpy>=2`**. Everything runs and all 19 tests pass, but that conflict is live
in the environment.

---

## 4. Architecture

One process. FastAPI serves the API, the artifacts, **and** the built UI on a
single origin, opened in a native OS window.

```
DepthWizard.exe
  └─ uvicorn on 127.0.0.1:<ephemeral>
       ├─ /                → built React SPA
       ├─ /api/analyze     → inference + calibration
       ├─ /artifacts/...   → job outputs
       └─ pywebview window (browser fallback)
```

### Display vs measurement — the key invariant

Several refinements improve how the 3D surface *looks*. **None touches exported
products.** `depth.npy`, the calibrated DSM GeoTIFF, and every metric use the
raw prediction. Verified: MAE and RMSE are bit-identical with refinement on and
off.

This separation exists because it was measured to matter. Applying the guided
filter to the prediction itself costs **2% RMSE at radius 8 and 15% at radius
32** — it transfers image albedo into geometry, so dark roads read as low and
bright roofs as high. It looks sharper and measures worse.

| Stage | Module | Effect |
| --- | --- | --- |
| Guided filter | `ml/refine.py` | Edge alignment 0.046 → 0.326 |
| Roof plane fitting | `ml/buildings.py` | 48 buildings found; planarity residual 0.326 → 0.020 (94%) |
| Anisotropic diffusion | `ml/refine.py` | Surfaces ~65% flatter |
| Wall extrusion | `TerrainViewer.tsx` | Vertical faces at height steps |
| Normal map | `backend/artifacts.py` | Lighting detail beyond mesh resolution |

### Transport

The mesh grid is **512 cells**, carried as a 16-bit PNG (360 KB) rather than a
JSON float array (which would be ~2 MB at that size). A coarse 192-cell JSON
array is still emitted as a fallback, so a client that ignores the encoded field
still renders. `normal.png` (462 KB) carries tangent-space normals derived from
the full-resolution prediction.

---

## 5. Known limitations — state these honestly

1. **Buildings are not architecturally correct.** Roofs are planar and walls are
   extruded, but close up a building is still a rounded volume with roof texture
   draped over it. Three causes: mesh resolution (fixed), the backbone's soft
   boundaries on nadir imagery (needs fine-tuning), and the fact that **a
   heightfield cannot represent a vertical wall** — one surface, one height per
   cell. The third is structural, not a bug.
2. **This is 2.5D, not photogrammetric 3D.** No façades, no overhangs.
3. **Shadow heights are above local ground, not a datum.** Nothing in one image
   fixes mean sea level. When an evaluation DSM is supplied a single constant
   aligns the datum; the offset and its meaning are recorded so relief agreement
   is never mistaken for absolute agreement. Without this the residential scene
   reported 46 m of error that was entirely datum difference.
4. **Forested terrain is ~2× worse** than built-up.
5. **Segmentation costs ~10 s** on a 1024 tile. Disable with
   `DEPTHWIZARD_FLATTEN_ROOFS=0` for a latency-sensitive demo.
6. **The packaged build could not read GeoTIFFs** until 2026-08-27. PyInstaller
   missed `rasterio.serde` and other dynamically imported submodules, so the exe
   started, served, and ran inference on a JPEG, then failed every GeoTIFF with
   `No module named 'rasterio.serde'` — silently removing the entire
   georeferenced path. Fixed by adding `collect_submodules` for rasterio and
   pyproj to the spec. **Any packaged-build check must use a GeoTIFF**; a JPEG
   passes straight through the gap.
7. **The staging cache is not the bundle.** `packaging/hf_cache` accumulates
   every checkpoint anyone evaluates with `HF_HOME` pointed at it — after the
   backbone comparison it held nine models and 10.5 GB, and the spec bundled all
   of it, taking the build from 3.3 GB to 11 GB. The spec now selects only
   `DEPTHWIZARD_MODEL_ID`. Prune the cache anyway if disk matters; only the
   default model is needed to ship.
8. **No SSAO.** `@react-three/postprocessing@3.1.0` requires `three >= 0.182`;
   the project is on `0.179.1`. Upgrading three risks the working viewer, so
   this was deliberately skipped. It remains the largest cheap realism win if
   someone wants to take the upgrade on.

---

## 6. Running it

```powershell
.\.venv\Scripts\Activate.ps1
cd frontend; npm install; npm run build; cd ..
python desktop.py
```

Headless: `$env:DEPTHWIZARD_PORT="8900"; python desktop.py --no-window`

Packaged build — **`dist\DepthWizard\DepthWizard.exe`**, not `build\`, which is
PyInstaller scratch with no Python runtime.

Current build verified 2026-08-27: 3.6 GB, native window opens, and a GeoTIFF
upload with ground truth returns `shadow_calibrated_dsm`, MAE 7.84 m / RMSE
11.45 m, a 512-cell mesh, a normal map, and a calibrated DSM GeoTIFF. Cold start
of the frozen bundle takes roughly 60-90 s; a check that gives up sooner reports
a connection refused that is not real.

```powershell
python -m pip install -r requirements-desktop.txt
python packaging/fetch_weights.py          # 372 MB, once
python -m PyInstaller packaging/depthwizard.spec --noconfirm
```

### Environment variables

| Variable | Purpose |
| --- | --- |
| `DEPTHWIZARD_PORT` | Fixed port |
| `DEPTHWIZARD_MODEL_ID` | Backbone selection |
| `DEPTHWIZARD_REFINE` / `_RADIUS` / `_EPSILON` | Guided filter (display-only) |
| `DEPTHWIZARD_FLATTEN_ROOFS` | Building segmentation (display-only) |
| `DEPTHWIZARD_FLATTEN_ITERATIONS` / `_KAPPA` | Diffusion (display-only) |
| `DEPTHWIZARD_ARTIFACT_DIR` | Job outputs |
| `DEPTHWIZARD_PUBLIC_BASE_URL` | Artifact origin behind a proxy |

### Test data

Five aligned RGB + LiDAR pairs in `data/sample/dc-*`. More:

```powershell
python scripts/download_dc_sample.py --scene rock-creek --size 1024
```

Presets: `downtown`, `mall`, `rock-creek`, `residential`, `waterfront`.

**ISPRS Potsdam is not needed** for evaluation — only for fine-tuning. The DC
ArcGIS services orthorectify on request, so each scene is ~4 MB rather than
13 GB. Potsdam password, if ever needed: `CjwcipT4-P8g`.

---

## 7. Next work

### P1 — the only real accuracy headroom
1. **Fine-tune Depth Anything V2 on aerial RGB/nDSM pairs.**
   [arXiv 2507.09681](https://arxiv.org/pdf/2507.09681) reports ~24% RMSE
   reduction — an order of magnitude more than any backbone swap measured here.
   Needs ISPRS Potsdam (13.3 GB, 5 cm GSD, aligned float32 DSMs), a regression
   head on the V2 encoder, and strict tile separation between calibration and
   held-out evaluation. A project, not an afternoon.

### P2 — visual fidelity
2. **True building extrusion from footprints.** Vectorise segment boundaries,
   fit each roof a polygon, extrude walls as real geometry. The current
   grid-based extrusion approximates this per cell.
3. **SSAO**, if the `three` upgrade is acceptable. See limitation 6.
4. **Validate on Indian imagery.** [Google Open Buildings 2.5D](https://sites.research.google/gr/open-buildings/temporal/)
   has CC-BY heights for India at ~4 m. These are *model-predicted*, not LiDAR —
   fine for validation, weak as a training target.

### Out of scope
Ground-level photos and 360° panoramas. The problem statement covers
remote-sensing imagery only.

---

## 8. Cloud hosting — deprioritised, and why

The account is a **free personal account** (`isPro: false`). Per current HF
docs: *"Gradio and Docker Spaces run on compute and require a paid plan to
create: PRO for personal accounts... Free personal accounts in good standing can
still host up to 2 Gradio Spaces running on ZeroGPU."*

- Docker Spaces: **impossible**
- CPU Basic on a compute Space: **impossible**
- ZeroGPU + Gradio: possible, max 2 — the only free compute path
- Static Spaces: free, no server-side compute

An access token does not change this; plan tier is an account property.

If revived, note that **ZeroGPU only allocates a GPU inside `@spaces.GPU`
functions invoked from Gradio events** — FastAPI routes mounted alongside never
get one. That is why the earlier attempt could not have worked regardless of the
errors it threw.

**Cloud hosting is not scored. Standalone deployment is.**

---

## 9. Bugs found and fixed — worth not rediscovering

1. **Satellite texture never applied.** `map` and `vertexColors` both change
   shader defines; the texture resolves asynchronously, so Three.js kept the
   program built for vertex colours. The 3D view silently showed the height ramp
   instead of the imagery.
2. **Artifacts 404'd while existing on disk.** Windows virtualizes
   `AppData\Local` for containerized apps, so a file resolves through
   `...\Packages\<app>\LocalCache\...` while its directory does not. Starlette's
   traversal guard compared them and rejected everything. Fixed with
   `follow_symlink`; traversal still blocked, verified against three attack
   patterns.
3. **Packaged app opened no window.** pywebview selects its backend at runtime,
   so nothing imports `webview.platforms.*` and PyInstaller collected none of
   them.
4. **Flythrough camera frozen.** Canvas renders on demand; toggling the
   `frameloop` prop after mount does not restart the loop, so `useFrame` never
   ran. Now driven by its own animation frame.
5. **Timezone bug in acquisition time.** `new Date().toISOString()` reinterpreted
   the value in the viewer's timezone — a Washington scene at 15:30 UTC entered
   from India arrived as 06:00 UTC, and calibration correctly refused with the
   sun 39.8° below the horizon.
6. **Shadow results mislabelled** as "Benchmark fit / not deployment" whenever
   ground truth was present, though shadow calibration never sees it.

### A caution about measurement

Roof flattening appeared broken and was not. The test measured **standard
deviation** over a roof patch, which cannot distinguish a flat roof from a
correctly tilted one — and the synthetic terrain had a slope, so a working plane
fit scored as zero change. Measuring **residual from the best-fit plane** showed
a 94% improvement. A "fix" was nearly applied to working code.

Separately, the edge-alignment proxy in `compare_depth_backbones.py` ranked
Metric-Outdoor **first** while ground truth ranked it **last**. It measures
visual fidelity, not accuracy. **Never promote a model on it.**

---

## 10. Security

Two HF access tokens were shared in plaintext, and one is stored unencrypted in
`.git/config` as part of the `hf` remote. **Rotate both** at
https://huggingface.co/settings/tokens, then:

```powershell
git remote set-url hf https://huggingface.co/spaces/MrT0nyStark/depthwizard-backend
```

---

## 11. Scientific constraints — do not relax

- RGB only → label output **Relative Depth / Relative DSM**.
- Metric DSM requires a reference: DEM, GCPs, or shadow geometry.
- Fitting against the full ground-truth DSM is **benchmark / feasibility only**,
  never deployable accuracy.
- The reconstruction is a **2.5D heightfield**, not photogrammetric 3D.
- Shadow-derived heights are **above local ground**, not a vertical datum.
- Do not fabricate accuracy claims.

---

## 12. Commands

```powershell
python -m pytest tests/                          # 19 tests
cd frontend; npm run build                       # tsc -b && vite build
python desktop.py                                # standalone app
python scripts/compare_depth_backbones.py --image <scene>   # sharpness proxy only
python scripts/check_official_sac_data.py        # recheck the ISRO reference repo
```

`PYTHONPATH` in this workspace includes `D:\claude-tools\pypkgs`, which shadows
the venv's `pydantic` and breaks pytest collection. Clear it if collection
errors appear.
