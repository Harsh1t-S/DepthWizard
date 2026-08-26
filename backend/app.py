"""FastAPI entry point for the DepthWizard prototype."""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .artifacts import default_artifact_root
from .demo import demo_response
from .model import (
    DEFAULT_MODEL_ID,
    ModelInferenceError,
    ModelLoadError,
    is_depth_estimator_loaded,
    select_device,
)
from .pipeline import analyze_bytes


API_VERSION = "0.1.0"


def _cors_origins() -> list[str]:
    configured = os.getenv("DEPTHWIZARD_CORS_ORIGINS", "")
    if configured.strip():
        origins = [origin.strip().rstrip("/") for origin in configured.split(",")]
        return [origin for origin in origins if origin]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://localhost:5173",
        "https://127.0.0.1:5173",
    ]


def frontend_bundle() -> Path | None:
    """Locate the built UI, or None when only the API should be served.

    Checked in order: an explicit override, the PyInstaller extraction root for
    a packaged build, and the in-repo Vite output for a source checkout.
    """

    configured = os.getenv("DEPTHWIZARD_FRONTEND_DIR", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        candidates.append(Path(bundled_root) / "frontend_dist")
    candidates.append(Path(__file__).resolve().parent.parent / "frontend" / "dist")

    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate.resolve()
    return None


def _max_upload_bytes() -> int:
    raw = os.getenv("DEPTHWIZARD_MAX_UPLOAD_MB", "64")
    try:
        megabytes = int(raw)
    except ValueError as exc:
        raise ValueError("DEPTHWIZARD_MAX_UPLOAD_MB must be an integer") from exc
    return max(megabytes, 1) * 1024 * 1024


async def _read_upload(upload: UploadFile, label: str) -> bytes:
    maximum = _max_upload_bytes()
    data = await upload.read(maximum + 1)
    if len(data) > maximum:
        raise HTTPException(
            status_code=413,
            detail=f"{label} exceeds the configured {maximum // (1024 * 1024)} MB limit",
        )
    if not data:
        raise HTTPException(status_code=422, detail=f"{label} is empty")
    return data


def _public_artifact_base(request: Request) -> str:
    # Behind a TLS-terminating proxy (Hugging Face Spaces, Cloud Run) the origin
    # scheme only reaches request.base_url when uvicorn runs with
    # --proxy-headers --forwarded-allow-ips. The explicit override is the escape
    # hatch for hosts that strip X-Forwarded-Proto; without a correct scheme the
    # artifact URLs are http:// and an HTTPS frontend blocks them as mixed content.
    configured = os.getenv("DEPTHWIZARD_PUBLIC_BASE_URL", "").strip()
    if configured:
        return f"{configured.rstrip('/')}/artifacts"
    return f"{str(request.base_url).rstrip('/')}/artifacts"


def create_app(artifact_root: Path | None = None) -> FastAPI:
    """Create an app; injectable artifact storage keeps API tests isolated."""

    root = Path(artifact_root or default_artifact_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    application = FastAPI(
        title="DepthWizard API",
        version=API_VERSION,
        description=(
            "Real Depth Anything V2 inference with deployment-reference or "
            "benchmark-only calibration and geospatial exports."
        ),
    )
    application.state.artifact_root = root
    origins = _cors_origins()
    # A wildcard origin and credentialed CORS are mutually exclusive per the
    # Fetch spec: browsers reject Access-Control-Allow-Origin: * when
    # Allow-Credentials is true. This API is unauthenticated, so drop
    # credentials rather than the wildcard the deployed frontend relies on.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials="*" not in origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    bundle = frontend_bundle()

    # Registered only for API-only deployments. An explicit route always wins
    # over the catch-all StaticFiles mount, so defining it unconditionally would
    # hide the real UI behind this placeholder.
    if bundle is None:

        @application.get("/", include_in_schema=False)
        async def index() -> HTMLResponse:
            return HTMLResponse(
                "<!doctype html><meta charset=utf-8>"
                "<title>DepthWizard API</title>"
                "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:34rem;"
                "margin:4rem auto;padding:0 1.5rem;background:#0b1620;color:#dce8f0}"
                "a{color:#6ad19f}code{background:#132434;padding:.15em .4em;border-radius:4px}</style>"
                "<h1>DepthWizard API</h1>"
                "<p>Monocular depth on aerial imagery (SIH26175). This host serves the "
                "backend only; the user interface is deployed separately.</p>"
                "<ul>"
                '<li><a href="/api/health">/api/health</a> — service and device status</li>'
                '<li><a href="/api/demo">/api/demo</a> — precomputed synthetic fixture</li>'
                '<li><code>POST /api/analyze</code> — multipart image upload</li>'
                '<li><a href="/docs">/docs</a> — OpenAPI reference</li>'
                "</ul>"
            )

    @application.get("/api/health", tags=["system"])
    async def health() -> dict[str, Any]:
        # This checks hardware and singleton state, but intentionally never calls
        # get_depth_estimator() and therefore never downloads/loads model weights.
        return {
            "status": "ok",
            "service": "DepthWizard API",
            "version": API_VERSION,
            "model": os.getenv("DEPTHWIZARD_MODEL_ID", DEFAULT_MODEL_ID),
            "device": select_device(),
            "model_loaded": is_depth_estimator_loaded(),
            "live_inference": "real_model_only",
        }

    @application.get("/api/demo", tags=["analysis"])
    async def demo(request: Request) -> dict[str, Any]:
        return await run_in_threadpool(
            demo_response,
            _public_artifact_base(request),
            request.app.state.artifact_root,
        )

    @application.post("/api/analyze", tags=["analysis"])
    async def analyze(
        request: Request,
        image: UploadFile = File(...),
        ground_truth_dsm: UploadFile | None = File(default=None),
        dsm: UploadFile | None = File(default=None),
        reference_dem: UploadFile | None = File(default=None),
        gcps: UploadFile | None = File(default=None),
        gcp_sampling: str = Form(default="bilinear"),
        quality_mode: str = Form(default="fast"),
        acquisition_time: str | None = Form(default=None),
    ) -> dict[str, Any]:
        if ground_truth_dsm is not None and dsm is not None:
            raise HTTPException(
                status_code=422,
                detail="Supply only ground_truth_dsm (preferred) or dsm, not both",
            )
        if reference_dem is not None and gcps is not None:
            raise HTTPException(
                status_code=422,
                detail="Supply only one deployment calibration reference: reference_dem or gcps",
            )
        selected_dsm = ground_truth_dsm or dsm
        image_bytes = await _read_upload(image, "image")
        ground_truth_bytes: bytes | None = None
        if selected_dsm is not None:
            ground_truth_bytes = await _read_upload(selected_dsm, "ground_truth_dsm")
        reference_dem_bytes: bytes | None = None
        if reference_dem is not None:
            reference_dem_bytes = await _read_upload(reference_dem, "reference_dem")
        gcps_bytes: bytes | None = None
        if gcps is not None:
            gcps_bytes = await _read_upload(gcps, "gcps")

        work = partial(
            analyze_bytes,
            image_bytes=image_bytes,
            image_filename=image.filename or "image",
            ground_truth_bytes=ground_truth_bytes,
            ground_truth_filename=(selected_dsm.filename if selected_dsm else None),
            reference_dem_bytes=reference_dem_bytes,
            reference_dem_filename=(reference_dem.filename if reference_dem else None),
            gcps_bytes=gcps_bytes,
            gcps_filename=(gcps.filename if gcps else None),
            gcp_sampling=gcp_sampling,
            quality_mode=quality_mode,
            acquisition_time=acquisition_time,
            artifact_root=request.app.state.artifact_root,
            public_artifact_base_url=_public_artifact_base(request),
        )
        try:
            return await run_in_threadpool(work)
        except (ModelLoadError, ModelInferenceError) as exc:
            # There is intentionally no fake live result when model loading or
            # inference fails. The explicit synthetic fixture remains /api/demo.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # follow_symlink makes StaticFiles compare an absolute path against the
    # served directory instead of a fully resolved one.
    #
    # Windows virtualizes AppData\Local for containerized apps, so a file under
    # the artifact root resolves through
    # AppData\Local\Packages\<app>\LocalCache\... while the directory itself
    # does not. Starlette's traversal guard compares the two, sees a mismatch,
    # and returns 404 for artifacts that exist -- every texture and preview in
    # the viewer fails silently. Comparing absolute paths still normalizes ".."
    # and still blocks traversal, which is what the guard is actually for.
    application.mount(
        "/artifacts",
        StaticFiles(directory=str(root), follow_symlink=True),
        name="artifacts",
    )

    # Mounted last so it only receives paths no API route or /artifacts claimed.
    # html=True serves index.html for unmatched paths, which keeps the single
    # origin working as one standalone application rather than two servers.
    if bundle is not None:
        application.mount(
            "/",
            StaticFiles(directory=str(bundle), html=True),
            name="ui",
        )
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
