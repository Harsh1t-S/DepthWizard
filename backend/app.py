"""FastAPI entry point for the DepthWizard prototype."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
    return f"{str(request.base_url).rstrip('/')}/artifacts"


def create_app(artifact_root: Path | None = None) -> FastAPI:
    """Create an app; injectable artifact storage keeps API tests isolated."""

    root = Path(artifact_root or default_artifact_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    application = FastAPI(
        title="DepthWizard API",
        version=API_VERSION,
        description=(
            "Real Depth Anything V2 Small inference with optional benchmark-only "
            "DSM calibration and geospatial exports."
        ),
    )
    application.state.artifact_root = root
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://localhost:5173",
            "https://127.0.0.1:5173",
        ],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
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
    ) -> dict[str, Any]:
        if ground_truth_dsm is not None and dsm is not None:
            raise HTTPException(
                status_code=422,
                detail="Supply only ground_truth_dsm (preferred) or dsm, not both",
            )
        selected_dsm = ground_truth_dsm or dsm
        image_bytes = await _read_upload(image, "image")
        ground_truth_bytes: bytes | None = None
        if selected_dsm is not None:
            ground_truth_bytes = await _read_upload(selected_dsm, "ground_truth_dsm")

        work = partial(
            analyze_bytes,
            image_bytes=image_bytes,
            image_filename=image.filename or "image",
            ground_truth_bytes=ground_truth_bytes,
            ground_truth_filename=(selected_dsm.filename if selected_dsm else None),
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

    application.mount(
        "/artifacts",
        StaticFiles(directory=str(root)),
        name="artifacts",
    )
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
