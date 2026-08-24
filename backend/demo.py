"""Deterministic, explicitly labeled synthetic demonstration fixture."""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactWriter, default_artifact_root, make_mesh_grid
from .evaluation import EvaluationResult, evaluate_relative_depth


DEMO_JOB_ID = "demo-synthetic-v1"
DEMO_NOTICE = "Precomputed synthetic demonstration fixture — not live model output"
_demo_write_lock = threading.Lock()


@lru_cache(maxsize=1)
def _fixture_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, EvaluationResult]:
    """Return the fixed synthetic fixture; no ML model is consulted."""

    width, height = 256, 176
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    hill = 0.72 * np.exp(-4.5 * ((xx + 0.28) ** 2 + (yy + 0.08) ** 2))
    ridge = 0.36 * np.exp(-18.0 * (yy - 0.32 * np.sin(2.4 * xx)) ** 2)
    basin = -0.18 * np.exp(-10.0 * ((xx - 0.48) ** 2 + (yy + 0.38) ** 2))
    terrain = (0.18 * (1.0 - yy) + hill + ridge + basin).astype(np.float32)
    terrain -= terrain.min()
    terrain /= max(float(terrain.max()), 1e-6)

    ground_truth = (132.0 + 86.0 * terrain).astype(np.float32)
    structured_model_error = 0.028 * np.sin(8.0 * xx) * np.cos(5.0 * yy)
    # Negative orientation intentionally exercises the same orientation logic
    # used for real relative-depth evaluation.
    relative_depth = (-(terrain + structured_model_error)).astype(np.float32)
    valid = np.ones(terrain.shape, dtype=bool)
    evaluation = evaluate_relative_depth(relative_depth, ground_truth, valid)

    # Terrain-tinted synthetic RGB with deterministic analytical shading.
    dy, dx = np.gradient(terrain)
    shade = np.clip(0.76 - 1.25 * dx - 0.9 * dy, 0.42, 1.12)
    red = np.clip((48 + terrain * 133) * shade, 0, 255)
    green = np.clip((82 + terrain * 118) * shade, 0, 255)
    blue = np.clip((58 + (1.0 - terrain) * 85) * shade, 0, 255)
    rgb = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    return rgb, relative_depth, ground_truth, evaluation


def demo_response(
    public_artifact_base_url: str,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize demo assets once and return the standard API response shape."""

    rgb, relative_depth, ground_truth, evaluation = _fixture_arrays()
    writer = ArtifactWriter(
        artifact_root or default_artifact_root(),
        DEMO_JOB_ID,
        public_base_url=public_artifact_base_url,
    )
    notices = [
        DEMO_NOTICE,
        "This fixture is deterministic and available without downloading the live model.",
        evaluation.metrics["label"],
    ]

    filenames = (
        "original.png",
        "depth.png",
        "depth.npy",
        "ground_truth.png",
        "error.png",
        "metrics.json",
    )
    with _demo_write_lock:
        if not all(writer.path(name).is_file() for name in filenames):
            writer.write_rgb("original.png", rgb)
            writer.write_preview("depth.png", relative_depth)
            writer.write_npy("depth.npy", relative_depth)
            writer.write_preview("ground_truth.png", ground_truth)
            writer.write_preview("error.png", evaluation.absolute_error, evaluation.valid_mask)
            writer.write_json(
                "metrics.json",
                {
                    "fixture": DEMO_NOTICE,
                    "metrics": evaluation.metrics,
                    "calibration": evaluation.calibration,
                    "notices": notices,
                },
            )

    artifacts = {
        "original_png": writer.reference("original.png"),
        "depth_png": writer.reference("depth.png"),
        "depth_npy": writer.reference("depth.npy"),
        "ground_truth_png": writer.reference("ground_truth.png"),
        "error_png": writer.reference("error.png"),
        "metrics_json": writer.reference("metrics.json"),
    }
    return {
        "job_id": DEMO_JOB_ID,
        "demo": True,
        "precomputed": True,
        "model": "precomputed-synthetic-fixture-v1",
        "device": "none",
        "mode": "precomputed_synthetic_demo",
        "input": {
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "filename": "synthetic_demo_rgb.png",
        },
        "processing_time_seconds": 0.0,
        "geospatial": None,
        "metrics": evaluation.metrics,
        "calibration": evaluation.calibration,
        "depth_grid": make_mesh_grid(evaluation.calibrated_height),
        "urls": {
            "original": artifacts["original_png"],
            "depth": artifacts["depth_png"],
            "ground_truth": artifacts["ground_truth_png"],
            "error": artifacts["error_png"],
        },
        "artifacts": artifacts,
        "notices": notices,
    }
