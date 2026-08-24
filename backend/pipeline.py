"""End-to-end DepthWizard analysis pipeline shared by the API and CLIs."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifacts import ArtifactWriter, default_artifact_root, make_mesh_grid, normalize_for_preview
from .evaluation import BENCHMARK_CALIBRATION_LABEL, EvaluationResult, evaluate_relative_depth
from .model import DepthEstimator, PredictionInfo, get_depth_estimator
from .raster_io import (
    ImageRaster,
    align_ground_truth,
    read_ground_truth_bytes,
    read_image_bytes,
)


def _max_decoded_pixels() -> int:
    raw = os.getenv("DEPTHWIZARD_MAX_DECODED_PIXELS", "50000000")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("DEPTHWIZARD_MAX_DECODED_PIXELS must be an integer") from exc
    return max(value, 1)


def _prediction_parts(
    result: Any,
    estimator: Any,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Accept the production tuple while keeping estimator test doubles tiny."""

    if isinstance(result, tuple) and len(result) == 2:
        depth, info = result
    else:
        depth, info = result, None
    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.shape != (height, width):
        raise ValueError(
            f"Depth estimator returned {depth_array.shape}, expected {(height, width)}"
        )
    if not np.isfinite(depth_array).any():
        raise ValueError("Depth estimator returned no finite pixels")

    if isinstance(info, PredictionInfo):
        metadata = {
            "model_id": info.model_id,
            "device": info.device,
            "inference_width": info.inference_width,
            "inference_height": info.inference_height,
        }
    elif isinstance(info, dict):
        metadata = dict(info)
    elif info is not None:
        metadata = {
            "model_id": getattr(info, "model_id", getattr(estimator, "model_id", "unknown")),
            "device": getattr(info, "device", getattr(estimator, "device", "unknown")),
            "inference_width": getattr(info, "inference_width", width),
            "inference_height": getattr(info, "inference_height", height),
        }
    else:
        metadata = {
            "model_id": getattr(estimator, "model_id", "mock-or-custom-estimator"),
            "device": getattr(estimator, "device", "unknown"),
            "inference_width": width,
            "inference_height": height,
        }
    return depth_array, metadata


def analyze_bytes(
    image_bytes: bytes,
    image_filename: str,
    ground_truth_bytes: bytes | None = None,
    ground_truth_filename: str | None = None,
    *,
    estimator: DepthEstimator | Any | None = None,
    artifact_root: Path | None = None,
    public_artifact_base_url: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run real model inference, optional evaluation, and artifact export."""

    started = time.perf_counter()
    filename = Path(image_filename or "image").name
    source = read_image_bytes(image_bytes, filename)
    if source.width * source.height > _max_decoded_pixels():
        raise ValueError(
            f"Decoded image has {source.width * source.height:,} pixels; the configured "
            f"limit is {_max_decoded_pixels():,}"
        )

    depth_estimator = estimator or get_depth_estimator()
    prediction = depth_estimator.predict(Image.fromarray(source.rgb, mode="RGB"))
    depth, prediction_info = _prediction_parts(
        prediction, depth_estimator, source.width, source.height
    )

    notices = [
        "Live output from Depth Anything V2 is relative monocular depth, not metric elevation."
    ]
    inference_width = int(prediction_info.get("inference_width", source.width))
    inference_height = int(prediction_info.get("inference_height", source.height))
    if (inference_width, inference_height) != (source.width, source.height):
        notices.append(
            f"Model inference was bounded to {inference_width}x{inference_height} and "
            f"resampled to the source grid {source.width}x{source.height}."
        )

    aligned_ground_truth: np.ndarray | None = None
    aligned_ground_truth_mask: np.ndarray | None = None
    evaluation: EvaluationResult | None = None
    if ground_truth_bytes is not None:
        gt_name = Path(ground_truth_filename or "ground_truth_dsm").name
        ground_truth = read_ground_truth_bytes(ground_truth_bytes, gt_name)
        aligned_ground_truth, aligned_ground_truth_mask, alignment_notices = align_ground_truth(
            ground_truth, source
        )
        notices.extend(alignment_notices)
        evaluation = evaluate_relative_depth(
            depth,
            aligned_ground_truth,
            aligned_ground_truth_mask,
            source.valid_mask,
        )
        notices.append(BENCHMARK_CALIBRATION_LABEL)

    artifact_job_id = job_id or uuid.uuid4().hex
    writer = ArtifactWriter(
        artifact_root or default_artifact_root(),
        artifact_job_id,
        public_base_url=public_artifact_base_url,
    )
    artifacts: dict[str, str] = {}
    artifacts["original_png"] = writer.write_rgb("original.png", source.rgb)
    artifacts["depth_png"] = writer.write_preview(
        "depth.png", depth, source.valid_mask
    )
    artifacts["depth_npy"] = writer.write_npy("depth.npy", depth)

    urls: dict[str, str | None] = {
        "original": artifacts["original_png"],
        "depth": artifacts["depth_png"],
        "ground_truth": None,
        "error": None,
    }

    metrics: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    if evaluation is not None and aligned_ground_truth is not None:
        metrics = evaluation.metrics
        calibration = evaluation.calibration
        artifacts["ground_truth_png"] = writer.write_preview(
            "ground_truth.png", aligned_ground_truth, evaluation.valid_mask
        )
        artifacts["error_png"] = writer.write_preview(
            "error.png", evaluation.absolute_error, evaluation.valid_mask
        )
        urls["ground_truth"] = artifacts["ground_truth_png"]
        urls["error"] = artifacts["error_png"]

        geotiff = writer.write_calibrated_geotiff(
            "calibrated_dsm.tif",
            evaluation.calibrated_height,
            source,
            source.valid_mask,
        )
        if geotiff is not None:
            artifacts["calibrated_dsm_geotiff"] = geotiff
            notices.append(
                "Calibrated predicted DSM GeoTIFF preserves the source CRS, transform, "
                "dimensions, valid-data mask, and source tags."
            )
        else:
            notices.append(
                "A calibrated DSM GeoTIFF was not exported because the source image "
                "did not provide a valid GeoTIFF CRS and transform."
            )

    if evaluation is not None:
        mesh_heights = evaluation.calibrated_height
        mesh_mask = source.valid_mask
        mode = "benchmark_calibrated_dsm"
    else:
        # A normalized grid is more stable for an uncalibrated 3-D preview than
        # arbitrary model logits; depth.npy retains the unmodified float output.
        mesh_heights = normalize_for_preview(depth, source.valid_mask)
        mesh_mask = source.valid_mask
        mode = "relative_depth"

    # Keep a machine-readable run record even when no benchmark DSM was
    # supplied; in that case its metric/calibration values are explicitly null.
    metrics_payload = {
        "mode": mode,
        "model": str(prediction_info.get("model_id", "unknown")),
        "device": str(prediction_info.get("device", "unknown")),
        "metrics": metrics,
        "calibration": calibration,
        "notices": notices,
    }
    artifacts["metrics_json"] = writer.write_json("metrics.json", metrics_payload)

    payload = {
        "job_id": artifact_job_id,
        "demo": False,
        "precomputed": False,
        "model": str(prediction_info.get("model_id", "unknown")),
        "device": str(prediction_info.get("device", "unknown")),
        "mode": mode,
        "input": {
            "width": source.width,
            "height": source.height,
            "filename": filename,
        },
        "processing_time_seconds": round(time.perf_counter() - started, 4),
        "geospatial": source.geospatial,
        "metrics": metrics,
        "calibration": calibration,
        "depth_grid": make_mesh_grid(mesh_heights, valid_mask=mesh_mask),
        "urls": urls,
        "artifacts": artifacts,
        "notices": notices,
    }
    return payload
