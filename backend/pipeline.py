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
from .calibration import (
    calibrate_depth_from_gcps,
    calibrate_depth_from_reference_raster,
    parse_gcps_bytes,
)
from .evaluation import (
    BENCHMARK_CALIBRATION_LABEL,
    FIXED_CALIBRATION_EVALUATION_LABEL,
    GCP_CALIBRATION_LABEL,
    REFERENCE_DEM_CALIBRATION_LABEL,
    EvaluationResult,
    evaluate_calibrated_height,
    evaluate_relative_depth,
)
from .model import DepthEstimator, PredictionInfo, get_depth_estimator
from .raster_io import (
    ImageRaster,
    align_elevation_raster,
    read_elevation_bytes,
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
    reference_dem_bytes: bytes | None = None,
    reference_dem_filename: str | None = None,
    gcps_bytes: bytes | None = None,
    gcps_filename: str | None = None,
    gcp_sampling: str = "bilinear",
    estimator: DepthEstimator | Any | None = None,
    artifact_root: Path | None = None,
    public_artifact_base_url: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run inference, optional deployment calibration/evaluation, and export."""

    started = time.perf_counter()
    if reference_dem_bytes is not None and gcps_bytes is not None:
        raise ValueError("Supply only one deployment calibration reference: reference_dem or gcps")
    normalized_sampling = gcp_sampling.strip().lower()
    if normalized_sampling not in {"nearest", "bilinear"}:
        raise ValueError("GCP sampling must be 'nearest' or 'bilinear'")

    filename = Path(image_filename or "image").name
    source = read_image_bytes(image_bytes, filename)
    if source.width * source.height > _max_decoded_pixels():
        raise ValueError(
            f"Decoded image has {source.width * source.height:,} pixels; the configured "
            f"limit is {_max_decoded_pixels():,}"
        )

    preprocessing_notices: list[str] = []
    aligned_ground_truth: np.ndarray | None = None
    aligned_ground_truth_mask: np.ndarray | None = None
    if ground_truth_bytes is not None:
        gt_name = Path(ground_truth_filename or "ground_truth_dsm").name
        ground_truth = read_ground_truth_bytes(ground_truth_bytes, gt_name)
        (
            aligned_ground_truth,
            aligned_ground_truth_mask,
            alignment_notices,
        ) = align_elevation_raster(
            ground_truth,
            source,
            label="Evaluation ground-truth DSM",
        )
        preprocessing_notices.extend(alignment_notices)

    aligned_reference_dem: np.ndarray | None = None
    aligned_reference_dem_mask: np.ndarray | None = None
    reference_dem_original_valid_count: int | None = None
    reference_name: str | None = None
    if reference_dem_bytes is not None:
        reference_name = Path(reference_dem_filename or "reference_dem").name
        reference_dem = read_elevation_bytes(
            reference_dem_bytes,
            reference_name,
            label="Reference DEM",
        )
        reference_dem_original_valid_count = int(reference_dem.valid_mask.sum())
        (
            aligned_reference_dem,
            aligned_reference_dem_mask,
            alignment_notices,
        ) = align_elevation_raster(reference_dem, source, label="Reference DEM")
        preprocessing_notices.extend(alignment_notices)

    gcp_points = None
    gcp_name: str | None = None
    if gcps_bytes is not None:
        gcp_name = Path(gcps_filename or "gcps.json").name
        gcp_points = parse_gcps_bytes(gcps_bytes, gcp_name)

    depth_estimator = estimator or get_depth_estimator()
    prediction = depth_estimator.predict(Image.fromarray(source.rgb, mode="RGB"))
    depth, prediction_info = _prediction_parts(
        prediction, depth_estimator, source.width, source.height
    )

    notices = [
        f"Live output from {prediction_info.get('model_id', 'the configured model')} "
        "is relative monocular depth, not metric elevation."
    ]
    notices.extend(preprocessing_notices)
    inference_width = int(prediction_info.get("inference_width", source.width))
    inference_height = int(prediction_info.get("inference_height", source.height))
    if (inference_width, inference_height) != (source.width, source.height):
        notices.append(
            f"Model inference was bounded to {inference_width}x{inference_height} and "
            f"resampled to the source grid {source.width}x{source.height}."
        )

    evaluation: EvaluationResult | None = None
    calibrated_height: np.ndarray | None = None
    calibrated_valid_mask: np.ndarray | None = None
    calibration: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    reference_summary: dict[str, Any] | None = None
    mode = "relative_depth"

    if aligned_reference_dem is not None and aligned_reference_dem_mask is not None:
        fit = calibrate_depth_from_reference_raster(
            depth,
            aligned_reference_dem,
            aligned_reference_dem_mask,
            source_valid_mask=source.valid_mask,
            label=REFERENCE_DEM_CALIBRATION_LABEL,
        )
        calibration = fit.calibration
        calibrated_height = fit.calibrated_height
        calibrated_valid_mask = fit.valid_mask
        mode = "reference_dem_calibrated_dsm"
        coverage = float(calibration["reference_samples"] / depth.size)
        reference_summary = {
            "type": "reference_dem",
            "source": "reference_dem",
            "filename": reference_name,
            "point_count": reference_dem_original_valid_count,
            "calibration_points": calibration["reference_samples"],
            "holdout_points": 0,
            "coverage": coverage,
            "units": "not_inferred_from_raster",
            "aligned_width": source.width,
            "aligned_height": source.height,
        }
        notices.append(REFERENCE_DEM_CALIBRATION_LABEL)
        notices.append(
            "The reference DEM supplies only a global affine scale and offset. Its "
            "resolution, vertical datum, terrain/surface definition, and acquisition "
            "date may differ from the image and predicted surface."
        )
    elif gcp_points is not None:
        fit = calibrate_depth_from_gcps(
            depth,
            gcp_points,
            method=normalized_sampling,
            source_valid_mask=source.valid_mask,
            label=GCP_CALIBRATION_LABEL,
        )
        calibration = fit.calibration
        calibrated_height = fit.calibrated_height
        calibrated_valid_mask = fit.valid_mask
        mode = "gcp_calibrated_dsm"
        reference_summary = {
            "type": "gcps",
            "source": "gcps",
            "filename": gcp_name,
            "point_count": len(gcp_points),
            "calibration_points": calibration["reference_samples"],
            "holdout_points": 0,
            "coverage": min(1.0, float(len(gcp_points) / depth.size)),
            "units": "not_inferred_from_points",
            "coordinate_system": "zero_based_image_pixels",
            "sampling": normalized_sampling,
        }
        notices.append(GCP_CALIBRATION_LABEL)
        notices.append(
            "GCP x/col coordinates increase right and y/row coordinates increase "
            f"down from zero; model depth was sampled with {normalized_sampling} interpolation."
        )

    if calibration is not None and aligned_ground_truth is not None:
        assert calibrated_height is not None
        assert aligned_ground_truth_mask is not None
        evaluation = evaluate_calibrated_height(
            calibrated_height,
            aligned_ground_truth,
            aligned_ground_truth_mask,
            source.valid_mask,
            label=FIXED_CALIBRATION_EVALUATION_LABEL,
        )
        metrics = evaluation.metrics
        metrics["calibration_source"] = calibration["source"]
        if reference_summary is not None:
            reference_summary["holdout_points"] = metrics["valid_pixels"]
        notices.append(FIXED_CALIBRATION_EVALUATION_LABEL)
    elif calibration is not None:
        notices.append(
            "No ground_truth_dsm was supplied, so independent Pearson, MAE, and RMSE "
            "evaluation metrics are null. Calibration-reference fit is not reported as accuracy."
        )
    elif aligned_ground_truth is not None:
        assert aligned_ground_truth_mask is not None
        evaluation = evaluate_relative_depth(
            depth,
            aligned_ground_truth,
            aligned_ground_truth_mask,
            source.valid_mask,
        )
        calibration = evaluation.calibration
        calibrated_height = evaluation.calibrated_height
        calibrated_valid_mask = np.isfinite(depth)
        if source.valid_mask is not None:
            calibrated_valid_mask &= source.valid_mask
        metrics = evaluation.metrics
        mode = "benchmark_calibrated_dsm"
        notices.append(BENCHMARK_CALIBRATION_LABEL)
        notices.append(
            "The full evaluation DSM was used to choose orientation and fit scale/offset; "
            "these same-scene metrics contain calibration leakage and are benchmark-only."
        )

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
    depth_for_export = depth.copy()
    if source.valid_mask is not None:
        depth_for_export[~source.valid_mask] = np.nan
    artifacts["depth_npy"] = writer.write_npy("depth.npy", depth_for_export)

    urls: dict[str, str | None] = {
        "original": artifacts["original_png"],
        "depth": artifacts["depth_png"],
        "ground_truth": None,
        "error": None,
        "reference_dem": None,
        "calibrated_dsm": None,
        "calibrated_dsm_npy": None,
    }

    if aligned_reference_dem is not None and aligned_reference_dem_mask is not None:
        artifacts["reference_dem_png"] = writer.write_preview(
            "reference_dem.png", aligned_reference_dem, aligned_reference_dem_mask
        )
        urls["reference_dem"] = artifacts["reference_dem_png"]

    if aligned_ground_truth is not None and aligned_ground_truth_mask is not None:
        ground_truth_preview_mask = (
            evaluation.valid_mask if evaluation is not None else aligned_ground_truth_mask
        )
        artifacts["ground_truth_png"] = writer.write_preview(
            "ground_truth.png", aligned_ground_truth, ground_truth_preview_mask
        )
        urls["ground_truth"] = artifacts["ground_truth_png"]

    if evaluation is not None and aligned_ground_truth is not None:
        artifacts["error_png"] = writer.write_preview(
            "error.png", evaluation.absolute_error, evaluation.valid_mask
        )
        urls["error"] = artifacts["error_png"]

    if calibrated_height is not None and calibrated_valid_mask is not None:
        calibrated_for_export = calibrated_height.copy()
        calibrated_for_export[~calibrated_valid_mask] = np.nan
        artifacts["calibrated_dsm_png"] = writer.write_preview(
            "calibrated_dsm.png", calibrated_height, calibrated_valid_mask
        )
        artifacts["calibrated_dsm_npy"] = writer.write_npy(
            "calibrated_dsm.npy", calibrated_for_export
        )
        urls["calibrated_dsm"] = artifacts["calibrated_dsm_png"]
        urls["calibrated_dsm_npy"] = artifacts["calibrated_dsm_npy"]

        assert calibration is not None
        geotiff = writer.write_calibrated_geotiff(
            "calibrated_dsm.tif",
            calibrated_height,
            source,
            calibrated_valid_mask,
            calibration_source=str(calibration["source"]),
        )
        if geotiff is not None:
            artifacts["calibrated_dsm_geotiff"] = geotiff
            notices.append(
                "Calibrated predicted DSM GeoTIFF preserves the source CRS, transform, "
                "dimensions, valid-data mask, and source tags. Horizontal CRS metadata "
                "does not establish a vertical datum or elevation unit."
            )
        else:
            notices.append(
                "A calibrated DSM GeoTIFF was not exported because the source image "
                "did not provide a valid GeoTIFF CRS and transform."
            )

    if calibrated_height is not None:
        mesh_heights = calibrated_height
        mesh_mask = calibrated_valid_mask
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
        "reference": reference_summary,
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
        "reference": reference_summary,
        "depth_grid": make_mesh_grid(mesh_heights, valid_mask=mesh_mask),
        "urls": urls,
        "artifacts": artifacts,
        "notices": notices,
    }
    return payload
