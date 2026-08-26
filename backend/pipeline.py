"""End-to-end DepthWizard analysis pipeline shared by the API and CLIs."""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifacts import (
    ArtifactWriter,
    default_artifact_root,
    make_mesh_grid,
    normalize_for_preview,
    quantize_height_grid,
)
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
from .model import QUALITY_MODES, DepthEstimator, PredictionInfo, get_depth_estimator
from ml.buildings import flatten_building_roofs
from ml.refine import flatten_surfaces, refine_depth_with_image

from .scene_calibration import calibrate_from_scene_shadows
from .raster_io import (
    ImageRaster,
    align_elevation_raster,
    read_elevation_bytes,
    read_ground_truth_bytes,
    read_image_bytes,
)


def _refinement_settings() -> tuple[bool, int, float]:
    """Read guided-refinement configuration from the environment."""

    enabled = os.getenv("DEPTHWIZARD_REFINE", "1").strip().lower() not in {"0", "false", "off"}
    try:
        radius = int(os.getenv("DEPTHWIZARD_REFINE_RADIUS", "32"))
    except ValueError as exc:
        raise ValueError("DEPTHWIZARD_REFINE_RADIUS must be an integer") from exc
    try:
        epsilon = float(os.getenv("DEPTHWIZARD_REFINE_EPSILON", "0.001"))
    except ValueError as exc:
        raise ValueError("DEPTHWIZARD_REFINE_EPSILON must be a number") from exc
    return enabled, max(1, radius), max(1e-8, epsilon)


def _flatten_settings() -> tuple[int, float]:
    """Read roof-flattening configuration from the environment."""

    try:
        iterations = int(os.getenv("DEPTHWIZARD_FLATTEN_ITERATIONS", "80"))
    except ValueError as exc:
        raise ValueError("DEPTHWIZARD_FLATTEN_ITERATIONS must be an integer") from exc
    try:
        kappa = float(os.getenv("DEPTHWIZARD_FLATTEN_KAPPA", "0.12"))
    except ValueError as exc:
        raise ValueError("DEPTHWIZARD_FLATTEN_KAPPA must be a number") from exc
    return max(0, iterations), max(1e-6, kappa)


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
            "quality_mode": info.quality_mode,
            "inference_passes": info.inference_passes,
            "tiled": info.tiled,
            "tile_count": info.tile_count,
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
    quality_mode: str = "fast",
    acquisition_time: str | None = None,
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
    normalized_quality = quality_mode.strip().lower()
    if normalized_quality not in QUALITY_MODES:
        raise ValueError(
            f"quality_mode must be one of {', '.join(QUALITY_MODES)}, "
            f"got {quality_mode!r}"
        )

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
    source_image = Image.fromarray(source.rgb, mode="RGB")
    if normalized_quality == "fast":
        # Preserve compatibility with custom/test estimators that implement the
        # original single-argument protocol, and keep the default path unchanged.
        prediction = depth_estimator.predict(source_image)
    else:
        prediction = depth_estimator.predict(
            source_image, quality_mode=normalized_quality
        )
    depth, prediction_info = _prediction_parts(
        prediction, depth_estimator, source.width, source.height
    )

    notices = [
        f"Live output from {prediction_info.get('model_id', 'the configured model')} "
        "is relative monocular depth, not metric elevation."
    ]

    notices.extend(preprocessing_notices)
    inference_passes = int(prediction_info.get("inference_passes", 1))
    tiled = bool(prediction_info.get("tiled", False))
    tile_count = int(prediction_info.get("tile_count", 0))
    if normalized_quality == "quality":
        if tiled:
            notices.append(
                f"Quality inference used one global pass plus {tile_count} overlapping "
                "local tiles with relative-scale alignment and feathered blending."
            )
        else:
            notices.append(
                "Quality inference averaged the global prediction with a horizontally "
                "flipped consistency pass after relative-scale alignment."
            )
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

    elif source.crs is not None and source.transform is not None:
        # No reference DEM and no GCPs, but the image is georeferenced. Solar
        # geometry plus shadow length yields metric heights from the scene
        # itself, which is the only route to an absolute DSM without an extra
        # file. It reports a reason instead of a number whenever it cannot be
        # trusted, so an unusable fit leaves the output relative.
        parsed_time: datetime | None = None
        if acquisition_time:
            try:
                parsed_time = datetime.fromisoformat(acquisition_time.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    "acquisition_time must be ISO 8601, for example 2021-04-15T15:30:00Z"
                ) from exc
            if parsed_time.tzinfo is None:
                parsed_time = parsed_time.replace(tzinfo=timezone.utc)
        shadow_fit = calibrate_from_scene_shadows(depth, source, acquisition_time=parsed_time)
        if shadow_fit.get("usable"):
            scale = float(shadow_fit["scale"])
            offset = float(shadow_fit["offset"])
            calibrated_height = (depth * scale + offset).astype(np.float32)
            calibrated_valid_mask = np.isfinite(depth)
            if source.valid_mask is not None:
                calibrated_valid_mask &= source.valid_mask
            calibration = {
                "source": "scene_shadows",
                "method": shadow_fit["method"],
                "scale": scale,
                "shift": offset,
                "units": shadow_fit["units"],
                "reference_samples": int(shadow_fit["shadow_samples"]),
                "solar_elevation_degrees": shadow_fit["solar"]["elevation_degrees"],
                "solar_azimuth_degrees": shadow_fit["solar"]["azimuth_degrees"],
                "acquisition_time": shadow_fit.get("acquisition_time"),
            }
            mode = "shadow_calibrated_dsm"
            reference_summary = {
                "type": "scene_shadows",
                "source": "scene_shadows",
                "filename": None,
                "point_count": int(shadow_fit["shadow_samples"]),
                "calibration_points": int(shadow_fit["shadow_samples"]),
                "holdout_points": 0,
                "coverage": shadow_fit.get("shadow_fraction"),
                "units": shadow_fit["units"],
            }
            notices.append(
                "Metric heights were derived from shadow length and solar "
                f"geometry at {shadow_fit['solar']['elevation_degrees']:.1f} degrees "
                f"elevation, using {shadow_fit['shadow_samples']} shadow runs. No "
                "external elevation reference was used."
            )
            notices.append(str(shadow_fit["note"]))
        else:
            notices.append(
                "Shadow-based metric calibration was not applied: "
                f"{shadow_fit.get('reason')}"
            )

    if calibration is not None and aligned_ground_truth is not None:
        assert calibrated_height is not None
        assert aligned_ground_truth_mask is not None

        # Shadow calibration yields height above local ground, with no vertical
        # datum: nothing in a single image fixes where mean sea level is. A
        # ground-truth DSM is absolute elevation, so comparing the two directly
        # measures the datum offset and swamps the real error. Removing a single
        # constant restores a like-for-like comparison of relief; this is one
        # degree of freedom, not a per-pixel fit, and it is recorded in the
        # metrics so the number is never mistaken for absolute agreement.
        if calibration.get("source") == "scene_shadows":
            comparable = np.isfinite(calibrated_height) & aligned_ground_truth_mask
            if source.valid_mask is not None:
                comparable &= source.valid_mask
            if int(comparable.sum()) >= 64:
                datum_offset = float(
                    np.median(aligned_ground_truth[comparable] - calibrated_height[comparable])
                )
                calibrated_height = (calibrated_height + datum_offset).astype(np.float32)
                calibration["vertical_datum_offset"] = round(datum_offset, 3)
                calibration["datum_note"] = (
                    "A single constant was added to align height-above-ground with "
                    "the evaluation DSM's vertical datum. Metrics therefore measure "
                    "relief agreement, not absolute elevation."
                )
                notices.append(
                    f"A constant vertical offset of {datum_offset:.2f} m aligned the "
                    "shadow-calibrated surface to the evaluation DSM datum; reported "
                    "metrics measure relief, not absolute elevation."
                )
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
        "height16": None,
        "normal": None,
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

    # Guided refinement is applied to the render surface only, never to the
    # exported depth, the calibrated DSM, or the metrics.
    #
    # It pulls depth edges onto the image edges, which sharpens building
    # outlines considerably. It also transfers some image albedo into the
    # height field, and measured against the DC LiDAR DSM that costs 2% RMSE at
    # radius 8 and 15% at radius 32. Accuracy is scored on the exported
    # products, so the raw prediction must stay untouched.
    refine_enabled, refine_radius, refine_epsilon = _refinement_settings()
    if refine_enabled and mesh_heights is not None:
        mesh_heights = refine_depth_with_image(
            mesh_heights,
            source.rgb,
            radius=refine_radius,
            epsilon=refine_epsilon,
            valid_mask=mesh_mask,
        )
        notices.append(
            "The 3-D surface was edge-aligned to the source image with a guided "
            f"filter (radius {refine_radius}). This is a display-only refinement; "
            "exported depth, the calibrated DSM, and all metrics use the "
            "unrefined prediction."
        )

    # Segment buildings and fit each roof its own plane. Diffusion alone levels
    # whatever is locally smooth; this identifies the regions that are actually
    # structures and makes their roofs planar, which is what separates a
    # building from a mound. Display-only, like the filters around it.
    if os.getenv("DEPTHWIZARD_FLATTEN_ROOFS", "1").strip().lower() not in {"0", "false", "off"}:
        mesh_heights, roof_report = flatten_building_roofs(
            mesh_heights, valid_mask=mesh_mask
        )
        if roof_report["buildings"]:
            notices.append(
                f"{roof_report['buildings']} building regions were segmented and "
                "their roofs fitted to planes for the 3-D surface. Display-only; "
                "exported products are unaffected."
            )

    # Anisotropic diffusion, independent of edge alignment: levels the interior
    # of each rooftop while refusing to cross its boundary, so buildings read as
    # flat-topped blocks rather than mounds. Also display-only.
    flatten_iterations, flatten_kappa = _flatten_settings()
    if flatten_iterations > 0 and mesh_heights is not None:
        mesh_heights = flatten_surfaces(
            mesh_heights,
            iterations=flatten_iterations,
            kappa=flatten_kappa,
            valid_mask=mesh_mask,
        )
        notices.append(
            f"The 3-D surface was levelled with {flatten_iterations} "
            "anisotropic-diffusion steps to flatten roof planes. Display-only; "
            "exported products are unaffected."
        )

    # The mesh grid drives how sharp the reconstruction can look, so it is
    # carried at 512 cells as a 16-bit PNG instead of a JSON float array. At the
    # old 192-cell cap a wall had a single cell to fall through and every roof
    # edge became a slope.
    depth_grid = make_mesh_grid(mesh_heights, valid_mask=mesh_mask)
    quantised, mesh_valid, mesh_low, mesh_high = quantize_height_grid(
        mesh_heights, valid_mask=mesh_mask
    )
    artifacts["height_png16"] = writer.write_height_png16(
        "height16.png", quantised, mesh_valid
    )
    depth_grid["encoded"] = {
        "url": artifacts["height_png16"],
        "format": "png16",
        "minimum": mesh_low,
        "maximum": mesh_high,
        "width": int(quantised.shape[1]),
        "height": int(quantised.shape[0]),
        "mask_url": (
            artifacts["height_png16"].replace("height16.png", "height16_mask.png")
            if not mesh_valid.all()
            else None
        ),
    }
    urls["height16"] = artifacts["height_png16"]

    # Derived from the full-resolution prediction rather than the 512-cell mesh,
    # so it carries relief the geometry cannot represent.
    artifacts["normal_png"] = writer.write_normal_map(
        "normal.png", mesh_heights, strength=24.0, valid_mask=mesh_mask
    )
    urls["normal"] = artifacts["normal_png"]

    inference_summary = {
        "quality_mode": normalized_quality,
        "passes": inference_passes,
        "tiled": tiled,
        "tile_count": tile_count,
        "bounded_width": inference_width,
        "bounded_height": inference_height,
    }

    # Keep a machine-readable run record even when no benchmark DSM was
    # supplied; in that case its metric/calibration values are explicitly null.
    metrics_payload = {
        "mode": mode,
        "model": str(prediction_info.get("model_id", "unknown")),
        "device": str(prediction_info.get("device", "unknown")),
        "inference": inference_summary,
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
        "inference": inference_summary,
        "geospatial": source.geospatial,
        "metrics": metrics,
        "calibration": calibration,
        "reference": reference_summary,
        "depth_grid": depth_grid,
        "urls": urls,
        "artifacts": artifacts,
        "notices": notices,
    }
    return payload
