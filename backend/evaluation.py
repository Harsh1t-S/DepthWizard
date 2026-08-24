"""Affine calibration and evaluation helpers for relative monocular depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BENCHMARK_CALIBRATION_LABEL = (
    "Benchmark-only affine calibration against supplied ground truth; "
    "not available for unseen terrain."
)

REFERENCE_DEM_CALIBRATION_LABEL = (
    "Deployment-style affine calibration against the supplied reference DEM. "
    "This establishes a scene-level scale and offset; it is not an accuracy benchmark."
)

GCP_CALIBRATION_LABEL = (
    "Deployment-style affine calibration against supplied pixel-coordinate ground "
    "control points. This establishes a scene-level scale and offset; it is not an "
    "accuracy benchmark."
)

FIXED_CALIBRATION_EVALUATION_LABEL = (
    "Evaluation against supplied ground truth using calibration fixed from a separate "
    "deployment reference; ground truth was not used to refit scale or offset."
)


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    calibration: dict[str, Any]
    calibrated_height: np.ndarray
    absolute_error: np.ndarray
    valid_mask: np.ndarray


@dataclass
class CalibrationResult:
    """A fitted affine transform applied to the complete model-depth grid."""

    calibration: dict[str, Any]
    calibrated_height: np.ndarray
    valid_mask: np.ndarray
    raw_correlation: float | None
    inverted_correlation: float | None
    relative_correlation: float | None


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Compute Pearson r without allocating a two-column correlation matrix."""

    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None
    x = x[valid]
    y = y[valid]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        return None
    value = float(np.dot(x_centered, y_centered) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _finite_number(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def fit_affine_calibration(
    depth: np.ndarray,
    sampled_depth: np.ndarray,
    sampled_elevation: np.ndarray,
    *,
    source: str,
    label: str,
    benchmark_only: bool,
    output_valid_mask: np.ndarray | None = None,
) -> CalibrationResult:
    """Fit an oriented affine transform from paired depth/elevation samples.

    The fit samples may come from a dense aligned elevation raster or sparse
    ground-control points.  The resulting transform is always applied to the
    complete full-resolution depth grid, independently of any later evaluation.
    """

    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim != 2:
        raise ValueError("Relative depth must be a two-dimensional array")
    x_raw = np.asarray(sampled_depth, dtype=np.float64).reshape(-1)
    y = np.asarray(sampled_elevation, dtype=np.float64).reshape(-1)
    if x_raw.shape != y.shape:
        raise ValueError("Calibration depth and elevation samples must have equal length")
    finite = np.isfinite(x_raw) & np.isfinite(y)
    x_raw = x_raw[finite]
    y = y[finite]
    sample_count = int(x_raw.size)
    if sample_count < 3:
        raise ValueError("At least three finite reference samples are required for calibration")

    raw_correlation = pearson_correlation(x_raw, y)
    inverted_correlation = pearson_correlation(-x_raw, y)
    use_negative = (
        inverted_correlation is not None
        and (raw_correlation is None or inverted_correlation > raw_correlation)
    )
    orientation_sign = -1.0 if use_negative else 1.0
    orientation = "negative_depth" if use_negative else "depth"
    x = x_raw * orientation_sign
    relative_correlation = pearson_correlation(x, y)

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    denominator = float(np.dot(x_centered, x_centered))
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        raise ValueError(
            "Relative depth is constant over the reference samples; calibration failed"
        )
    scale = float(np.dot(x_centered, y - y_mean) / denominator)
    shift = float(y_mean - scale * x_mean)
    if not np.isfinite(scale) or not np.isfinite(shift):
        raise ValueError("Affine calibration produced non-finite coefficients")

    calibrated = (scale * (depth_array.astype(np.float64) * orientation_sign) + shift).astype(
        np.float32
    )
    valid_output = np.isfinite(depth_array) & np.isfinite(calibrated)
    if output_valid_mask is not None:
        candidate_mask = np.asarray(output_valid_mask, dtype=bool)
        if candidate_mask.shape != depth_array.shape:
            raise ValueError("Output validity mask must share the depth pixel grid")
        valid_output &= candidate_mask

    calibration: dict[str, Any] = {
        "method": "affine_least_squares",
        "source": source,
        "scale": scale,
        "shift": shift,
        "orientation": orientation,
        "reference_samples": sample_count,
        "benchmark_only": bool(benchmark_only),
        "label": label,
        # Compatibility aliases retained for existing clients.
        "orientation_sign": orientation_sign,
        "equation": "height = a * oriented_depth + b",
        "a": scale,
        "b": shift,
        "coefficients": {"a": scale, "b": shift},
    }
    return CalibrationResult(
        calibration=calibration,
        calibrated_height=calibrated,
        valid_mask=valid_output,
        raw_correlation=_finite_number(raw_correlation),
        inverted_correlation=_finite_number(inverted_correlation),
        relative_correlation=_finite_number(relative_correlation),
    )


def evaluate_calibrated_height(
    calibrated_height: np.ndarray,
    ground_truth: np.ndarray,
    ground_truth_valid_mask: np.ndarray,
    source_valid_mask: np.ndarray | None = None,
    *,
    label: str = FIXED_CALIBRATION_EVALUATION_LABEL,
) -> EvaluationResult:
    """Score already-calibrated heights against ground truth without refitting."""

    calibrated = np.asarray(calibrated_height, dtype=np.float32)
    truth = np.asarray(ground_truth, dtype=np.float32)
    valid = np.asarray(ground_truth_valid_mask, dtype=bool)
    if calibrated.shape != truth.shape or calibrated.shape != valid.shape:
        raise ValueError(
            "Calibrated height, ground truth, and validity mask must share a pixel grid"
        )
    valid = valid & np.isfinite(calibrated) & np.isfinite(truth)
    if source_valid_mask is not None:
        source_mask = np.asarray(source_valid_mask, dtype=bool)
        if source_mask.shape != calibrated.shape:
            raise ValueError("Source validity mask must share the evaluation pixel grid")
        valid &= source_mask
    valid_count = int(valid.sum())
    if valid_count < 1:
        raise ValueError(
            "At least one finite, non-NoData aligned pixel is required for evaluation"
        )

    residual = calibrated[valid].astype(np.float64) - truth[valid].astype(np.float64)
    absolute_error = np.full(calibrated.shape, np.nan, dtype=np.float32)
    absolute_error[valid] = np.abs(residual).astype(np.float32)
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    correlation = pearson_correlation(calibrated[valid], truth[valid])
    total_pixels = int(calibrated.size)
    metrics: dict[str, Any] = {
        "label": label,
        "evaluation_type": "fixed_calibration_holdout",
        "evaluation_scope": "holdout_fixed_calibration",
        "calibration_refit": False,
        "valid_pixels": valid_count,
        "total_pixels": total_pixels,
        "valid_fraction": float(valid_count / total_pixels),
        "pearson_correlation": _finite_number(correlation),
        "mae": _finite_number(mae),
        "rmse": _finite_number(rmse),
        # Existing response names remain available to current clients.
        "calibrated_pearson_correlation": _finite_number(correlation),
        "calibrated_mae": _finite_number(mae),
        "calibrated_rmse": _finite_number(rmse),
    }
    return EvaluationResult(
        metrics=metrics,
        calibration={},
        calibrated_height=calibrated,
        absolute_error=absolute_error,
        valid_mask=valid,
    )


def evaluate_relative_depth(
    depth: np.ndarray,
    ground_truth: np.ndarray,
    ground_truth_valid_mask: np.ndarray,
    source_valid_mask: np.ndarray | None = None,
) -> EvaluationResult:
    """Orient relative depth and fit ``height = a * depth + b``.

    Both the original depth and its negation are tested.  The orientation with
    the higher raw Pearson correlation is chosen before an unconstrained affine
    least-squares fit.  Metrics are computed only over finite, non-NoData pixels.
    """

    depth = np.asarray(depth, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    ground_truth_valid_mask = np.asarray(ground_truth_valid_mask, dtype=bool)
    if depth.shape != ground_truth.shape or depth.shape != ground_truth_valid_mask.shape:
        raise ValueError("Depth, ground truth, and validity mask must share a pixel grid")

    valid = ground_truth_valid_mask & np.isfinite(depth) & np.isfinite(ground_truth)
    if source_valid_mask is not None:
        source_valid_mask = np.asarray(source_valid_mask, dtype=bool)
        if source_valid_mask.shape != depth.shape:
            raise ValueError("Source validity mask must share the depth pixel grid")
        valid &= source_valid_mask

    valid_count = int(valid.sum())
    if valid_count < 3:
        raise ValueError(
            "At least three finite, non-NoData aligned pixels are required for evaluation"
        )

    fit = fit_affine_calibration(
        depth,
        depth[valid],
        ground_truth[valid],
        source="ground_truth_dsm",
        label=BENCHMARK_CALIBRATION_LABEL,
        benchmark_only=True,
        output_valid_mask=source_valid_mask,
    )
    calibrated = fit.calibrated_height
    residual = calibrated[valid].astype(np.float64) - ground_truth[valid].astype(
        np.float64
    )
    absolute_error = np.full(depth.shape, np.nan, dtype=np.float32)
    absolute_error[valid] = np.abs(residual).astype(np.float32)

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    calibrated_correlation = pearson_correlation(calibrated[valid], ground_truth[valid])
    total_pixels = int(depth.size)

    metrics: dict[str, Any] = {
        "label": BENCHMARK_CALIBRATION_LABEL,
        "valid_pixels": valid_count,
        "total_pixels": total_pixels,
        "valid_fraction": float(valid_count / total_pixels),
        "evaluation_type": "benchmark_full_ground_truth_refit",
        "evaluation_scope": "same_scene_benchmark_refit",
        "calibration_refit": True,
        "raw_pearson_correlation": fit.raw_correlation,
        "negative_depth_pearson_correlation": fit.inverted_correlation,
        "relative_pearson_correlation": fit.relative_correlation,
        "calibrated_mae": _finite_number(mae),
        "calibrated_rmse": _finite_number(rmse),
        "calibrated_pearson_correlation": _finite_number(calibrated_correlation),
        "mae": _finite_number(mae),
        "rmse": _finite_number(rmse),
        "pearson_correlation": _finite_number(calibrated_correlation),
    }
    return EvaluationResult(
        metrics=metrics,
        calibration=fit.calibration,
        calibrated_height=calibrated,
        absolute_error=absolute_error,
        valid_mask=valid,
    )
