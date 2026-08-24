"""Relative-depth orientation and benchmark-only affine calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BENCHMARK_CALIBRATION_LABEL = (
    "Benchmark-only affine calibration against supplied ground truth; "
    "not available for unseen terrain."
)


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    calibration: dict[str, Any]
    calibrated_height: np.ndarray
    absolute_error: np.ndarray
    valid_mask: np.ndarray


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

    x_raw = depth[valid].astype(np.float64)
    y = ground_truth[valid].astype(np.float64)
    raw_correlation = pearson_correlation(x_raw, y)
    inverted_correlation = pearson_correlation(-x_raw, y)

    use_negative = (
        inverted_correlation is not None
        and (raw_correlation is None or inverted_correlation > raw_correlation)
    )
    orientation_sign = -1.0 if use_negative else 1.0
    orientation = "negative_depth" if use_negative else "depth"
    oriented = depth.astype(np.float64) * orientation_sign
    x = oriented[valid]
    relative_correlation = pearson_correlation(x, y)

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    denominator = float(np.dot(x_centered, x_centered))
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        raise ValueError("Relative depth is constant over valid DSM pixels; calibration failed")
    scale = float(np.dot(x_centered, y - y_mean) / denominator)
    offset = float(y_mean - scale * x_mean)

    calibrated = (scale * oriented + offset).astype(np.float32)
    residual = calibrated[valid].astype(np.float64) - y
    absolute_error = np.full(depth.shape, np.nan, dtype=np.float32)
    absolute_error[valid] = np.abs(residual).astype(np.float32)

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    calibrated_correlation = pearson_correlation(calibrated[valid], y)
    total_pixels = int(depth.size)

    metrics: dict[str, Any] = {
        "label": BENCHMARK_CALIBRATION_LABEL,
        "valid_pixels": valid_count,
        "total_pixels": total_pixels,
        "valid_fraction": float(valid_count / total_pixels),
        "raw_pearson_correlation": _finite_number(raw_correlation),
        "negative_depth_pearson_correlation": _finite_number(inverted_correlation),
        "relative_pearson_correlation": _finite_number(relative_correlation),
        "calibrated_mae": _finite_number(mae),
        "calibrated_rmse": _finite_number(rmse),
        "calibrated_pearson_correlation": _finite_number(calibrated_correlation),
    }
    calibration: dict[str, Any] = {
        "method": "affine_least_squares",
        "orientation": orientation,
        "scale": scale,
        "shift": offset,
        "label": BENCHMARK_CALIBRATION_LABEL,
        "benchmark_only": True,
        "orientation_sign": orientation_sign,
        "equation": "height = a * oriented_depth + b",
        "a": scale,
        "b": offset,
        "coefficients": {"a": scale, "b": offset},
    }
    return EvaluationResult(
        metrics=metrics,
        calibration=calibration,
        calibrated_height=calibrated,
        absolute_error=absolute_error,
        valid_mask=valid,
    )
