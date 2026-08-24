"""Stable backend-facing imports for the depth estimator."""

from ml.depth_anything import (
    DEFAULT_MODEL_ID,
    DepthEstimator,
    ModelInferenceError,
    ModelLoadError,
    PredictionInfo,
    get_depth_estimator,
    is_depth_estimator_loaded,
    select_device,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DepthEstimator",
    "ModelInferenceError",
    "ModelLoadError",
    "PredictionInfo",
    "get_depth_estimator",
    "is_depth_estimator_loaded",
    "select_device",
]
