"""Machine-learning components for DepthWizard."""

from .depth_anything import (
    DEFAULT_MODEL_ID,
    DepthEstimator,
    ModelInferenceError,
    ModelLoadError,
    get_depth_estimator,
    is_depth_estimator_loaded,
    select_device,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DepthEstimator",
    "ModelInferenceError",
    "ModelLoadError",
    "get_depth_estimator",
    "is_depth_estimator_loaded",
    "select_device",
]
