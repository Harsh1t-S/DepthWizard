"""Lazy Depth Anything V2 inference backed by Hugging Face and PyTorch.

There is deliberately no synthetic or heuristic fallback in this module.  If the
model cannot be loaded, live analysis fails with a clear error instead of
returning an output that only looks like model inference.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DEFAULT_MAX_INPUT_SIZE = 1024
DEFAULT_PATCH_MULTIPLE = 14
QUALITY_MODES = ("fast", "quality")
QUALITY_TILE_OVERLAP_FRACTION = 0.25
QUALITY_GLOBAL_BLEND = 0.15


class ModelLoadError(RuntimeError):
    """Raised when the configured depth model cannot be loaded."""


class ModelInferenceError(RuntimeError):
    """Raised when an already configured model cannot process an image."""


def select_device() -> str:
    """Choose CUDA, then Apple MPS, and finally CPU."""

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        # Import/runtime probing should not prevent the health endpoint loading.
        pass
    return "cpu"


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 256:
        raise ValueError(f"{name} must be at least 256")
    return value


@dataclass(frozen=True)
class PredictionInfo:
    """Useful inference metadata exposed to the pipeline."""

    model_id: str
    device: str
    inference_width: int
    inference_height: int
    quality_mode: str = "fast"
    inference_passes: int = 1
    tiled: bool = False
    tile_count: int = 0


class DepthEstimator:
    """Depth Anything V2 estimator with lazy, thread-safe model loading."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        max_input_size: int | None = None,
    ) -> None:
        self.model_id = model_id or os.getenv("DEPTHWIZARD_MODEL_ID", DEFAULT_MODEL_ID)
        self.device = device or select_device()
        if max_input_size is None:
            self.max_input_size = _positive_int_env(
                "DEPTHWIZARD_MAX_INPUT_SIZE", DEFAULT_MAX_INPUT_SIZE
            )
        else:
            if int(max_input_size) < 256:
                raise ValueError("max_input_size must be at least 256 pixels")
            self.max_input_size = int(max_input_size)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _patch_multiple(self) -> int:
        """Read the model stride, falling back to V2's documented 14 pixels."""

        config = getattr(self._model, "config", None)
        backbone_config = getattr(config, "backbone_config", None)
        raw = getattr(config, "patch_size", None) or getattr(
            backbone_config, "patch_size", DEFAULT_PATCH_MULTIPLE
        )
        if isinstance(raw, (tuple, list)):
            raw = max(raw)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = DEFAULT_PATCH_MULTIPLE
        return max(1, value)

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                processor = AutoImageProcessor.from_pretrained(self.model_id)
                model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
                model.to(self.device)
                model.eval()
            except Exception as exc:
                raise ModelLoadError(
                    f"Could not load {self.model_id!r} on {self.device}. "
                    "Ensure the model is cached or network access is available. "
                    f"Underlying error: {exc}"
                ) from exc
            self._processor = processor
            self._model = model

    def _predict_once(
        self, source: Image.Image
    ) -> tuple[np.ndarray, int, int]:
        """Run one bounded model pass and return depth on ``source``'s grid."""

        self._ensure_loaded()
        source_width, source_height = source.size
        # Depth Anything V2 uses a 14-pixel patch stride.  Resize exactly once
        # to patch-aligned dimensions bounded by max_input_size, then disable
        # the Hugging Face processor's default 518px resize below.
        patch_multiple = self._patch_multiple()
        scale = min(1.0, self.max_input_size / max(source_width, source_height))
        inference_width = max(
            patch_multiple,
            int(round(source_width * scale / patch_multiple)) * patch_multiple,
        )
        inference_height = max(
            patch_multiple,
            int(round(source_height * scale / patch_multiple)) * patch_multiple,
        )
        while max(inference_width, inference_height) > self.max_input_size:
            if inference_width >= inference_height and inference_width > patch_multiple:
                inference_width -= patch_multiple
            elif inference_height > patch_multiple:
                inference_height -= patch_multiple
            else:
                break
        working = source
        if (inference_width, inference_height) != source.size:
            working = source.resize(
                (inference_width, inference_height), Image.Resampling.LANCZOS
            )

        try:
            import torch
            import torch.nn.functional as functional

            assert self._processor is not None
            assert self._model is not None
            inputs = self._processor(
                images=working,
                return_tensors="pt",
                do_resize=False,
            )
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None or pixel_values.ndim != 4:
                raise ModelInferenceError("Image processor returned no pixel tensor")
            actual_height = int(pixel_values.shape[-2])
            actual_width = int(pixel_values.shape[-1])
            # A lock avoids unsafe concurrent access to a single GPU/MPS model and
            # bounds peak memory when multiple API requests arrive together.
            with self._inference_lock, torch.inference_mode():
                device_inputs = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                prediction = self._model(**device_inputs).predicted_depth
                prediction = functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(source_height, source_width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                depth = prediction.detach().float().cpu().numpy().astype(np.float32)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelInferenceError(
                f"Depth Anything inference failed on {self.device}: {exc}"
            ) from exc

        if depth.shape != (source_height, source_width) or not np.isfinite(depth).any():
            raise ModelInferenceError("Model returned an invalid depth map")

        return depth, actual_width, actual_height

    @staticmethod
    def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
        """Return starts that cover an axis fully, including its final edge."""

        if length <= tile_size:
            return [0]
        stride = max(1, tile_size - overlap)
        starts = list(range(0, length - tile_size + 1, stride))
        final_start = length - tile_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    @staticmethod
    def _blend_weights(height: int, width: int) -> np.ndarray:
        """Create a feathering window with a non-zero border for image edges."""

        def axis_weights(length: int) -> np.ndarray:
            if length <= 2:
                return np.ones(length, dtype=np.float32)
            phase = np.linspace(0.0, np.pi, length, dtype=np.float32)
            return np.maximum(np.sin(phase), np.float32(0.05))

        return np.outer(axis_weights(height), axis_weights(width)).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _match_relative_scale(
        prediction: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Align a local relative-depth tile to the global prediction robustly.

        Monocular depth has arbitrary scale and offset.  Matching the 5th/50th/
        95th percentiles before blending prevents independently inferred tiles
        from creating artificial height steps at their boundaries.
        """

        valid = np.isfinite(prediction) & np.isfinite(reference)
        if int(valid.sum()) < 32:
            return prediction
        predicted_values = prediction[valid].astype(np.float64, copy=False)
        reference_values = reference[valid].astype(np.float64, copy=False)
        predicted_low, predicted_mid, predicted_high = np.quantile(
            predicted_values, (0.05, 0.5, 0.95)
        )
        reference_low, reference_mid, reference_high = np.quantile(
            reference_values, (0.05, 0.5, 0.95)
        )
        predicted_span = predicted_high - predicted_low
        reference_span = reference_high - reference_low
        epsilon = np.finfo(np.float32).eps
        if predicted_span <= epsilon or reference_span <= epsilon:
            scale = 1.0
        else:
            # A positive scale preserves the depth orientation of the global pass.
            scale = reference_span / predicted_span
        offset = reference_mid - scale * predicted_mid
        return (prediction * np.float32(scale) + np.float32(offset)).astype(
            np.float32, copy=False
        )

    def _predict_quality(
        self, source: Image.Image, global_depth: np.ndarray
    ) -> tuple[np.ndarray, int, bool, int]:
        """Add flip consistency or overlap-tiled local detail to a global pass."""

        source_width, source_height = source.size
        if max(source_width, source_height) <= self.max_input_size:
            flipped = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped_depth, _, _ = self._predict_once(flipped)
            flipped_depth = np.ascontiguousarray(np.fliplr(flipped_depth))
            flipped_depth = self._match_relative_scale(flipped_depth, global_depth)
            return (
                ((global_depth + flipped_depth) * np.float32(0.5)).astype(
                    np.float32, copy=False
                ),
                2,
                False,
                0,
            )

        tile_size = self.max_input_size
        overlap = min(
            tile_size - 1,
            max(
                self._patch_multiple() * 4,
                int(tile_size * QUALITY_TILE_OVERLAP_FRACTION),
            ),
        )
        x_starts = self._tile_starts(source_width, tile_size, overlap)
        y_starts = self._tile_starts(source_height, tile_size, overlap)
        accumulated = np.zeros((source_height, source_width), dtype=np.float32)
        total_weight = np.zeros_like(accumulated)
        tile_count = 0

        for y_start in y_starts:
            y_stop = min(source_height, y_start + tile_size)
            for x_start in x_starts:
                x_stop = min(source_width, x_start + tile_size)
                tile = source.crop((x_start, y_start, x_stop, y_stop))
                tile_depth, _, _ = self._predict_once(tile)
                global_tile = global_depth[y_start:y_stop, x_start:x_stop]
                tile_depth = self._match_relative_scale(tile_depth, global_tile)
                weights = self._blend_weights(tile_depth.shape[0], tile_depth.shape[1])
                accumulated[y_start:y_stop, x_start:x_stop] += tile_depth * weights
                total_weight[y_start:y_stop, x_start:x_stop] += weights
                tile_count += 1

        np.divide(
            accumulated,
            total_weight,
            out=accumulated,
            where=total_weight > 0,
        )
        # Retain a small contribution from the global prediction so local tiles
        # cannot erase broad scene structure while contributing finer detail.
        accumulated *= np.float32(1.0 - QUALITY_GLOBAL_BLEND)
        accumulated += global_depth * np.float32(QUALITY_GLOBAL_BLEND)
        return accumulated, tile_count + 1, True, tile_count

    def predict(
        self, image: Image.Image, quality_mode: str = "fast"
    ) -> tuple[np.ndarray, PredictionInfo]:
        """Return relative depth, optionally using a slower aerial-quality path."""

        normalized_quality = quality_mode.strip().lower()
        if normalized_quality not in QUALITY_MODES:
            raise ValueError(
                f"quality_mode must be one of {', '.join(QUALITY_MODES)}, "
                f"got {quality_mode!r}"
            )
        source = image.convert("RGB")
        source_width, source_height = source.size
        if source_width < 1 or source_height < 1:
            raise ModelInferenceError("Input image has no pixels")

        depth, actual_width, actual_height = self._predict_once(source)
        inference_passes = 1
        tiled = False
        tile_count = 0
        if normalized_quality == "quality":
            depth, inference_passes, tiled, tile_count = self._predict_quality(
                source, depth
            )

        if depth.shape != (source_height, source_width) or not np.isfinite(depth).any():
            raise ModelInferenceError("Model returned an invalid depth map")

        return depth, PredictionInfo(
            model_id=self.model_id,
            device=self.device,
            inference_width=actual_width,
            inference_height=actual_height,
            quality_mode=normalized_quality,
            inference_passes=inference_passes,
            tiled=tiled,
            tile_count=tile_count,
        )


_singleton: DepthEstimator | None = None
_singleton_lock = threading.Lock()


def get_depth_estimator() -> DepthEstimator:
    """Return the process-wide estimator without loading its model eagerly."""

    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = DepthEstimator()
    return _singleton


def is_depth_estimator_loaded() -> bool:
    """Report singleton load state without creating or loading it."""

    return bool(_singleton is not None and _singleton.loaded)
