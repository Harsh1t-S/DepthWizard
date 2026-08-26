from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.pipeline as pipeline
from backend.app import create_app
from backend.pipeline import analyze_bytes
from ml.depth_anything import DepthEstimator


def _gradient_image(width: int, height: int) -> Image.Image:
    xx, yy = np.meshgrid(
        np.linspace(0, 255, width, dtype=np.uint8),
        np.linspace(0, 255, height, dtype=np.uint8),
    )
    rgb = np.stack((xx, yy, ((xx.astype(int) + yy) // 2).astype(np.uint8)), axis=-1)
    return Image.fromarray(rgb, mode="RGB")


class DeterministicQualityEstimator(DepthEstimator):
    """Exercise quality orchestration without loading Torch or model weights."""

    def __init__(self, full_size: tuple[int, int]) -> None:
        super().__init__(model_id="test-model", device="cpu", max_input_size=256)
        self.full_size = full_size
        self.calls: list[tuple[int, int]] = []

    def _patch_multiple(self) -> int:
        return 14

    def _predict_once(self, source: Image.Image) -> tuple[np.ndarray, int, int]:
        self.calls.append(source.size)
        rgb = np.asarray(source, dtype=np.float32) / np.float32(255.0)
        depth = rgb[..., 0] + np.float32(2.0) * rgb[..., 1]
        # Mimic arbitrary per-tile monocular scale/offset. The quality path must
        # align this local prediction to the global map before blending it.
        if source.size != self.full_size:
            depth = np.float32(2.5) * depth + np.float32(7.0)
        return depth.astype(np.float32), source.width, source.height


def test_fast_mode_remains_one_global_pass() -> None:
    image = _gradient_image(520, 360)
    estimator = DeterministicQualityEstimator(image.size)

    depth, info = estimator.predict(image)

    assert depth.shape == (360, 520)
    assert len(estimator.calls) == 1
    assert info.quality_mode == "fast"
    assert info.inference_passes == 1
    assert info.tiled is False
    assert info.tile_count == 0


def test_quality_mode_blends_overlapping_tiles_on_large_image() -> None:
    image = _gradient_image(520, 360)
    estimator = DeterministicQualityEstimator(image.size)
    expected_rgb = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    expected = expected_rgb[..., 0] + np.float32(2.0) * expected_rgb[..., 1]

    depth, info = estimator.predict(image, quality_mode="quality")

    assert depth.shape == expected.shape
    assert np.isfinite(depth).all()
    assert info.quality_mode == "quality"
    assert info.tiled is True
    assert info.tile_count == 6
    assert info.inference_passes == 7
    assert len(estimator.calls) == info.inference_passes
    assert np.max(np.abs(depth - expected)) < 1e-4


def test_quality_mode_uses_flip_consistency_for_small_image() -> None:
    image = _gradient_image(120, 80)
    estimator = DeterministicQualityEstimator(image.size)
    expected_rgb = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    expected = expected_rgb[..., 0] + np.float32(2.0) * expected_rgb[..., 1]

    depth, info = estimator.predict(image, quality_mode="quality")

    assert info.tiled is False
    assert info.inference_passes == 2
    assert len(estimator.calls) == 2
    assert np.max(np.abs(depth - expected)) < 1e-6


def test_pipeline_exposes_quality_inference_metadata(tmp_path) -> None:
    image = _gradient_image(120, 80)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    estimator = DeterministicQualityEstimator(image.size)

    result = analyze_bytes(
        encoded.getvalue(),
        "scene.png",
        quality_mode="quality",
        estimator=estimator,
        artifact_root=tmp_path,
    )

    assert result["inference"] == {
        "quality_mode": "quality",
        "passes": 2,
        "tiled": False,
        "tile_count": 0,
        "bounded_width": 120,
        "bounded_height": 80,
    }
    assert any("horizontally flipped" in notice for notice in result["notices"])


def test_api_accepts_quality_mode_form_field(tmp_path, monkeypatch) -> None:
    image = _gradient_image(120, 80)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    estimator = DeterministicQualityEstimator(image.size)
    monkeypatch.setattr(pipeline, "get_depth_estimator", lambda: estimator)
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/analyze",
        files={"image": ("scene.png", encoded.getvalue(), "image/png")},
        data={"quality_mode": "quality"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["inference"]["quality_mode"] == "quality"
    assert response.json()["inference"]["passes"] == 2


def test_quality_mode_rejects_unknown_value() -> None:
    estimator = DeterministicQualityEstimator((120, 80))

    with pytest.raises(ValueError, match="quality_mode must be one of"):
        estimator.predict(_gradient_image(120, 80), quality_mode="ultra")
