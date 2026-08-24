from __future__ import annotations

import numpy as np
import pytest

from backend.evaluation import evaluate_relative_depth
from backend.raster_io import GroundTruthRaster, ImageRaster, align_ground_truth


def test_negative_depth_orientation_and_affine_calibration() -> None:
    base = np.arange(30, dtype=np.float32).reshape(5, 6) / 10.0
    depth = -base
    ground_truth = 7.0 + 3.0 * base
    valid = np.ones(base.shape, dtype=bool)
    valid[0, 0] = False
    ground_truth[0, 0] = np.nan

    result = evaluate_relative_depth(depth, ground_truth, valid)

    assert result.calibration["orientation"] == "negative_depth"
    assert result.calibration["orientation_sign"] == -1.0
    assert result.calibration["a"] == pytest.approx(3.0, rel=1e-6)
    assert result.calibration["b"] == pytest.approx(7.0, rel=1e-6)
    assert result.metrics["raw_pearson_correlation"] == pytest.approx(-1.0)
    assert result.metrics["relative_pearson_correlation"] == pytest.approx(1.0)
    assert result.metrics["calibrated_rmse"] == pytest.approx(0.0, abs=2e-6)
    assert result.metrics["valid_pixels"] == base.size - 1


def test_ground_truth_resize_is_explicit_and_masks_nodata() -> None:
    source = ImageRaster(
        filename="image.png",
        rgb=np.zeros((8, 12, 3), dtype=np.uint8),
    )
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    valid = np.ones(values.shape, dtype=bool)
    valid[0, 0] = False
    ground_truth = GroundTruthRaster("dsm.npy", values, valid)

    aligned, aligned_valid, notices = align_ground_truth(ground_truth, source)

    assert aligned.shape == (8, 12)
    assert aligned_valid.shape == (8, 12)
    assert not aligned_valid[0, 0]
    assert any("resized from 6x4 to 12x8" in notice for notice in notices)
