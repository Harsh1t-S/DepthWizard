from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.pipeline as pipeline
from backend.app import create_app
from backend.calibration import parse_gcps_bytes, sample_depth_at_gcps


class CalibrationEstimator:
    model_id = "calibration-test-estimator"
    device = "cpu"

    def predict(self, image: Image.Image):
        width, height = image.size
        xx, yy = np.meshgrid(
            np.linspace(0.0, 1.0, width, dtype=np.float32),
            np.linspace(0.0, 1.0, height, dtype=np.float32),
        )
        return -(xx + 2.0 * yy), {
            "model_id": self.model_id,
            "device": self.device,
            "inference_width": width,
            "inference_height": height,
        }


def _png_bytes(width: int, height: int) -> bytes:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)
    rgb[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _npy_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return output.getvalue()


def _base(width: int, height: int) -> np.ndarray:
    xx, yy = np.meshgrid(
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        np.linspace(0.0, 1.0, height, dtype=np.float32),
    )
    return xx + 2.0 * yy


def test_reference_dem_calibrates_and_holdout_does_not_refit(tmp_path: Path) -> None:
    width, height = 14, 10
    base = _base(width, height)
    reference = 50.0 + 7.0 * base
    ground_truth = reference + 2.5 * np.square(base)

    result = pipeline.analyze_bytes(
        _png_bytes(width, height),
        "scene.png",
        _npy_bytes(ground_truth),
        "holdout.npy",
        reference_dem_bytes=_npy_bytes(reference),
        reference_dem_filename="coarse-reference.npy",
        estimator=CalibrationEstimator(),
        artifact_root=tmp_path,
    )

    assert result["mode"] == "reference_dem_calibrated_dsm"
    assert result["calibration"]["source"] == "reference_dem"
    assert result["calibration"]["benchmark_only"] is False
    assert result["calibration"]["orientation"] == "negative_depth"
    assert result["calibration"]["scale"] == pytest.approx(7.0, rel=1e-5)
    assert result["calibration"]["shift"] == pytest.approx(50.0, rel=1e-5)
    assert result["metrics"]["calibration_refit"] is False
    assert result["metrics"]["evaluation_scope"] == "holdout_fixed_calibration"
    assert result["metrics"]["rmse"] > 1.0
    assert result["urls"]["reference_dem"]
    assert Path(result["artifacts"]["calibrated_dsm_npy"]).is_file()


def test_gcp_json_aliases_fit_full_resolution_surface(tmp_path: Path) -> None:
    width, height = 12, 8
    base = _base(width, height)
    coordinates = [(0, 0), (11, 0), (0, 7), (11, 7), (5, 3)]
    points = [
        {"pixel_x": x, "pixel_y": y, "z": float(100.0 + 5.0 * base[y, x])}
        for x, y in coordinates
    ]
    result = pipeline.analyze_bytes(
        _png_bytes(width, height),
        "scene.png",
        gcps_bytes=json.dumps({"coordinate_space": "pixel", "points": points}).encode(),
        gcps_filename="anchors.json",
        estimator=CalibrationEstimator(),
        artifact_root=tmp_path,
    )

    assert result["mode"] == "gcp_calibrated_dsm"
    assert result["metrics"] is None
    assert result["calibration"]["source"] == "gcps"
    assert result["calibration"]["reference_samples"] == len(points)
    assert result["calibration"]["scale"] == pytest.approx(5.0, rel=1e-5)
    assert result["calibration"]["shift"] == pytest.approx(100.0, rel=1e-5)
    exported = np.load(result["artifacts"]["calibrated_dsm_npy"])
    assert exported.shape == (height, width)
    assert np.allclose(exported, 100.0 + 5.0 * base, atol=2e-5)


def test_gcp_csv_and_fractional_bilinear_sampling() -> None:
    points = parse_gcps_bytes(
        b"col,row,height\n0.5,0.5,10\n2,0,20\n0,2,30\n",
        "points.csv",
    )
    depth = np.arange(9, dtype=np.float32).reshape(3, 3)
    sampled, elevations = sample_depth_at_gcps(depth, points, method="bilinear")

    assert sampled.tolist() == pytest.approx([2.0, 2.0, 6.0])
    assert elevations.tolist() == [10.0, 20.0, 30.0]


def test_gcp_validation_rejects_non_pixel_duplicate_and_out_of_bounds_points() -> None:
    with pytest.raises(ValueError, match="coordinate_space"):
        parse_gcps_bytes(
            b'{"coordinate_space":"map","points":[{"x":1,"y":2,"elevation":3},'
            b'{"x":2,"y":3,"elevation":4},{"x":3,"y":4,"elevation":5}]}',
            "map.json",
        )
    with pytest.raises(ValueError, match="distinct"):
        parse_gcps_bytes(
            b"x,y,elevation\n0,0,1\n0,0,2\n0,0,3\n",
            "duplicates.csv",
        )
    points = parse_gcps_bytes(
        b"x,y,elevation\n0,0,1\n1,1,2\n4,1,3\n",
        "outside.csv",
    )
    with pytest.raises(ValueError, match="outside"):
        sample_depth_at_gcps(np.ones((3, 3), dtype=np.float32), points)


def test_api_rejects_two_deployment_references_before_inference(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "get_depth_estimator", lambda: CalibrationEstimator())
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/analyze",
        files={
            "image": ("scene.png", _png_bytes(8, 6), "image/png"),
            "reference_dem": ("reference.npy", _npy_bytes(_base(8, 6))),
            "gcps": ("points.csv", b"x,y,elevation\n0,0,1\n1,0,2\n0,1,3\n"),
        },
    )

    assert response.status_code == 422
    assert "reference_dem or gcps" in response.json()["detail"]

    base = _base(8, 6)
    rows = ["x,y,elevation"]
    for x, y in ((0, 0), (7, 0), (0, 5), (7, 5)):
        rows.append(f"{x},{y},{20.0 + 4.0 * base[y, x]}")
    calibrated = client.post(
        "/api/analyze",
        files={
            "image": ("scene.png", _png_bytes(8, 6), "image/png"),
            "gcps": ("points.csv", "\n".join(rows).encode(), "text/csv"),
        },
        data={"gcp_sampling": "nearest"},
    )
    assert calibrated.status_code == 200, calibrated.text
    payload = calibrated.json()
    assert payload["mode"] == "gcp_calibrated_dsm"
    assert payload["calibration"]["sampling"] == "nearest"
    assert payload["metrics"] is None
