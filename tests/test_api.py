from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.pipeline as pipeline
from backend.app import create_app
from backend.demo import DEMO_NOTICE


class FakeEstimator:
    model_id = "unit-test-depth-estimator"
    device = "cpu"

    def predict(self, image: Image.Image):
        width, height = image.size
        xx, yy = np.meshgrid(
            np.linspace(0.0, 1.0, width, dtype=np.float32),
            np.linspace(0.0, 1.0, height, dtype=np.float32),
        )
        depth = -(xx + 2.0 * yy)
        return depth, {
            "model_id": self.model_id,
            "device": self.device,
            "inference_width": width,
            "inference_height": height,
        }


def _png_bytes(width: int = 24, height: int = 16) -> bytes:
    xx, yy = np.meshgrid(
        np.arange(width, dtype=np.uint8), np.arange(height, dtype=np.uint8)
    )
    rgb = np.stack((xx * 7, yy * 11, (xx + yy) * 5), axis=-1).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _npy_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, values.astype(np.float32), allow_pickle=False)
    return output.getvalue()


def test_health_does_not_load_model_and_demo_has_absolute_artifacts(
    tmp_path, monkeypatch
) -> None:
    import backend.app as app_module

    monkeypatch.setattr(app_module, "is_depth_estimator_loaded", lambda: False)
    monkeypatch.setattr(app_module, "select_device", lambda: "cpu")
    client = TestClient(create_app(tmp_path))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert health.json()["live_inference"] == "real_model_only"

    response = client.get("/api/demo")
    assert response.status_code == 200
    payload = response.json()
    assert payload["demo"] is True
    assert payload["precomputed"] is True
    assert DEMO_NOTICE in payload["notices"]
    assert payload["urls"]["depth"].startswith("http://testserver/artifacts/")
    assert len(payload["depth_grid"]["values"]) == (
        payload["depth_grid"]["width"] * payload["depth_grid"]["height"]
    )
    assert client.get(payload["urls"]["depth"]).status_code == 200


def test_live_analysis_uses_mock_only_and_serves_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "get_depth_estimator", lambda: FakeEstimator())
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/analyze",
        files={"image": ("scene.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["demo"] is False
    assert payload["precomputed"] is False
    assert payload["model"] == FakeEstimator.model_id
    assert payload["mode"] == "relative_depth"
    assert payload["metrics"] is None
    assert payload["calibration"] is None
    assert "metrics_json" in payload["artifacts"]
    assert payload["input"] == {"width": 24, "height": 16, "filename": "scene.png"}
    assert 160 <= max(payload["depth_grid"]["width"], payload["depth_grid"]["height"]) <= 256
    for artifact_url in payload["artifacts"].values():
        assert artifact_url.startswith("http://testserver/artifacts/")
        assert client.get(artifact_url).status_code == 200


def test_live_evaluation_contract_and_canonical_multipart_name(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pipeline, "get_depth_estimator", lambda: FakeEstimator())
    client = TestClient(create_app(tmp_path))
    width, height = 24, 16
    xx, yy = np.meshgrid(
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        np.linspace(0.0, 1.0, height, dtype=np.float32),
    )
    dsm = 42.0 + 8.0 * (xx + 2.0 * yy)

    response = client.post(
        "/api/analyze",
        files={
            "image": ("scene.png", _png_bytes(width, height), "image/png"),
            "ground_truth_dsm": ("truth.npy", _npy_bytes(dsm), "application/octet-stream"),
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mode"] == "benchmark_calibrated_dsm"
    assert payload["calibration"]["benchmark_only"] is True
    assert payload["calibration"]["orientation"] == "negative_depth"
    assert payload["calibration"]["a"] == pytest.approx(8.0, rel=1e-5)
    assert payload["metrics"]["calibrated_rmse"] < 1e-4
    assert payload["urls"]["ground_truth"]
    assert payload["urls"]["error"]
    assert "metrics_json" in payload["artifacts"]
    assert "calibrated_dsm_geotiff" not in payload["artifacts"]
