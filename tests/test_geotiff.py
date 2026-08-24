from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


rasterio = pytest.importorskip("rasterio")
from rasterio.io import MemoryFile  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from backend.pipeline import analyze_bytes  # noqa: E402


class GeoFakeEstimator:
    model_id = "geotiff-test-estimator"
    device = "cpu"

    def predict(self, image: Image.Image):
        width, height = image.size
        values = np.arange(width * height, dtype=np.float32).reshape(height, width)
        return -values, {
            "model_id": self.model_id,
            "device": self.device,
            "inference_width": width,
            "inference_height": height,
        }


def _geotiff_bytes(values: np.ndarray, count: int, nodata=None) -> bytes:
    height, width = values.shape[-2:]
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": str(values.dtype),
        "crs": "EPSG:32643",
        "transform": from_origin(500000.0, 2000000.0, 2.0, 2.0),
        "nodata": nodata,
    }
    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            if count == 1:
                dataset.write(values, 1)
            else:
                dataset.write(values)
            dataset.update_tags(SOURCE_FIXTURE="unit-test")
        return memory_file.read()


def test_calibrated_dsm_preserves_source_geospatial_grid(tmp_path) -> None:
    height, width = 12, 18
    pixels = np.arange(width * height, dtype=np.uint16).reshape(height, width)
    source_bands = np.stack((pixels, pixels + 10, pixels + 20))
    source_bytes = _geotiff_bytes(source_bands, count=3)
    ground_truth = (100.0 + 0.5 * pixels).astype(np.float32)
    ground_truth[0, 0] = -9999.0
    ground_truth_bytes = _geotiff_bytes(ground_truth, count=1, nodata=-9999.0)

    result = analyze_bytes(
        source_bytes,
        "source.tif",
        ground_truth_bytes,
        "truth.tif",
        estimator=GeoFakeEstimator(),
        artifact_root=tmp_path,
    )

    assert result["geospatial"]["crs"] == "EPSG:32643"
    output_path = Path(result["artifacts"]["calibrated_dsm_geotiff"])
    assert output_path.is_file()
    with rasterio.open(output_path) as exported:
        assert exported.crs.to_string() == "EPSG:32643"
        assert exported.transform == from_origin(500000.0, 2000000.0, 2.0, 2.0)
        assert (exported.width, exported.height, exported.count) == (width, height, 1)
        assert exported.dtypes == ("float32",)
        assert exported.tags()["SOURCE_FIXTURE"] == "unit-test"
        assert exported.tags()["DEPTHWIZARD_CALIBRATION"] == "benchmark_only_affine"
        assert exported.dataset_mask()[0, 0] == 255
