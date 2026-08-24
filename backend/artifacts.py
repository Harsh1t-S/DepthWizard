"""Artifact serialization and compact mesh-grid helpers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .raster_io import ImageRaster


ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def default_artifact_root() -> Path:
    configured = os.getenv("DEPTHWIZARD_ARTIFACT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parent / "artifacts").resolve()


def normalize_for_preview(
    values: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Percentile-normalize a raster into a finite [0, 1] preview."""

    array = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(array)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    output = np.zeros(array.shape, dtype=np.float32)
    if not valid.any():
        return output
    low, high = np.percentile(array[valid], (2.0, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(array[valid].min())
        high = float(array[valid].max())
    if high <= low:
        output[valid] = 0.5
        return output
    output[valid] = np.clip((array[valid] - low) / (high - low), 0.0, 1.0)
    return output


def _colorize(normalized: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Apply a small perceptual blue-to-yellow color ramp without matplotlib."""

    values = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    stops = np.array([0.0, 0.22, 0.48, 0.72, 1.0], dtype=np.float32)
    colors = np.array(
        [
            [24, 35, 82],
            [31, 104, 165],
            [37, 184, 164],
            [221, 213, 83],
            [249, 134, 38],
        ],
        dtype=np.float32,
    )
    rgb = np.stack(
        [np.interp(values, stops, colors[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype(np.uint8)
    if valid_mask is not None:
        rgb[~np.asarray(valid_mask, dtype=bool)] = np.array([15, 20, 31], dtype=np.uint8)
    return rgb


def make_mesh_grid(
    heights: np.ndarray,
    target_long_edge: int = 192,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Resize a height raster to a roughly 160--256-cell mesh long edge."""

    if not 160 <= target_long_edge <= 256:
        raise ValueError("Mesh target long edge must be between 160 and 256")
    array = np.asarray(heights, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Mesh heights must be a two-dimensional array")
    height, width = array.shape
    scale = target_long_edge / max(width, height)
    mesh_width = max(2, int(round(width * scale)))
    mesh_height = max(2, int(round(height * scale)))

    finite = np.isfinite(array)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    fill = float(np.median(array[finite])) if finite.any() else 0.0
    resize_values = array.copy()
    resize_values[~finite] = fill
    resized = np.asarray(
        Image.fromarray(resize_values, mode="F").resize(
            (mesh_width, mesh_height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    # Keep JSON compact and, critically, free of NaN/Infinity tokens.
    resized = np.nan_to_num(resized, nan=fill, posinf=fill, neginf=fill)
    payload: dict[str, Any] = {
        "width": mesh_width,
        "height": mesh_height,
        "values": np.round(resized.reshape(-1), 6).tolist(),
    }
    if not finite.all():
        resized_valid = np.asarray(
            Image.fromarray(finite.astype(np.uint8) * 255).resize(
                (mesh_width, mesh_height), Image.Resampling.NEAREST
            )
        ) > 0
        payload["valid_mask"] = resized_valid.reshape(-1).tolist()
    return payload


class ArtifactWriter:
    """Write one job's outputs and generate public or local references."""

    def __init__(
        self,
        root: Path,
        job_id: str,
        public_base_url: str | None = None,
    ) -> None:
        if not ARTIFACT_ID_PATTERN.fullmatch(job_id):
            raise ValueError("Invalid artifact job identifier")
        self.root = Path(root).resolve()
        self.job_id = job_id
        self.directory = self.root / job_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None

    def path(self, filename: str) -> Path:
        if Path(filename).name != filename:
            raise ValueError("Artifact filename must not contain a path")
        return self.directory / filename

    def reference(self, filename: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{self.job_id}/{filename}"
        return str(self.path(filename).resolve())

    def write_rgb(self, filename: str, rgb: np.ndarray) -> str:
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
            self.path(filename), format="PNG", optimize=True
        )
        return self.reference(filename)

    def write_preview(
        self,
        filename: str,
        values: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> str:
        normalized = normalize_for_preview(values, valid_mask)
        rgb = _colorize(normalized, valid_mask)
        return self.write_rgb(filename, rgb)

    def write_npy(self, filename: str, values: np.ndarray) -> str:
        np.save(self.path(filename), np.asarray(values, dtype=np.float32), allow_pickle=False)
        return self.reference(filename)

    def write_json(self, filename: str, payload: dict[str, Any]) -> str:
        with self.path(filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return self.reference(filename)

    def write_calibrated_geotiff(
        self,
        filename: str,
        calibrated_height: np.ndarray,
        source: ImageRaster,
        valid_mask: np.ndarray | None = None,
    ) -> str | None:
        """Write float32 heights on the source GeoTIFF grid, when it is valid."""

        if (
            source.source_profile is None
            or source.crs is None
            or source.transform is None
            or calibrated_height.shape != (source.height, source.width)
        ):
            return None
        try:
            import rasterio

            nodata = -9999.0
            profile = source.source_profile.copy()
            # Strip source-band layout options that can be invalid for a one-band
            # float raster while retaining its CRS, transform, dimensions, and
            # other safe GeoTIFF metadata.
            for key in (
                "blockxsize",
                "blockysize",
                "interleave",
                "photometric",
                "nbits",
            ):
                profile.pop(key, None)
            profile.update(
                driver="GTiff",
                width=source.width,
                height=source.height,
                count=1,
                dtype="float32",
                crs=source.crs,
                transform=source.transform,
                nodata=nodata,
                compress="deflate",
                predictor=3,
            )
            values = np.asarray(calibrated_height, dtype=np.float32).copy()
            valid = np.isfinite(values)
            if valid_mask is not None:
                valid &= np.asarray(valid_mask, dtype=bool)
            if source.valid_mask is not None:
                valid &= source.valid_mask
            values[~valid] = nodata
            with rasterio.open(self.path(filename), "w", **profile) as dataset:
                dataset.write(values, 1)
                dataset.write_mask(valid.astype(np.uint8) * 255)
                if source.source_tags:
                    dataset.update_tags(**source.source_tags)
                dataset.update_tags(
                    DEPTHWIZARD_OUTPUT="calibrated_predicted_dsm",
                    DEPTHWIZARD_CALIBRATION="benchmark_only_affine",
                )
            return self.reference(filename)
        except Exception as exc:
            raise ValueError(f"Could not write calibrated GeoTIFF: {exc}") from exc
