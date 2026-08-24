"""Image and DSM decoding with optional GeoTIFF metadata preservation."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


TIFF_SUFFIXES = {".tif", ".tiff", ".geotiff"}


def _max_decoded_pixels() -> int:
    raw = os.getenv("DEPTHWIZARD_MAX_DECODED_PIXELS", "50000000")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("DEPTHWIZARD_MAX_DECODED_PIXELS must be an integer") from exc
    return max(value, 1)


def _enforce_decoded_size(width: int, height: int, label: str) -> None:
    pixels = int(width) * int(height)
    maximum = _max_decoded_pixels()
    if width < 1 or height < 1:
        raise ValueError(f"{label} contains no raster pixels")
    if pixels > maximum:
        raise ValueError(
            f"{label} has {pixels:,} decoded pixels; the configured limit is {maximum:,}"
        )


@dataclass
class ImageRaster:
    filename: str
    rgb: np.ndarray
    geospatial: dict[str, Any] | None = None
    valid_mask: np.ndarray | None = None
    crs: Any | None = field(default=None, repr=False)
    transform: Any | None = field(default=None, repr=False)
    source_profile: dict[str, Any] | None = field(default=None, repr=False)
    source_tags: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


@dataclass
class GroundTruthRaster:
    filename: str
    values: np.ndarray
    valid_mask: np.ndarray
    crs: Any | None = field(default=None, repr=False)
    transform: Any | None = field(default=None, repr=False)
    nodata: float | int | None = None

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    @property
    def height(self) -> int:
        return int(self.values.shape[0])


def _looks_like_tiff(data: bytes, filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in TIFF_SUFFIXES or data[:4] in (b"II*\x00", b"MM\x00*")


def _stretch_band(band: np.ndarray, mask: np.ndarray) -> np.ndarray:
    band_float = np.asarray(band, dtype=np.float32)
    valid = mask & np.isfinite(band_float)
    output = np.zeros(band_float.shape, dtype=np.uint8)
    if not valid.any():
        return output

    values = band_float[valid]
    if band.dtype == np.uint8:
        output[valid] = band[valid]
        return output

    low, high = np.percentile(values, (2.0, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        output[valid] = 127
        return output
    scaled = np.clip((band_float - low) / (high - low), 0.0, 1.0)
    output[valid] = np.round(scaled[valid] * 255.0).astype(np.uint8)
    return output


def _dataset_geospatial(dataset: Any) -> dict[str, Any]:
    bounds = dataset.bounds
    transform_values = [float(value) for value in tuple(dataset.transform)[:6]]
    resolution = dataset.res
    crs = dataset.crs
    return {
        "crs": crs.to_string() if crs is not None else None,
        "bounds": {
            "left": float(bounds.left),
            "bottom": float(bounds.bottom),
            "right": float(bounds.right),
            "top": float(bounds.top),
        },
        "resolution": [abs(float(resolution[0])), abs(float(resolution[1]))],
        "pixel_size": [abs(float(resolution[0])), abs(float(resolution[1]))],
        "transform": transform_values,
        "nodata": _json_number(dataset.nodata),
        "valid_for_dsm_export": bool(crs is not None and dataset.transform is not None),
    }


def _json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def read_image_bytes(data: bytes, filename: str) -> ImageRaster:
    """Decode a normal image or GeoTIFF into display-ready RGB."""

    if not data:
        raise ValueError("Uploaded image is empty")

    if _looks_like_tiff(data, filename):
        try:
            from rasterio.io import MemoryFile

            with MemoryFile(data) as memory_file, memory_file.open() as dataset:
                if dataset.width < 1 or dataset.height < 1 or dataset.count < 1:
                    raise ValueError("GeoTIFF contains no raster pixels")
                _enforce_decoded_size(dataset.width, dataset.height, "GeoTIFF")
                dataset_mask = dataset.dataset_mask() > 0
                if dataset.count >= 3:
                    raw = dataset.read((1, 2, 3))
                    rgb = np.stack(
                        [_stretch_band(raw[index], dataset_mask) for index in range(3)],
                        axis=-1,
                    )
                else:
                    raw = dataset.read(1)
                    gray = _stretch_band(raw, dataset_mask)
                    rgb = np.repeat(gray[..., None], 3, axis=2)

                profile = dataset.profile.copy()
                # The export writer deliberately replaces these fields while
                # retaining spatial metadata and safe creation options.
                profile.update(
                    driver="GTiff",
                    width=dataset.width,
                    height=dataset.height,
                    count=1,
                    dtype="float32",
                )
                return ImageRaster(
                    filename=filename,
                    rgb=np.ascontiguousarray(rgb),
                    geospatial=_dataset_geospatial(dataset),
                    valid_mask=np.asarray(dataset_mask, dtype=bool),
                    crs=dataset.crs,
                    transform=dataset.transform,
                    source_profile=profile,
                    source_tags=dataset.tags(),
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not decode GeoTIFF {filename!r}: {exc}") from exc

    try:
        with Image.open(io.BytesIO(data)) as opened:
            _enforce_decoded_size(opened.width, opened.height, "Image")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8).copy()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Could not decode image {filename!r}: {exc}") from exc
    return ImageRaster(filename=filename, rgb=rgb)


def read_ground_truth_bytes(data: bytes, filename: str) -> GroundTruthRaster:
    """Decode a single-band DSM from GeoTIFF, NumPy, or a regular image."""

    if not data:
        raise ValueError("Uploaded ground-truth DSM is empty")
    suffix = Path(filename).suffix.lower()

    if _looks_like_tiff(data, filename):
        try:
            from rasterio.io import MemoryFile

            with MemoryFile(data) as memory_file, memory_file.open() as dataset:
                if dataset.count < 1:
                    raise ValueError("Ground-truth GeoTIFF contains no raster band")
                _enforce_decoded_size(
                    dataset.width, dataset.height, "Ground-truth GeoTIFF"
                )
                values = dataset.read(1).astype(np.float32)
                valid = dataset.read_masks(1) > 0
                nodata = dataset.nodata
                if nodata is not None:
                    if np.isnan(nodata):
                        valid &= ~np.isnan(values)
                    else:
                        valid &= values != np.float32(nodata)
                valid &= np.isfinite(values)
                return GroundTruthRaster(
                    filename=filename,
                    values=values,
                    valid_mask=valid,
                    crs=dataset.crs,
                    transform=dataset.transform,
                    nodata=nodata,
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Could not decode ground-truth GeoTIFF {filename!r}: {exc}"
            ) from exc

    if suffix == ".npy":
        try:
            values = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Could not decode NumPy DSM {filename!r}: {exc}") from exc
        values = np.squeeze(values)
        if values.ndim != 2:
            raise ValueError("A NumPy ground-truth DSM must be a two-dimensional array")
        _enforce_decoded_size(values.shape[1], values.shape[0], "NumPy DSM")
        values = values.astype(np.float32)
        return GroundTruthRaster(
            filename=filename,
            values=values,
            valid_mask=np.isfinite(values),
        )

    try:
        with Image.open(io.BytesIO(data)) as opened:
            _enforce_decoded_size(opened.width, opened.height, "Ground-truth image")
            values = np.asarray(opened.convert("F"), dtype=np.float32).copy()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Could not decode ground-truth DSM {filename!r}: {exc}") from exc
    return GroundTruthRaster(
        filename=filename,
        values=values,
        valid_mask=np.isfinite(values),
    )


def _transforms_match(first: Any, second: Any) -> bool:
    if first is None or second is None:
        return False
    return bool(
        np.allclose(
            np.asarray(tuple(first)[:6], dtype=np.float64),
            np.asarray(tuple(second)[:6], dtype=np.float64),
            rtol=1e-8,
            atol=1e-8,
        )
    )


def align_ground_truth(
    ground_truth: GroundTruthRaster,
    source: ImageRaster,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align a DSM to the source pixel grid and retain a strict validity mask."""

    target_shape = (source.height, source.width)
    notices: list[str] = []
    same_shape = ground_truth.values.shape == target_shape

    can_georeference = (
        ground_truth.crs is not None
        and ground_truth.transform is not None
        and source.crs is not None
        and source.transform is not None
    )
    same_crs = can_georeference and ground_truth.crs == source.crs
    same_transform = _transforms_match(ground_truth.transform, source.transform)

    if same_shape and same_crs and same_transform:
        return (
            ground_truth.values.astype(np.float32, copy=False),
            ground_truth.valid_mask.astype(bool, copy=False),
            notices,
        )

    if can_georeference and not (same_shape and same_crs and same_transform):
        try:
            from rasterio.warp import Resampling, reproject

            source_values = ground_truth.values.astype(np.float32, copy=True)
            source_values[~ground_truth.valid_mask] = np.nan
            aligned = np.full(target_shape, np.nan, dtype=np.float32)
            reproject(
                source=source_values,
                destination=aligned,
                src_transform=ground_truth.transform,
                src_crs=ground_truth.crs,
                src_nodata=np.nan,
                dst_transform=source.transform,
                dst_crs=source.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            aligned_mask = np.zeros(target_shape, dtype=np.uint8)
            reproject(
                source=ground_truth.valid_mask.astype(np.uint8),
                destination=aligned_mask,
                src_transform=ground_truth.transform,
                src_crs=ground_truth.crs,
                src_nodata=0,
                dst_transform=source.transform,
                dst_crs=source.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
            valid = (aligned_mask > 0) & np.isfinite(aligned)
            notices.append(
                "Ground-truth DSM was geospatially reprojected/resampled to the "
                "source image grid because its CRS, transform, or dimensions differed."
            )
            return aligned, valid, notices
        except Exception as exc:
            raise ValueError(f"Could not align the georeferenced DSM: {exc}") from exc

    if same_shape:
        if ground_truth.crs is not None or source.crs is not None:
            notices.append(
                "Ground truth and source have matching dimensions but incomplete "
                "georeferencing; pixel-for-pixel alignment was assumed."
            )
        return (
            ground_truth.values.astype(np.float32, copy=False),
            ground_truth.valid_mask.astype(bool, copy=False),
            notices,
        )

    # Numeric resizing is a last resort when one or both rasters lack a usable
    # CRS/transform.  The accompanying notice is returned in the API response.
    values_for_resize = ground_truth.values.astype(np.float32, copy=True)
    valid_values = values_for_resize[ground_truth.valid_mask]
    fill = float(np.median(valid_values)) if valid_values.size else 0.0
    values_for_resize[~ground_truth.valid_mask] = fill
    aligned = np.asarray(
        Image.fromarray(values_for_resize, mode="F").resize(
            (source.width, source.height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    valid = np.asarray(
        Image.fromarray(ground_truth.valid_mask.astype(np.uint8) * 255).resize(
            (source.width, source.height), Image.Resampling.NEAREST
        )
    ) > 0
    valid &= np.isfinite(aligned)
    notices.append(
        f"Ground-truth DSM was resized from {ground_truth.width}x{ground_truth.height} "
        f"to {source.width}x{source.height} because usable geospatial alignment "
        "metadata was unavailable. Bilinear values and a nearest-neighbor validity "
        "mask were used."
    )
    return aligned, valid, notices
