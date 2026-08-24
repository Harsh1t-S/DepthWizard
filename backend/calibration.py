"""Deployment-reference parsing and sampling for metric depth calibration."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import CalibrationResult, fit_affine_calibration


@dataclass(frozen=True)
class GroundControlPoint:
    """A surveyed elevation at a zero-based image pixel coordinate."""

    x: float
    y: float
    elevation: float


_X_ALIASES = ("x", "pixel_x", "col")
_Y_ALIASES = ("y", "pixel_y", "row")
_ELEVATION_ALIASES = ("elevation", "z", "height")
_PIXEL_COORDINATE_SPACES = {"pixel", "pixels", "image_pixel", "image_pixels"}


def _normalized_mapping(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key).strip().lower().replace("-", "_"): value
        for key, value in record.items()
    }


def _field(record: Mapping[str, Any], aliases: Sequence[str], label: str, index: int) -> Any:
    for alias in aliases:
        if alias in record and record[alias] is not None and str(record[alias]).strip() != "":
            return record[alias]
    accepted = ", ".join(aliases)
    raise ValueError(
        f"GCP point {index} is missing {label}; accepted field names are {accepted}"
    )


def _finite_float(value: Any, label: str, index: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"GCP point {index} has a non-numeric {label}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GCP point {index} has a non-numeric {label}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"GCP point {index} has a non-finite {label}")
    return numeric


def _point_from_mapping(record: Mapping[str, Any], index: int) -> GroundControlPoint:
    normalized = _normalized_mapping(record)
    declared_space = normalized.get("coordinate_space")
    if declared_space is not None and str(declared_space).strip():
        candidate = str(declared_space).strip().lower().replace("-", "_")
        if candidate not in _PIXEL_COORDINATE_SPACES:
            raise ValueError(
                f"GCP point {index} declares unsupported coordinate_space "
                f"{declared_space!r}; only zero-based image pixels are accepted"
            )
    return GroundControlPoint(
        x=_finite_float(_field(normalized, _X_ALIASES, "x coordinate", index), "x", index),
        y=_finite_float(_field(normalized, _Y_ALIASES, "y coordinate", index), "y", index),
        elevation=_finite_float(
            _field(normalized, _ELEVATION_ALIASES, "elevation", index),
            "elevation",
            index,
        ),
    )


def parse_gcps_bytes(data: bytes, filename: str) -> list[GroundControlPoint]:
    """Parse GCPs from CSV or JSON using documented coordinate aliases.

    JSON may be a list of point objects or an object containing ``points``.
    CSV headers are matched case-insensitively.  Coordinates are zero-based
    pixel coordinates: ``x``/``col`` increases right and ``y``/``row`` down.
    """

    if not data:
        raise ValueError("Uploaded GCP file is empty")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("GCP CSV/JSON must be UTF-8 encoded") from exc
    stripped = text.lstrip()
    suffix = Path(filename or "gcps").suffix.lower()
    is_json = suffix == ".json" or stripped.startswith("[") or stripped.startswith("{")

    records: Any
    if is_json:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse GCP JSON: {exc.msg}") from exc
        if isinstance(document, dict):
            declared_space = document.get("coordinate_space")
            if declared_space is not None:
                candidate = str(declared_space).strip().lower().replace("-", "_")
                if candidate not in _PIXEL_COORDINATE_SPACES:
                    raise ValueError(
                        "GCP JSON coordinate_space must describe zero-based image pixels"
                    )
            records = document.get("points")
            if records is None:
                raise ValueError("GCP JSON object must contain a 'points' list")
        else:
            records = document
        if not isinstance(records, list):
            raise ValueError("GCP JSON must be a point list or an object containing 'points'")
    else:
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if reader.fieldnames is None:
                raise ValueError("GCP CSV must include a header row")
            records = [
                row
                for row in reader
                if any(value is not None and str(value).strip() for value in row.values())
            ]
        except csv.Error as exc:
            raise ValueError(f"Could not parse GCP CSV: {exc}") from exc

    points: list[GroundControlPoint] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"GCP point {index} must be an object or CSV row")
        points.append(_point_from_mapping(record, index))
    if len(points) < 3:
        raise ValueError("At least three GCP points are required for calibration")
    locations = {(point.x, point.y) for point in points}
    if len(locations) < 3:
        raise ValueError("At least three distinct GCP pixel locations are required")
    if len(locations) != len(points):
        raise ValueError("Duplicate GCP pixel locations are not allowed")
    return points


def _bilinear_depth_sample(
    depth: np.ndarray,
    valid: np.ndarray,
    x: float,
    y: float,
    index: int,
) -> float:
    height, width = depth.shape
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = x - x0
    ty = y - y0
    weighted_pixels: dict[tuple[int, int], float] = {}
    for row, col, weight in (
        (y0, x0, (1.0 - tx) * (1.0 - ty)),
        (y0, x1, tx * (1.0 - ty)),
        (y1, x0, (1.0 - tx) * ty),
        (y1, x1, tx * ty),
    ):
        if weight > 0.0:
            weighted_pixels[(row, col)] = weighted_pixels.get((row, col), 0.0) + weight
    value = 0.0
    for (row, col), weight in weighted_pixels.items():
        if not valid[row, col] or not np.isfinite(depth[row, col]):
            raise ValueError(
                f"GCP point {index} samples an invalid or non-finite model-depth pixel"
            )
        value += weight * float(depth[row, col])
    return value


def sample_depth_at_gcps(
    depth: np.ndarray,
    points: Sequence[GroundControlPoint],
    *,
    method: str = "bilinear",
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample model depth at validated, zero-based pixel-coordinate GCPs."""

    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim != 2:
        raise ValueError("Relative depth must be a two-dimensional array")
    normalized_method = method.strip().lower()
    if normalized_method not in {"nearest", "bilinear"}:
        raise ValueError("GCP sampling must be 'nearest' or 'bilinear'")
    valid = np.isfinite(depth_array)
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=bool)
        if supplied_mask.shape != depth_array.shape:
            raise ValueError("Source validity mask must share the depth pixel grid")
        valid &= supplied_mask

    height, width = depth_array.shape
    sampled: list[float] = []
    elevations: list[float] = []
    for index, point in enumerate(points, start=1):
        if not (0.0 <= point.x <= width - 1 and 0.0 <= point.y <= height - 1):
            raise ValueError(
                f"GCP point {index} ({point.x}, {point.y}) is outside the zero-based "
                f"source pixel bounds x=0..{width - 1}, y=0..{height - 1}"
            )
        if normalized_method == "nearest":
            col = min(int(math.floor(point.x + 0.5)), width - 1)
            row = min(int(math.floor(point.y + 0.5)), height - 1)
            if not valid[row, col]:
                raise ValueError(
                    f"GCP point {index} samples an invalid or non-finite model-depth pixel"
                )
            sampled_value = float(depth_array[row, col])
        else:
            sampled_value = _bilinear_depth_sample(
                depth_array, valid, point.x, point.y, index
            )
        if not math.isfinite(sampled_value):
            raise ValueError(f"GCP point {index} produced a non-finite model-depth sample")
        sampled.append(sampled_value)
        elevations.append(point.elevation)
    return np.asarray(sampled, dtype=np.float64), np.asarray(elevations, dtype=np.float64)


def calibrate_depth_from_gcps(
    depth: np.ndarray,
    points: Sequence[GroundControlPoint],
    *,
    method: str,
    source_valid_mask: np.ndarray | None,
    label: str,
) -> CalibrationResult:
    sampled_depth, elevations = sample_depth_at_gcps(
        depth, points, method=method, valid_mask=source_valid_mask
    )
    result = fit_affine_calibration(
        depth,
        sampled_depth,
        elevations,
        source="gcps",
        label=label,
        benchmark_only=False,
        output_valid_mask=source_valid_mask,
    )
    result.calibration["sampling"] = method.strip().lower()
    result.calibration["coordinate_system"] = "zero_based_image_pixels"
    return result


def calibrate_depth_from_reference_raster(
    depth: np.ndarray,
    reference_elevation: np.ndarray,
    reference_valid_mask: np.ndarray,
    *,
    source_valid_mask: np.ndarray | None,
    label: str,
) -> CalibrationResult:
    """Fit against a DEM already aligned to the source/depth pixel grid."""

    depth_array = np.asarray(depth, dtype=np.float32)
    elevation = np.asarray(reference_elevation, dtype=np.float32)
    valid = np.asarray(reference_valid_mask, dtype=bool)
    if depth_array.shape != elevation.shape or depth_array.shape != valid.shape:
        raise ValueError("Depth and aligned reference DEM must share a pixel grid")
    valid &= np.isfinite(depth_array) & np.isfinite(elevation)
    if source_valid_mask is not None:
        source_mask = np.asarray(source_valid_mask, dtype=bool)
        if source_mask.shape != depth_array.shape:
            raise ValueError("Source validity mask must share the depth pixel grid")
        valid &= source_mask
    return fit_affine_calibration(
        depth_array,
        depth_array[valid],
        elevation[valid],
        source="reference_dem",
        label=label,
        benchmark_only=False,
        output_valid_mask=source_valid_mask,
    )
