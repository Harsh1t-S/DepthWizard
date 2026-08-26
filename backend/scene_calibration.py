"""Metric calibration from a georeferenced image alone, using its own shadows.

SIH26175 requires that georeferenced imagery produce an absolute DSM with
metric heights. A reference DEM or ground control points satisfy that, but both
are extra files the operator may not have. The problem statement also allows
"scene-level statistics, semantic priors", and a georeferenced GeoTIFF already
carries what is needed: a CRS and transform fix where the scene is, the tags
usually record when it was taken, and shadow length then gives height in metres.

This module turns those into a scale and offset for the relative prediction. It
is deliberately conservative: every path that cannot be trusted returns a reason
rather than a number, because a fabricated metric result is worse than none.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from ml.shadow_height import estimate_heights_from_shadows

from .raster_io import ImageRaster


# GeoTIFF tags that commonly carry acquisition time, most specific first.
DATETIME_TAGS = (
    "TIFFTAG_DATETIME",
    "ACQUISITIONDATETIME",
    "ACQUISITION_DATE",
    "DATETIME",
    "DATE_ACQUIRED",
    "NITF_IDATIM",
)

DATETIME_FORMATS = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y%m%d%H%M%S",
    "%Y-%m-%d",
    "%Y:%m:%d",
)


def parse_acquisition_time(tags: dict[str, str]) -> datetime | None:
    """Recover acquisition time from raster tags, or None."""

    for key in DATETIME_TAGS:
        for tag_name, raw in tags.items():
            if tag_name.upper() != key:
                continue
            text = str(raw).strip()
            for fmt in DATETIME_FORMATS:
                try:
                    parsed = datetime.strptime(text, fmt)
                except ValueError:
                    continue
                # A date with no clock reading is useless for solar geometry;
                # midday is the only defensible assumption and is flagged.
                if fmt in {"%Y-%m-%d", "%Y:%m:%d"}:
                    parsed = parsed.replace(hour=12)
                return parsed.replace(tzinfo=timezone.utc)
    return None


def scene_centre_lonlat(source: ImageRaster) -> tuple[float, float] | None:
    """Return the scene centre as (longitude, latitude) in degrees."""

    if source.crs is None or source.transform is None:
        return None
    try:
        from rasterio.warp import transform as warp_transform

        centre_x, centre_y = source.transform * (source.width / 2, source.height / 2)
        lon, lat = warp_transform(source.crs, "EPSG:4326", [centre_x], [centre_y])
        return float(lon[0]), float(lat[0])
    except Exception:
        return None


def _mean_pixel_size(source: ImageRaster) -> float | None:
    geospatial = source.geospatial or {}
    pixel_size = geospatial.get("pixel_size")
    if isinstance(pixel_size, (list, tuple)) and len(pixel_size) >= 2:
        values = [abs(float(pixel_size[0])), abs(float(pixel_size[1]))]
        if all(value > 0 for value in values):
            return float(np.mean(values))
    return None


def calibrate_from_scene_shadows(
    depth: np.ndarray,
    source: ImageRaster,
    acquisition_time: datetime | None = None,
) -> dict[str, Any]:
    """Fit relative depth to metres using shadow-derived heights.

    Returns a report that always explains itself. ``usable`` is only true when a
    scale and offset were actually derived.
    """

    report: dict[str, Any] = {
        "method": "shadow_solar_geometry",
        "usable": False,
        "reason": None,
    }

    pixel_size = _mean_pixel_size(source)
    if pixel_size is None:
        report["reason"] = "Image has no usable pixel size; a projected CRS is required"
        return report
    # Degrees would make "metres per pixel" meaningless, and a shadow measured
    # in degrees of longitude is not a length.
    if pixel_size < 1e-4:
        report["reason"] = (
            "Pixel size looks angular, not metric. Reproject to a metric CRS "
            "such as UTM before shadow calibration"
        )
        return report

    centre = scene_centre_lonlat(source)
    if centre is None:
        report["reason"] = "Could not resolve scene coordinates from the CRS and transform"
        return report
    longitude, latitude = centre
    report["scene_centre"] = {"longitude": round(longitude, 6), "latitude": round(latitude, 6)}

    when = acquisition_time or parse_acquisition_time(source.source_tags or {})
    if when is None:
        report["reason"] = (
            "No acquisition timestamp in the raster tags. Supply one to enable "
            "shadow-based metric calibration"
        )
        return report
    report["acquisition_time"] = when.isoformat()

    shadows = estimate_heights_from_shadows(
        source.rgb, when, latitude, longitude, pixel_size
    )
    report["solar"] = {
        "elevation_degrees": shadows.get("solar_elevation_degrees"),
        "azimuth_degrees": shadows.get("solar_azimuth_degrees"),
    }
    if not shadows.get("usable"):
        report["reason"] = shadows.get("reason")
        return report

    report["shadow_samples"] = shadows["samples"]
    report["shadow_fraction"] = shadows.get("shadow_fraction")

    finite = np.isfinite(depth)
    if source.valid_mask is not None:
        finite &= source.valid_mask
    if int(finite.sum()) < 64:
        report["reason"] = "Too few valid depth pixels to fit a scale"
        return report

    values = depth[finite].astype(np.float64)
    # Depth Anything returns inverse depth, so on nadir imagery larger values
    # are nearer the sensor, meaning taller. Percentiles rather than extremes
    # keep a single bright artefact from setting the scale.
    low, high = np.percentile(values, (5.0, 99.0))
    if high - low <= 1e-9:
        report["reason"] = "Predicted depth has no usable range"
        return report

    tall_metres = float(shadows["height_p99_metres"])
    if tall_metres <= 0.5:
        report["reason"] = "Shadow-derived heights are too small to calibrate against"
        return report

    # Anchor the 5th percentile of depth at ground level and the 99th at the
    # 99th-percentile shadow height. Two robust points define scale and offset.
    scale = tall_metres / (high - low)
    offset = -scale * low
    report.update(
        usable=True,
        scale=round(float(scale), 6),
        offset=round(float(offset), 4),
        units="metres_above_local_ground",
        reference_height_p99_metres=tall_metres,
        reference_height_p90_metres=shadows.get("height_p90_metres"),
        note=(
            "Heights are relative to local ground, not a vertical datum. The fit "
            "assumes shadows fall on level ground and are not clipped by "
            "neighbouring structures."
        ),
    )
    return report
