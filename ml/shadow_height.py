"""Absolute building height from shadow length and solar geometry.

SIH26175 asks that georeferenced imagery yield an *absolute* DSM with metric
heights. Monocular depth alone cannot supply that: a single RGB image fixes
neither scale nor vertical datum. The usual remedy is an external DEM or ground
control points, but the problem statement also permits "scene-level statistics,
semantic priors", and a georeferenced image already carries the two things
needed to recover real metres without any extra file.

Given the acquisition time and the scene coordinates, the sun's elevation and
azimuth are computable. A vertical structure of height ``h`` then casts a
shadow of length ``L`` on level ground:

    h = L * tan(solar_elevation)

Measuring shadow lengths across a scene yields height samples in metres, which
calibrate the relative prediction the same way a set of GCPs would.

Assumptions, which the caller must be willing to state: shadows fall on ground
that is level relative to the structure, they are not clipped by neighbouring
buildings, and the acquisition timestamp and CRS are trustworthy. Estimates are
returned with the samples so a caller can reject a weak fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


# Below roughly 10 degrees, shadows lengthen fast and small errors in elevation
# translate into large height errors; above roughly 80 they are too short to
# measure. Both extremes are refused rather than silently reported.
MIN_SOLAR_ELEVATION_DEGREES = 10.0
MAX_SOLAR_ELEVATION_DEGREES = 80.0


@dataclass(frozen=True)
class SolarPosition:
    """Sun position for one instant and place, in degrees."""

    elevation: float
    azimuth: float

    @property
    def usable(self) -> bool:
        return MIN_SOLAR_ELEVATION_DEGREES <= self.elevation <= MAX_SOLAR_ELEVATION_DEGREES


def solar_position(when: datetime, latitude: float, longitude: float) -> SolarPosition:
    """Compute solar elevation and azimuth (NOAA low-precision algorithm).

    Accurate to a few arcminutes over a century either side of 2000, far
    tighter than the error in measuring a shadow edge from imagery.
    """

    moment = when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)

    # Julian day, then centuries from the J2000.0 epoch.
    year, month = moment.year, moment.month
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    day_fraction = (
        moment.hour + moment.minute / 60 + moment.second / 3600
    ) / 24.0
    julian_day = (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + moment.day
        + day_fraction
        + b
        - 1524.5
    )
    t = (julian_day - 2451545.0) / 36525.0

    mean_longitude = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    mean_anomaly = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    anomaly_radians = math.radians(mean_anomaly)
    centre = (
        math.sin(anomaly_radians) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * anomaly_radians) * (0.019993 - 0.000101 * t)
        + math.sin(3 * anomaly_radians) * 0.000289
    )
    true_longitude = mean_longitude + centre
    omega = 125.04 - 1934.136 * t
    apparent_longitude = true_longitude - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    obliquity = (
        23.0
        + (26.0 + ((21.448 - t * (46.815 + t * (0.00059 - t * 0.001813)))) / 60.0) / 60.0
    )
    obliquity_corrected = obliquity + 0.00256 * math.cos(math.radians(omega))

    declination = math.degrees(
        math.asin(
            math.sin(math.radians(obliquity_corrected))
            * math.sin(math.radians(apparent_longitude))
        )
    )

    y = math.tan(math.radians(obliquity_corrected / 2)) ** 2
    equation_of_time = 4 * math.degrees(
        y * math.sin(2 * math.radians(mean_longitude))
        - 2 * eccentricity * math.sin(anomaly_radians)
        + 4 * eccentricity * y * math.sin(anomaly_radians) * math.cos(2 * math.radians(mean_longitude))
        - 0.5 * y * y * math.sin(4 * math.radians(mean_longitude))
        - 1.25 * eccentricity * eccentricity * math.sin(2 * anomaly_radians)
    )

    minutes = moment.hour * 60 + moment.minute + moment.second / 60.0
    true_solar_time = (minutes + equation_of_time + 4 * longitude) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0

    latitude_radians = math.radians(latitude)
    declination_radians = math.radians(declination)
    hour_radians = math.radians(hour_angle)

    cos_zenith = math.sin(latitude_radians) * math.sin(declination_radians) + math.cos(
        latitude_radians
    ) * math.cos(declination_radians) * math.cos(hour_radians)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    elevation = 90.0 - math.degrees(math.acos(cos_zenith))

    azimuth_denominator = math.cos(latitude_radians) * math.sin(math.radians(90.0 - elevation))
    if abs(azimuth_denominator) < 1e-9:
        azimuth = 180.0
    else:
        cos_azimuth = (
            math.sin(latitude_radians) * math.cos(math.radians(90.0 - elevation))
            - math.sin(declination_radians)
        ) / azimuth_denominator
        cos_azimuth = max(-1.0, min(1.0, cos_azimuth))
        # NOAA's convention: the arccosine gives an angle measured from south,
        # so it is rotated to a compass bearing clockwise from north. Without
        # the rotation the result is 180 degrees out and every shadow is traced
        # toward the sun instead of away from it.
        azimuth = math.degrees(math.acos(cos_azimuth))
        azimuth = (azimuth + 180.0) % 360.0 if hour_angle > 0 else (540.0 - azimuth) % 360.0

    return SolarPosition(elevation=elevation, azimuth=azimuth)


def detect_shadow_mask(rgb: np.ndarray, percentile: float = 12.0) -> np.ndarray:
    """Mark cast shadows: dark, and blue-shifted by skylight.

    Shadowed ground is lit only by the sky, so it is both darker and relatively
    bluer than sunlit ground. Combining the two separates shadow from dark roof
    material, which brightness alone cannot do.
    """

    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Shadow detection needs an RGB image")
    scale = 255.0 if image.max() > 1.5 else 1.0
    red, green, blue = (image[..., i] / scale for i in range(3))

    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    # Ratio rather than difference so the test is invariant to exposure.
    blueness = (blue + 1e-6) / (red + 1e-6)

    dark = luminance <= np.percentile(luminance, percentile)
    bluish = blueness >= np.median(blueness)
    return dark & bluish


def _run_lengths_along(mask: np.ndarray, azimuth_degrees: float) -> list[float]:
    """Measure shadow runs along the sun's ground-projection direction."""

    # Shadows extend away from the sun, so travel along the anti-solar bearing.
    bearing = math.radians((azimuth_degrees + 180.0) % 360.0)
    step_x = math.sin(bearing)
    step_y = -math.cos(bearing)

    height, width = mask.shape
    lengths: list[float] = []
    # Sampling every fourth row and column keeps this linear in image size while
    # still collecting thousands of runs on a typical tile.
    for start_y in range(0, height, 4):
        for start_x in range(0, width, 4):
            if not mask[start_y, start_x]:
                continue
            # Only start a run at a shadow boundary, or every pixel of one
            # shadow contributes a run.
            previous_x = int(round(start_x - step_x))
            previous_y = int(round(start_y - step_y))
            if (
                0 <= previous_x < width
                and 0 <= previous_y < height
                and mask[previous_y, previous_x]
            ):
                continue
            length = 0.0
            x, y = float(start_x), float(start_y)
            while True:
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < width and 0 <= yi < height) or not mask[yi, xi]:
                    break
                length += 1.0
                x += step_x
                y += step_y
                if length > max(height, width):
                    break
            if length >= 3.0:
                lengths.append(length)
    return lengths


def estimate_heights_from_shadows(
    rgb: np.ndarray,
    when: datetime,
    latitude: float,
    longitude: float,
    pixel_size_metres: float,
) -> dict[str, object]:
    """Estimate a metric height distribution for the scene from its shadows."""

    if pixel_size_metres <= 0:
        raise ValueError("pixel_size_metres must be positive")

    sun = solar_position(when, latitude, longitude)
    result: dict[str, object] = {
        "solar_elevation_degrees": round(sun.elevation, 3),
        "solar_azimuth_degrees": round(sun.azimuth, 3),
        "pixel_size_metres": pixel_size_metres,
        "usable": False,
        "reason": None,
        "samples": 0,
    }
    if not sun.usable:
        result["reason"] = (
            f"Solar elevation {sun.elevation:.1f} degrees is outside the "
            f"{MIN_SOLAR_ELEVATION_DEGREES:.0f}-{MAX_SOLAR_ELEVATION_DEGREES:.0f} "
            "degree band where shadow length is a reliable height cue"
        )
        return result

    mask = detect_shadow_mask(rgb)
    shadow_fraction = float(mask.mean())
    result["shadow_fraction"] = round(shadow_fraction, 4)
    if shadow_fraction < 0.005:
        result["reason"] = "Too little shadow detected to estimate heights"
        return result

    lengths = _run_lengths_along(mask, sun.azimuth)
    if len(lengths) < 20:
        result["reason"] = f"Only {len(lengths)} shadow runs found; need at least 20"
        return result

    tangent = math.tan(math.radians(sun.elevation))
    heights = np.asarray(lengths, dtype=np.float64) * pixel_size_metres * tangent
    # The tail is where real buildings live; the bulk is noise and clutter.
    result.update(
        usable=True,
        samples=len(lengths),
        height_p50_metres=round(float(np.percentile(heights, 50)), 2),
        height_p90_metres=round(float(np.percentile(heights, 90)), 2),
        height_p99_metres=round(float(np.percentile(heights, 99)), 2),
        height_max_metres=round(float(heights.max()), 2),
    )
    return result
