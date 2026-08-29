"""Fetch and align vector building footprints from OpenStreetMap for georeferenced scenes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_OSM_CACHE_DIR = Path("data/cache/osm")


def default_osm_cache_dir() -> Path:
    """Where Overpass responses are cached.

    A relative path resolves against the launch directory, which a packaged
    build does not control: Explorer can start the executable anywhere, and an
    install under Program Files is not writable at all. The desktop launcher
    therefore names a per-user location, and the repo-relative path stays the
    default for source checkouts.
    """

    configured = os.getenv("DEPTHWIZARD_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve() / "osm"
    return REPO_OSM_CACHE_DIR


def fetch_osm_footprints(
    source_crs: Any,
    bounds: tuple[float, float, float, float] | dict[str, float] | list[float],
    width: int,
    height: int,
    transform: Any,
    cache_dir: Path | None = None,
    timeout: int = 12,
) -> list[dict[str, Any]]:
    """Query OpenStreetMap Overpass API for building footprint polygons in the scene bounds.

    Returns a list of footprint dictionaries with normalized coordinates [u, v] in [0, 1].
    Cached locally on disk for fast offline reuse.
    """
    if source_crs is None or transform is None:
        return []

    try:
        from rasterio.transform import rowcol
        from rasterio.warp import transform_bounds, transform as transform_coords
    except ImportError:
        return []

    # Extract bounding box in native CRS
    if isinstance(bounds, dict):
        left, bottom, right, top = bounds["left"], bounds["bottom"], bounds["right"], bounds["top"]
    elif isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
        left, bottom, right, top = bounds[0], bounds[1], bounds[2], bounds[3]
    else:
        return []

    # Reproject bounding box to WGS84 EPSG:4326 (lat/lon)
    try:
        min_lon, min_lat, max_lon, max_lat = transform_bounds(
            source_crs, "EPSG:4326", left, bottom, right, top
        )
    except Exception as exc:
        logger.debug("Failed to reproject bounds to EPSG:4326: %s", exc)
        return []

    # Unique cache key for this bounding box
    bbox_key = f"{min_lat:.5f},{min_lon:.5f},{max_lat:.5f},{max_lon:.5f}"
    cache_hash = hashlib.sha256(bbox_key.encode()).hexdigest()[:16]
    if cache_dir is None:
        cache_dir = default_osm_cache_dir()
    # An unwritable cache costs a round trip, not the footprints, so keep going.
    cache_file: Path | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"osm_buildings_{cache_hash}.json"
    except OSError as exc:
        logger.debug("OSM cache directory %s is unusable: %s", cache_dir, exc)

    osm_data = None
    if cache_file is not None and cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                osm_data = json.load(f)
        except Exception:
            osm_data = None

    if osm_data is None:
        query = f"""
        [out:json][timeout:{timeout}];
        (
          way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """
        try:
            import requests

            headers = {"User-Agent": "DepthWizard/1.0 (https://github.com/Harsh1t-S/DepthWizard)"}
            resp = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code == 200:
                osm_data = resp.json()
                if cache_file is not None:
                    try:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(osm_data, f)
                    except OSError as exc:
                        logger.debug("Could not cache the OSM response: %s", exc)
        except Exception as exc:
            logger.debug("Overpass API request failed or timed out: %s", exc)
            return []

    if not osm_data or "elements" not in osm_data:
        return []

    elements = osm_data["elements"]
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}
    ways = [e for e in elements if e["type"] == "way" and "nodes" in e]

    footprints: list[dict[str, Any]] = []

    for way in ways:
        way_nodes = way.get("nodes", [])
        if len(way_nodes) < 4:
            continue
        # Collect lat/lon coordinates
        coords_lon = []
        coords_lat = []
        for nid in way_nodes:
            if nid in nodes:
                lon, lat = nodes[nid]
                coords_lon.append(lon)
                coords_lat.append(lat)
        if len(coords_lon) < 4:
            continue

        # Reproject WGS84 lat/lon to raster native CRS
        try:
            xs, ys = transform_coords("EPSG:4326", source_crs, coords_lon, coords_lat)
        except Exception:
            continue

        # Convert native CRS (x, y) to image pixel (col, row)
        points_norm: list[list[float]] = []
        for x, y in zip(xs, ys):
            r, c = rowcol(transform, x, y)
            u = float(np.clip(c / max(1, width - 1), 0.0, 1.0))
            v = float(np.clip(r / max(1, height - 1), 0.0, 1.0))
            points_norm.append([u, v])

        # Remove duplicate consecutive points
        cleaned: list[list[float]] = []
        for pt in points_norm:
            if not cleaned or abs(cleaned[-1][0] - pt[0]) > 1e-5 or abs(cleaned[-1][1] - pt[1]) > 1e-5:
                cleaned.append(pt)
        if len(cleaned) > 1 and abs(cleaned[0][0] - cleaned[-1][0]) <= 1e-5 and abs(cleaned[0][1] - cleaned[-1][1]) <= 1e-5:
            cleaned.pop()
        if len(cleaned) < 3:
            continue

        tags = way.get("tags", {})
        levels = tags.get("building:levels")
        level_count = float(levels) if levels and levels.replace(".", "", 1).isdigit() else None

        footprints.append({
            "points": cleaned,
            "osm_id": way["id"],
            "building_type": tags.get("building", "yes"),
            "name": tags.get("name"),
            "levels": level_count,
            "source": "osm",
        })

    return footprints
