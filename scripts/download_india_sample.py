"""Download high-resolution georeferenced satellite imagery for Mumbai and other Indian cities."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import requests
import rasterio
from rasterio.transform import from_bounds

# High-profile Indian urban regions
INDIAN_SCENES = {
    "mumbai-bkc": {
        "name": "Mumbai Bandra-Kurla Complex (BKC)",
        "description": "Financial district with modern high-rises, wide avenues, and commercial towers.",
        "center_lat": 19.0658,
        "center_lon": 72.8685,
        "size": 1024,
        "span": 650.0,
    },
    "mumbai-nariman": {
        "name": "Mumbai Nariman Point & Marine Drive",
        "description": "Iconic coastal skyline with dense high-rise commercial towers along the Arabian Sea.",
        "center_lat": 18.9265,
        "center_lon": 72.8235,
        "size": 1024,
        "span": 650.0,
    },
    "mumbai-cst": {
        "name": "Mumbai Fort & CST Heritage District",
        "description": "Historic urban district with dense heritage architecture, railway terminus, and Victorian buildings.",
        "center_lat": 18.9400,
        "center_lon": 72.8350,
        "size": 1024,
        "span": 650.0,
    },
    "delhi-cp": {
        "name": "New Delhi Connaught Place",
        "description": "Famous concentric circular urban planning with colonnaded heritage blocks and radiating radial roads.",
        "center_lat": 28.6315,
        "center_lon": 77.2167,
        "size": 1024,
        "span": 650.0,
    },
    "bengaluru-techpark": {
        "name": "Bengaluru Manyata Tech Park",
        "description": "Major IT corridor with massive glass campus blocks and multi-level parking structures.",
        "center_lat": 13.0485,
        "center_lon": 77.6210,
        "size": 1024,
        "span": 650.0,
    },
}


def lat_lon_to_meters(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 lat/lon to Web Mercator EPSG:3857 (meters)."""
    origin_shift = 2.0 * math.pi * 6378137.0 / 2.0
    mx = lon * origin_shift / 180.0
    my = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * origin_shift / 180.0
    return mx, my


def download_arcgis_imagery(
    center_lat: float,
    center_lon: float,
    out_path: Path,
    width_px: int = 1024,
    height_px: int = 1024,
    meters_span: float = 650.0,
) -> bool:
    """Export high-resolution Maxar/WorldView satellite orthophoto from ArcGIS World Imagery."""
    mx, my = lat_lon_to_meters(center_lat, center_lon)
    half = meters_span / 2.0
    xmin, ymin, xmax, ymax = mx - half, my - half, mx + half, my + half

    url = (
        "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
    )
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{width_px},{height_px}",
        "format": "tiff",
        "f": "image",
    }
    headers = {"User-Agent": "DepthWizard-SIH26175/1.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=25)
        if resp.status_code != 200 or len(resp.content) < 1000:
            print(f"ArcGIS export returned status {resp.status_code}")
            return False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Parse image and write as standard GeoTIFF
        with Image.open(io.BytesIO(resp.content)) as img:
            rgb = np.asarray(img.convert("RGB"))

        transform = from_bounds(xmin, ymin, xmax, ymax, width_px, height_px)
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height_px,
            width=width_px,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform,
        ) as dst:
            for b in range(3):
                dst.write(rgb[:, :, b], b + 1)
        print(f"Saved: {out_path} ({width_px}x{height_px}, EPSG:3857, ~{meters_span/width_px:.2f}m/px GSD)")
        return True
    except Exception as exc:
        print(f"Failed to download imagery for ({center_lat}, {center_lon}): {exc}")
        return False


def main() -> None:
    output_base = Path("data/sample")
    for key, info in INDIAN_SCENES.items():
        scene_dir = output_base / key
        scene_dir.mkdir(parents=True, exist_ok=True)
        tif_path = scene_dir / "rgb_india.tif"
        print(f"\n--- Downloading {info['name']} ---")
        success = download_arcgis_imagery(
            center_lat=info["center_lat"],
            center_lon=info["center_lon"],
            out_path=tif_path,
            width_px=info["size"],
            height_px=info["size"],
            meters_span=info.get("span", 650.0),
        )
        if success:
            source_info = {
                "name": info["name"],
                "description": info["description"],
                "center": {"lat": info["center_lat"], "lon": info["center_lon"]},
                "crs": "EPSG:3857",
                "resolution_meters_per_pixel": round(info.get("span", 650.0) / info["size"], 3),
                "source": "High-Resolution Maxar/WorldView Satellite Imagery (ArcGIS World Imagery)",
            }
            with open(scene_dir / "SOURCE.json", "w", encoding="utf-8") as f:
                json.dump(source_info, f, indent=2)


if __name__ == "__main__":
    main()
