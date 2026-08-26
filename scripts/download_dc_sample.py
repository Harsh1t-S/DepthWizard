#!/usr/bin/env python
"""Download a small, real, public urban RGB/DSM pair from DC GIS services."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import rasterio


ORTHO_SERVICE = (
    "https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/"
    "Ortho_2021/ImageServer"
)
DSM_SERVICE = (
    "https://imagery.dcgis.dc.gov/dcgis/rest/services/Lidar/"
    "DSM_2020/ImageServer"
)
DEFAULT_BBOX = (397800.0, 135000.0, 398824.0, 136024.0)
SOURCE_CRS = 26985

# SIH26175 is scored partly on "performance stability across urban, sparse,
# hilly, and forested landscapes", so a single downtown tile is not enough to
# support an accuracy claim. These are 1024 m squares in Maryland State Plane
# (EPSG:26985) chosen to cover distinct landscape types, each a few megabytes.
SCENES: dict[str, tuple[tuple[float, float, float, float], str]] = {
    "downtown": ((397800.0, 135000.0, 398824.0, 136024.0), "Dense high-rise core"),
    "mall": ((396200.0, 134400.0, 397224.0, 135424.0), "Open ground and low monuments"),
    "rock-creek": ((395400.0, 139000.0, 396424.0, 140024.0), "Forested valley with relief"),
    "residential": ((399600.0, 140200.0, 400624.0, 141224.0), "Low-rise housing and tree cover"),
    "waterfront": ((399000.0, 132600.0, 400024.0, 133624.0), "River, bridges, flat terrain"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a compact Washington, DC urban orthophoto and LiDAR DSM "
            "onto the same requested grid. This is a smoke-test pair, not an "
            "official SIH/ISRO evaluation dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sample/dc-urban"),
        help="Destination directory (default: data/sample/dc-urban)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Square output pixels; 256-2048 (default: 1024)",
    )
    parser.add_argument(
        "--scene",
        default="downtown",
        choices=tuple(SCENES),
        help="Named landscape type to download (default: downtown)",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help=f"Explicit bounding box in EPSG:{SOURCE_CRS}, overriding --scene",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing downloaded sample",
    )
    return parser.parse_args()


def export_url(
    service: str,
    size: int,
    pixel_type: str,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
) -> tuple[str, str]:
    params = {
        "bbox": ",".join(str(value) for value in bbox),
        "bboxSR": SOURCE_CRS,
        "imageSR": SOURCE_CRS,
        "size": f"{size},{size}",
        "format": "tiff",
        "pixelType": pixel_type,
        "f": "json",
    }
    request_url = f"{service}/exportImage?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        request_url,
        headers={"User-Agent": "DepthWizard-SIH26175/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if "error" in payload or not payload.get("href"):
        raise RuntimeError(f"DC GIS export failed: {payload.get('error', payload)}")
    return request_url, str(payload["href"])


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DepthWizard-SIH26175/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def raster_summary(path: Path) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        return {
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "crs": str(dataset.crs),
            "bounds": [float(value) for value in dataset.bounds],
            "transform": [float(value) for value in dataset.transform[:6]],
            "dtypes": list(dataset.dtypes),
        }


def main() -> int:
    args = parse_args()
    if not 256 <= args.size <= 2048:
        raise SystemExit("--size must be between 256 and 2048")

    if args.bbox:
        bbox = tuple(args.bbox)
        scene_label = "custom"
    else:
        bbox, description = SCENES[args.scene]
        scene_label = args.scene
        print(f"Scene '{scene_label}': {description}")

    # Keep each landscape in its own directory unless one was named explicitly,
    # so repeated runs build an evaluation set instead of overwriting one tile.
    if args.output_dir == Path("data/sample/dc-urban") and scene_label != "custom":
        args.output_dir = Path("data/sample") / f"dc-{scene_label}"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / "rgb_2021.tif"
    dsm_path = output_dir / "dsm_2020.tif"
    metadata_path = output_dir / "SOURCE.json"

    existing = [path for path in (rgb_path, dsm_path, metadata_path) if path.exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"Refusing to overwrite {names}; pass --force to replace them")

    print("Requesting aligned DC GIS exports...")
    rgb_request, rgb_download = export_url(ORTHO_SERVICE, args.size, "U8", bbox)
    dsm_request, dsm_download = export_url(DSM_SERVICE, args.size, "F32", bbox)
    print(f"Downloading RGB -> {rgb_path}")
    download(rgb_download, rgb_path)
    print(f"Downloading DSM -> {dsm_path}")
    download(dsm_download, dsm_path)

    rgb = raster_summary(rgb_path)
    dsm = raster_summary(dsm_path)
    if (rgb["width"], rgb["height"]) != (dsm["width"], dsm["height"]):
        raise RuntimeError("Downloaded RGB and DSM dimensions do not match")
    if rgb["bounds"] != dsm["bounds"]:
        raise RuntimeError("Downloaded RGB and DSM bounds do not match")

    metadata = {
        "title": "DepthWizard DC urban feasibility sample",
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "purpose": "Real public RGB/DSM pipeline smoke test; not an SIH/ISRO benchmark",
        "license": "CC BY 4.0",
        "attribution": "District of Columbia Office of the Chief Technology Officer (DC GIS)",
        "important_notice": (
            "The RGB orthophoto was flown in 2021 and the LiDAR DSM in 2020. "
            "The exports share a requested grid, but temporal scene changes make "
            "this unsuitable for claiming model accuracy."
        ),
        "scene": scene_label,
        "bbox": list(bbox),
        "crs": f"EPSG:{SOURCE_CRS}",
        "rgb": {
            "year": 2021,
            "catalog": "https://catalog.data.gov/dataset/aerial-photography-orthophoto-2021",
            "service": ORTHO_SERVICE,
            "export_request": rgb_request,
            "raster": rgb,
        },
        "dsm": {
            "year": 2020,
            "catalog": "https://catalog.data.gov/dataset/2020-lidar-digital-surface-model",
            "service": DSM_SERVICE,
            "export_request": dsm_request,
            "raster": dsm,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Download complete.")
    print(f"RGB: {rgb_path}")
    print(f"DSM: {dsm_path}")
    print("Reminder: this temporally mismatched pair is a smoke test, not an accuracy claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
