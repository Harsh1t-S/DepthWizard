#!/usr/bin/env python
"""Calibrate live relative depth from a reference DEM or sparse GCPs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import analyze_bytes  # noqa: E402
from ml.depth_anything import DepthEstimator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Depth Anything V2 and fit a deployment-style affine height "
            "calibration from exactly one reference DEM or GCP CSV/JSON file."
        )
    )
    parser.add_argument(
        "--image", type=Path, required=True, help="RGB image or source GeoTIFF"
    )
    references = parser.add_mutually_exclusive_group(required=True)
    references.add_argument(
        "--reference-dem",
        type=Path,
        help="Coarse/reference elevation GeoTIFF, NPY, or image",
    )
    references.add_argument(
        "--gcps",
        type=Path,
        help="UTF-8 CSV or JSON with zero-based pixel x/y/elevation points",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional independent evaluation DSM; never used to refit reference calibration",
    )
    parser.add_argument(
        "--gcp-sampling",
        choices=("bilinear", "nearest"),
        default="bilinear",
        help="Depth sampling at GCP coordinates (default: bilinear)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("depthwizard-output"),
        help="Artifact root (default: ./depthwizard-output)",
    )
    parser.add_argument(
        "--max-input-size",
        type=int,
        default=None,
        help="Maximum model-input long edge in pixels (default: env or 1024)",
    )
    parser.add_argument("--model-id", default=None, help="Hugging Face model ID override")
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=None,
        help="Inference device override (default: auto-detect)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [args.image, args.reference_dem or args.gcps]
    if args.ground_truth is not None:
        inputs.append(args.ground_truth)
    missing = [path for path in inputs if path is not None and not path.is_file()]
    if missing:
        print(f"Input does not exist: {missing[0]}", file=sys.stderr)
        return 2

    estimator = DepthEstimator(
        model_id=args.model_id,
        device=args.device,
        max_input_size=args.max_input_size,
    )
    reference_kwargs = {}
    if args.reference_dem is not None:
        reference_kwargs = {
            "reference_dem_bytes": args.reference_dem.read_bytes(),
            "reference_dem_filename": args.reference_dem.name,
        }
    else:
        reference_kwargs = {
            "gcps_bytes": args.gcps.read_bytes(),
            "gcps_filename": args.gcps.name,
            "gcp_sampling": args.gcp_sampling,
        }
    try:
        result = analyze_bytes(
            args.image.read_bytes(),
            args.image.name,
            args.ground_truth.read_bytes() if args.ground_truth else None,
            args.ground_truth.name if args.ground_truth else None,
            estimator=estimator,
            artifact_root=args.output_dir,
            **reference_kwargs,
        )
    except Exception as exc:
        print(f"Depth calibration failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "job_id": result["job_id"],
        "model": result["model"],
        "device": result["device"],
        "mode": result["mode"],
        "processing_time_seconds": result["processing_time_seconds"],
        "metrics": result["metrics"],
        "calibration": result["calibration"],
        "reference": result["reference"],
        "artifacts": result["artifacts"],
        "notices": result["notices"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
