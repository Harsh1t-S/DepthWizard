#!/usr/bin/env python
"""Run real Depth Anything V2 inference on one image from the command line."""

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
        description="Run Depth Anything V2 Small and export relative-depth artifacts."
    )
    parser.add_argument("image", type=Path, help="RGB image or GeoTIFF")
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
    if not args.image.is_file():
        print(f"Image does not exist: {args.image}", file=sys.stderr)
        return 2
    estimator = DepthEstimator(
        model_id=args.model_id,
        device=args.device,
        max_input_size=args.max_input_size,
    )
    try:
        result = analyze_bytes(
            args.image.read_bytes(),
            args.image.name,
            estimator=estimator,
            artifact_root=args.output_dir,
        )
    except Exception as exc:
        print(f"Depth inference failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "job_id": result["job_id"],
        "model": result["model"],
        "device": result["device"],
        "mode": result["mode"],
        "processing_time_seconds": result["processing_time_seconds"],
        "artifacts": result["artifacts"],
        "notices": result["notices"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
