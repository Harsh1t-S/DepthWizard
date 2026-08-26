"""Compare monocular depth backbones for edge quality on aerial imagery.

No ground-truth DSM is required. The metrics measure whether predicted depth
edges land on real image edges and how crisp those transitions are, which is
what determines whether buildings reconstruct as sharp blocks or rounded mounds.

    python scripts/compare_depth_backbones.py --image path/to/scene.jpg

Metrics, all higher-is-better:

* ``edge_alignment`` - Pearson correlation between the depth gradient magnitude
  and the image gradient magnitude. Low values mean the model is inventing
  structure that is not in the picture.
* ``edge_contrast`` - 95th-percentile gradient divided by the median gradient.
  A blurry prediction spreads gradient everywhere and scores near 1; a crisp
  one concentrates it at boundaries.
* ``flat_cleanliness`` - inverse of the median gradient over non-edge regions,
  normalised. Penalises models that are merely noisy rather than detailed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CANDIDATES = {
    "v2-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "v2-large": "depth-anything/Depth-Anything-V2-Large-hf",
    "depth-pro": "apple/DepthPro-hf",
    "metric-outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(array.astype(np.float32))
    return np.hypot(dx, dy)


def _normalize(array: np.ndarray) -> np.ndarray:
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(array[finite], (1.0, 99.0))
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def score(depth: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    depth_gradient = _gradient_magnitude(_normalize(depth))
    image_gradient = _gradient_magnitude(gray / 255.0)

    a = depth_gradient.ravel()
    b = image_gradient.ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        alignment = 0.0
    else:
        alignment = float(np.corrcoef(a, b)[0, 1])

    median = float(np.median(depth_gradient))
    high = float(np.percentile(depth_gradient, 95.0))
    contrast = high / median if median > 1e-9 else float("inf")

    # "Flat" = the quietest half of the image by image gradient. A good model is
    # smooth there; a noisy one is not.
    threshold = np.percentile(image_gradient, 50.0)
    flat = depth_gradient[image_gradient <= threshold]
    flat_median = float(np.median(flat)) if flat.size else 0.0

    return {
        "edge_alignment": round(alignment, 4),
        "edge_contrast": round(contrast, 3),
        "flat_noise": round(flat_median, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--models", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/backbone-comparison"))
    parser.add_argument("--max-input-size", type=int, default=1024)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.image).convert("RGB")
    gray = np.asarray(source.convert("L"), dtype=np.float32)
    print(f"Scene: {args.image.name}  {source.size[0]}x{source.size[1]}\n")

    from ml.depth_anything import DepthEstimator

    results: list[dict[str, Any]] = []
    for alias in args.models:
        model_id = CANDIDATES.get(alias, alias)
        print(f"[{alias}] {model_id}", flush=True)
        entry: dict[str, Any] = {"alias": alias, "model_id": model_id}
        try:
            estimator = DepthEstimator(model_id=model_id, max_input_size=args.max_input_size)
            started = time.perf_counter()
            depth, info = estimator.predict(source)
            entry["seconds"] = round(time.perf_counter() - started, 2)
            entry["device"] = info.device
            entry.update(score(depth, gray))

            preview = (_normalize(depth) * 255).astype(np.uint8)
            out = args.output_dir / f"depth_{alias}.png"
            Image.fromarray(preview).save(out)
            entry["preview"] = str(out)
            print(f"    {entry['seconds']}s  {ded(entry)}\n", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad backbone must not stop the sweep
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {entry['error']}\n", flush=True)
        results.append(entry)

    report = args.output_dir / "comparison.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = [r for r in results if "edge_alignment" in r]
    if ok:
        print("=" * 74)
        print(f"{'model':24s} {'align':>8s} {'contrast':>10s} {'flat noise':>12s} {'sec':>7s}")
        for r in sorted(ok, key=lambda x: -x["edge_alignment"]):
            print(
                f"{r['alias']:24s} {r['edge_alignment']:>8.4f} {r['edge_contrast']:>10.3f} "
                f"{r['flat_noise']:>12.6f} {r['seconds']:>7.2f}"
            )
        print("\nHigher alignment and contrast are better; lower flat noise is better.")
    print(f"\nReport: {report}")
    return 0


def ded(entry: dict[str, Any]) -> str:
    return (
        f"align={entry['edge_alignment']:.4f} "
        f"contrast={entry['edge_contrast']:.3f} "
        f"flat_noise={entry['flat_noise']:.6f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
