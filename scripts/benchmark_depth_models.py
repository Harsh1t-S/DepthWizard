#!/usr/bin/env python
"""Compare Depth Anything V2 variants on one aligned RGB/DSM pair.

This is a feasibility benchmark.  Its affine calibration is fit against every
valid ground-truth pixel in the same scene, so the resulting MAE/RMSE are not a
claim about performance on unseen imagery or deployable absolute elevation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation import evaluate_relative_depth  # noqa: E402
from backend.raster_io import (  # noqa: E402
    align_ground_truth,
    read_ground_truth_bytes,
    read_image_bytes,
)
from ml.depth_anything import DepthEstimator, select_device  # noqa: E402


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "small": {
        "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "parameters_millions": 24.8,
        "estimated_download_mib": 100,
        "estimated_fp32_vram_gib_at_700px": "1-3",
    },
    "base": {
        "model_id": "depth-anything/Depth-Anything-V2-Base-hf",
        "parameters_millions": 97.5,
        "estimated_download_mib": 390,
        "estimated_fp32_vram_gib_at_700px": "3-6",
    },
    "large": {
        "model_id": "depth-anything/Depth-Anything-V2-Large-hf",
        "parameters_millions": 335.3,
        "estimated_download_mib": 1350,
        "estimated_fp32_vram_gib_at_700px": "8-14",
    },
}

BENCHMARK_LABEL = "Benchmark calibration / feasibility evaluation"
BENCHMARK_WARNING = (
    "Affine scale and offset are fit against all valid pixels of this same DSM. "
    "Metrics measure same-scene technical feasibility, not unseen-scene metric "
    "elevation accuracy."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Depth Anything V2 variants on an aligned RGB/DSM pair. "
            "Small and Base are selected by default; Large has a VRAM guard."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/sample/dc-urban/rgb_2021.tif"),
        help="RGB image or GeoTIFF (default: DC smoke-test RGB)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/sample/dc-urban/dsm_2020.tif"),
        help="Aligned DSM raster (default: DC smoke-test DSM)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_PROFILES),
        default=["small", "base"],
        help="Model variants in run order (default: small base)",
    )
    parser.add_argument(
        "--max-input-size",
        type=int,
        default=700,
        help="Common model-input long edge; at least 256 (default: 700)",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=1,
        help="Timed inferences after one load/warm-up inference (default: 1)",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default=None,
        help="Inference device (default: auto-detect)",
    )
    parser.add_argument(
        "--force-large",
        action="store_true",
        help="Run Large despite the default 12 GiB CUDA/accelerator safety guard",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model-benchmark.json"),
        help="JSON report path (default: outputs/model-benchmark.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned models and hardware without loading model weights",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cuda_synchronize(device: str) -> None:
    if device != "cuda":
        return
    import torch

    torch.cuda.synchronize()


def _hardware(device: str) -> dict[str, Any]:
    import torch
    import transformers

    details: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "selected_device": device,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        details.update(
            cuda_device_name=properties.name,
            cuda_total_memory_gib=round(properties.total_memory / 1024**3, 3),
        )
    return details


def _large_safety_reason(device: str, hardware: dict[str, Any]) -> str | None:
    if device != "cuda":
        return (
            "Large is skipped by default on CPU/MPS because this FP32 pipeline is "
            "slow or memory-intensive; pass --force-large to accept that risk."
        )
    total_gib = hardware.get("cuda_total_memory_gib")
    if total_gib is None or float(total_gib) < 12.0:
        return (
            "Large is skipped because the selected CUDA GPU has less than the "
            "recommended 12 GiB for this FP32 benchmark; pass --force-large to "
            "accept possible out-of-memory failure."
        )
    return None


def _model_parameter_summary(estimator: DepthEstimator) -> dict[str, Any]:
    model = getattr(estimator, "_model", None)
    if model is None:
        return {}
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    storage_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    config = getattr(model, "config", None)
    return {
        "parameter_count": int(parameter_count),
        "parameter_storage_mib_loaded": round(storage_bytes / 1024**2, 3),
        "parameter_dtype": str(parameters[0].dtype) if parameters else None,
        "model_revision": getattr(config, "_commit_hash", None),
    }


def _cleanup_accelerator(device: str) -> None:
    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


def _run_model(
    alias: str,
    *,
    source_image: Image.Image,
    source_valid_mask: Any,
    ground_truth: Any,
    ground_truth_valid_mask: Any,
    device: str,
    max_input_size: int,
    timed_runs: int,
) -> dict[str, Any]:
    import torch

    profile = MODEL_PROFILES[alias]
    estimator = DepthEstimator(
        model_id=str(profile["model_id"]),
        device=device,
        max_input_size=max_input_size,
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # The first inference includes local-cache/model loading and acts as warm-up.
    _cuda_synchronize(device)
    warmup_started = time.perf_counter()
    depth, prediction_info = estimator.predict(source_image)
    _cuda_synchronize(device)
    load_and_warmup_seconds = time.perf_counter() - warmup_started

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inference_seconds: list[float] = []
    for _ in range(timed_runs):
        _cuda_synchronize(device)
        started = time.perf_counter()
        depth, prediction_info = estimator.predict(source_image)
        _cuda_synchronize(device)
        inference_seconds.append(time.perf_counter() - started)

    evaluation = evaluate_relative_depth(
        depth,
        ground_truth,
        ground_truth_valid_mask,
        source_valid_mask,
    )
    metrics = evaluation.metrics
    calibration = evaluation.calibration
    result: dict[str, Any] = {
        "status": "completed",
        "variant": alias,
        "model_id": profile["model_id"],
        "device": prediction_info.device,
        "max_input_size": max_input_size,
        "inference_grid": {
            "width": prediction_info.inference_width,
            "height": prediction_info.inference_height,
        },
        "load_and_warmup_seconds": round(load_and_warmup_seconds, 6),
        "timed_inference_seconds": [round(value, 6) for value in inference_seconds],
        "median_inference_seconds": round(statistics.median(inference_seconds), 6),
        "raw_pearson_correlation": metrics["raw_pearson_correlation"],
        "oriented_relative_pearson_correlation": metrics[
            "relative_pearson_correlation"
        ],
        "affine_calibrated_pearson_correlation": metrics[
            "calibrated_pearson_correlation"
        ],
        "affine_calibrated_mae": metrics["calibrated_mae"],
        "affine_calibrated_rmse": metrics["calibrated_rmse"],
        "valid_pixels": metrics["valid_pixels"],
        "calibration": {
            "orientation": calibration["orientation"],
            "scale": calibration["scale"],
            "shift": calibration["shift"],
            "fit_scope": "all_valid_pixels_in_same_ground_truth_dsm",
        },
        **_model_parameter_summary(estimator),
    }
    if device == "cuda":
        result["cuda_peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / 1024**2, 3
        )
    del estimator
    _cleanup_accelerator(device)
    return result


def main() -> int:
    args = parse_args()
    if args.max_input_size < 256:
        raise SystemExit("--max-input-size must be at least 256")
    if args.timed_runs < 1:
        raise SystemExit("--timed-runs must be at least 1")

    device = args.device or select_device()
    hardware = _hardware(device)
    plan = []
    for alias in args.models:
        safety_reason = (
            _large_safety_reason(device, hardware) if alias == "large" else None
        )
        plan.append(
            {
                "variant": alias,
                **MODEL_PROFILES[alias],
                "will_run": not safety_reason or args.force_large,
                "safety_notice": safety_reason,
            }
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "label": BENCHMARK_LABEL,
                    "warning": BENCHMARK_WARNING,
                    "hardware": hardware,
                    "max_input_size": args.max_input_size,
                    "timed_runs": args.timed_runs,
                    "plan": plan,
                },
                indent=2,
            )
        )
        return 0

    missing = [path for path in (args.image, args.ground_truth) if not path.is_file()]
    if missing:
        print(f"Input does not exist: {missing[0]}", file=sys.stderr)
        return 2

    image_path = args.image.resolve()
    ground_truth_path = args.ground_truth.resolve()
    source = read_image_bytes(image_path.read_bytes(), image_path.name)
    ground_truth_raster = read_ground_truth_bytes(
        ground_truth_path.read_bytes(), ground_truth_path.name
    )
    aligned_ground_truth, aligned_valid, alignment_notices = align_ground_truth(
        ground_truth_raster, source
    )
    source_image = Image.fromarray(source.rgb)

    source_metadata_path = image_path.parent / "SOURCE.json"
    source_metadata: dict[str, Any] | None = None
    if source_metadata_path.is_file():
        try:
            source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            source_metadata = {"notice": "SOURCE.json could not be decoded"}

    report: dict[str, Any] = {
        "label": BENCHMARK_LABEL,
        "warning": BENCHMARK_WARNING,
        "evaluation_type": "same_scene_full_ground_truth_affine_refit",
        "generated_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hardware": hardware,
        "settings": {
            "max_input_size": args.max_input_size,
            "timed_runs_after_warmup": args.timed_runs,
            "models": list(args.models),
        },
        "dataset": {
            "image": str(image_path),
            "ground_truth": str(ground_truth_path),
            "image_sha256": _sha256(image_path),
            "ground_truth_sha256": _sha256(ground_truth_path),
            "width": source.width,
            "height": source.height,
            "valid_ground_truth_pixels": int(aligned_valid.sum()),
            "alignment_notices": alignment_notices,
            "source_metadata": source_metadata,
        },
        "model_plan": plan,
        "results": [],
    }

    for planned in plan:
        alias = str(planned["variant"])
        print(f"Benchmarking {alias}: {planned['model_id']}", file=sys.stderr)
        if not planned["will_run"]:
            report["results"].append(
                {
                    "status": "skipped_safety",
                    "variant": alias,
                    "model_id": planned["model_id"],
                    "reason": planned["safety_notice"],
                }
            )
            continue
        try:
            result = _run_model(
                alias,
                source_image=source_image,
                source_valid_mask=source.valid_mask,
                ground_truth=aligned_ground_truth,
                ground_truth_valid_mask=aligned_valid,
                device=device,
                max_input_size=args.max_input_size,
                timed_runs=args.timed_runs,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "variant": alias,
                "model_id": planned["model_id"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            _cleanup_accelerator(device)
        report["results"].append(result)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report written to {output_path}", file=sys.stderr)
    return 0 if any(item["status"] == "completed" for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
