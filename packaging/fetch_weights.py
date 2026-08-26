"""Stage model weights so a packaged build runs without network access.

Downloads the configured Depth Anything V2 checkpoint into
``packaging/hf_cache``. The PyInstaller spec picks that directory up
automatically, and ``desktop.py`` points ``HF_HOME`` at it and enables offline
mode when it is present in the bundle.

    python packaging/fetch_weights.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "hf_cache"
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"


def main() -> int:
    model_id = os.getenv("DEPTHWIZARD_MODEL_ID", DEFAULT_MODEL_ID)

    # Must be set before transformers is imported, or the download lands in the
    # user's default cache instead of the staging directory.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print(f"Fetching {model_id} into {CACHE_DIR} ...", flush=True)
    AutoImageProcessor.from_pretrained(model_id)
    AutoModelForDepthEstimation.from_pretrained(model_id)

    total = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
    print(f"Staged {total / 1_048_576:.0f} MB. The packaged build will run offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
