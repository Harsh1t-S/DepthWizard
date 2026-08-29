"""Standalone DepthWizard desktop entry point.

Starts the FastAPI service on a loopback port and opens it in a native window,
so the whole application is one process the user launches directly rather than
a backend and a frontend server started separately.

Run from a source checkout with:

    python desktop.py

The UI bundle must exist first (``cd frontend && npm run build``); without it
the window shows the API landing page instead.
"""

from __future__ import annotations

import os
import socket
import sys

# Remove any external shadowing tool paths if present
sys.path = [p for p in sys.path if "claude-tools" not in p]
if "claude-tools" in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = ""

import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

APP_TITLE = "DepthWizard — Single-View Height Estimation"
HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 60.0


def _bundle_root() -> Path:
    """Directory holding bundled resources, for both source and frozen runs."""

    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parent


def _configure_environment() -> None:
    """Apply desktop defaults without overriding a deliberate setting."""

    root = _bundle_root()

    # Writable state must not live inside the read-only PyInstaller extraction
    # directory, so job artifacts go to a per-user application data folder.
    if not os.getenv("DEPTHWIZARD_ARTIFACT_DIR"):
        base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
        artifacts = Path(base) / "DepthWizard" / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        os.environ["DEPTHWIZARD_ARTIFACT_DIR"] = str(artifacts)

    # Prefer weights shipped alongside the application so a packaged build runs
    # with no network access; fall back to the user's normal Hugging Face cache.
    offline_cache = root / "hf_cache"
    if offline_cache.is_dir() and not os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = str(offline_cache)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _free_port() -> int:
    configured = os.getenv("DEPTHWIZARD_PORT", "").strip()
    if configured:
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])


def _serve(port: int) -> None:
    import uvicorn

    from backend.app import app

    uvicorn.run(app, host=HOST, port=port, log_level="warning")


def _wait_until_ready(url: str, timeout: float = STARTUP_TIMEOUT_SECONDS) -> bool:
    """Poll the health endpoint so the window never opens on a dead port."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.25)
    return False


def _open_window(url: str) -> None:
    """Show a native window, falling back to the default browser."""

    def _browser_fallback(reason: str) -> None:
        print(f"Native window unavailable ({reason}); opening a browser.", file=sys.stderr)
        print(f"DepthWizard is running at {url}", flush=True)
        webbrowser.open(url)
        # Nothing else owns the main thread in the browser fallback, so block
        # here; the serving thread is a daemon and would otherwise exit at once.
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass

    # A missing GUI backend is not an ImportError: pywebview imports fine and
    # only fails when start() cannot resolve a platform. Catching just
    # ImportError leaves the app running with no window and no message.
    try:
        import webview

        webview.create_window(APP_TITLE, url, width=1480, height=940, min_size=(1024, 700))
        webview.start()
    except ImportError:
        _browser_fallback("pywebview is not installed")
    except Exception as exc:  # noqa: BLE001 - any backend failure must still show the UI
        _browser_fallback(f"{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    # --no-window runs the same single-origin service without a GUI, which is
    # how the launcher is smoke-tested and how it can be used as a plain server.
    headless = "--no-window" in args

    _configure_environment()

    from backend.app import frontend_bundle

    if frontend_bundle() is None:
        print(
            "No UI bundle found. Build it first:\n"
            "    cd frontend && npm install && npm run build",
            file=sys.stderr,
        )

    port = _free_port()
    url = f"http://{HOST}:{port}"

    threading.Thread(target=_serve, args=(port,), daemon=True).start()

    if not _wait_until_ready(url):
        print(
            f"DepthWizard did not become ready within {STARTUP_TIMEOUT_SECONDS:.0f}s.",
            file=sys.stderr,
        )
        return 1

    print(f"DepthWizard ready at {url}", flush=True)
    if headless:
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        return 0

    _open_window(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
