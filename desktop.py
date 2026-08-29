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
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

APP_TITLE = "DepthWizard — Single-View Height Estimation"
HOST = "127.0.0.1"
# A frozen bundle cold-starts in roughly 60-90 s on this machine, longer on a
# slower disk, so a 60 s budget can expire while the import is still running
# and take the app down on exactly the machines least able to spare it.
STARTUP_TIMEOUT_SECONDS = 180.0
LOG_FILE_NAME = "DepthWizard.log"


def _isolate_import_path() -> None:
    """Import from this project and its environment only.

    PYTHONPATH is searched ahead of the virtual environment, so a machine-wide
    setting aimed at an unrelated package directory shadows this project's own
    dependencies -- with extension modules built for a different Python version,
    which fails at import with nothing that names the cause. A frozen build is
    covered by packaging/rthook_isolate_path.py, which runs earlier still.
    """

    if getattr(sys, "frozen", False):
        return
    injected = [
        entry
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    if not injected:
        return
    unwanted = {os.path.abspath(entry) for entry in injected}
    unwanted.discard(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [
        entry for entry in sys.path if os.path.abspath(entry) not in unwanted
    ]
    os.environ.pop("PYTHONPATH", None)


_isolate_import_path()


def _bundle_root() -> Path:
    """Directory holding bundled resources, for both source and frozen runs."""

    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parent


def _state_dir() -> Path:
    """Per-user writable directory for artifacts and logs."""

    base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(base) / "DepthWizard"


def _ensure_std_streams() -> None:
    """Guarantee real ``sys.stdout`` and ``sys.stderr`` objects.

    A windowed build launched from Explorer gets no console, and PyInstaller
    then leaves both streams set to ``None``. uvicorn builds its log formatter
    with ``sys.stdout.isatty()``, so ``uvicorn.run`` raises
    ``ValueError: Unable to configure formatter 'default'`` before the socket
    is ever bound. The serving thread is a daemon, so nothing surfaces: the
    launcher waits out the readiness timeout and exits with no window and no
    message. Point the streams at a log file instead, which also leaves a
    record behind when a packaged run does fail.
    """

    if sys.stdout is not None and sys.stderr is not None:
        return

    stream = None
    try:
        directory = _state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        stream = open(directory / LOG_FILE_NAME, "a", encoding="utf-8", buffering=1)
    except OSError:
        try:
            stream = open(os.devnull, "w", encoding="utf-8")
        except OSError:
            return

    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _configure_environment() -> None:
    """Apply desktop defaults without overriding a deliberate setting."""

    root = _bundle_root()

    # Writable state must not live inside the read-only PyInstaller extraction
    # directory, so job artifacts go to a per-user application data folder.
    if not os.getenv("DEPTHWIZARD_ARTIFACT_DIR"):
        artifacts = _state_dir() / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        os.environ["DEPTHWIZARD_ARTIFACT_DIR"] = str(artifacts)

    # Caches follow the same rule: the working directory a packaged build is
    # launched from is not ours to write into.
    if not os.getenv("DEPTHWIZARD_CACHE_DIR"):
        cache = _state_dir() / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["DEPTHWIZARD_CACHE_DIR"] = str(cache)

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


def _serve(port: int, failure: list[str]) -> None:
    """Run the API. Record why it stopped, since a daemon thread dies quietly."""

    try:
        import uvicorn

        from backend.app import app

        uvicorn.run(app, host=HOST, port=port, log_level="warning")
    except BaseException:  # noqa: BLE001 - the launcher must report any cause
        failure.append(traceback.format_exc())
        raise


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

    _ensure_std_streams()
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

    failure: list[str] = []
    threading.Thread(target=_serve, args=(port, failure), daemon=True).start()

    if not _wait_until_ready(url):
        detail = failure[0] if failure else (
            f"No response within {STARTUP_TIMEOUT_SECONDS:.0f}s."
        )
        print("DepthWizard did not start.", file=sys.stderr)
        print(detail, file=sys.stderr)
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
