# PyInstaller spec for the standalone DepthWizard desktop application.
#
# Build with:
#     python -m PyInstaller packaging/depthwizard.spec --noconfirm
#
# Produces a onedir bundle in dist/DepthWizard. onedir rather than onefile:
# Torch and rasterio ship hundreds of megabytes of native libraries, and a
# onefile build would re-extract all of them to a temporary directory on every
# launch, adding a long delay before the window appears.

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

PROJECT_ROOT = Path(SPECPATH).resolve().parent

datas = []
binaries = []
hiddenimports = [
    "backend.app",
    "ml.depth_anything",
    # Uvicorn resolves its protocol implementations by string at runtime, so
    # static analysis cannot see them.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

# pywebview picks its GUI backend at runtime, so no import statement points at
# webview.platforms.* and PyInstaller collects none of them. Without the
# backends the window never opens and the app runs headless.
for package in ("webview", "clr_loader"):
    try:
        hiddenimports += collect_submodules(package)
        datas += collect_data_files(package)
    except Exception:
        pass

# The built UI is the application; without it the window shows only the API
# landing page. backend.frontend_bundle() looks for it under this exact name.
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if not (frontend_dist / "index.html").is_file():
    raise SystemExit(
        "frontend/dist/index.html is missing. Build the UI first:\n"
        "    cd frontend && npm install && npm run build"
    )
datas.append((str(frontend_dist), "frontend_dist"))

# Model weights, when staged, make the bundle run with no network access.
# Populate with: python packaging/fetch_weights.py
weights = PROJECT_ROOT / "packaging" / "hf_cache"
if (weights / "hub").is_dir():
    datas.append((str(weights), "hf_cache"))

# rasterio and pyproj carry GDAL/PROJ data files that must travel with the
# binary, or opening any GeoTIFF fails at runtime with a projection error.
for package in ("rasterio", "pyproj"):
    try:
        datas += collect_data_files(package)
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

a = Analysis(
    [str(PROJECT_ROOT / "desktop.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Torch bundles its full test suite and both CUDA and CPU toolchains;
    # excluding development-only packages keeps the build from ballooning.
    excludes=["tkinter", "matplotlib", "pytest", "IPython", "notebook"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DepthWizard",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DepthWizard",
)
