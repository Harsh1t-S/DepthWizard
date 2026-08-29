"""Package a built DepthWizard bundle into an archive for sharing.

Run after PyInstaller, pointing at the bundle directory to ship:

    python scripts/pack_distribution.py --source dist-cpu/DepthWizard

Produces a single DEFLATE zip, which every Windows machine can open from
Explorer with no extra tool. Only when the archive exceeds the sharing limit
does it also emit numbered parts and a batch file that rejoins them.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"

# WhatsApp refuses a document larger than 2 GB.
SHARING_LIMIT_BYTES = 2 * 1024**3
PART_SIZE_BYTES = 900 * 1024**2


def _gigabytes(size: int) -> str:
    return f"{size / 1024**3:.2f} GB ({size:,} bytes)"


def compress(source: Path, output: Path) -> Path:
    print(f"Compressing {source} ...")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    # DEFLATE, not LZMA or BZIP2: the recipient must be able to right-click and
    # extract with the Windows shell, which reads no other method.
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for directory, _subdirs, files in os.walk(source):
            for name in files:
                path = Path(directory) / name
                archive.write(path, str(path.relative_to(source.parent)))

    print(f"Wrote {output.name}: {_gigabytes(output.stat().st_size)}")
    return output


def split(archive: Path, part_size: int = PART_SIZE_BYTES) -> list[Path]:
    """Cut an oversized archive into parts, with a script that rejoins them."""

    split_dir = archive.parent / f"{archive.stem}-Split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for stale in split_dir.iterdir():
        if stale.is_file():
            stale.unlink()

    parts: list[Path] = []
    with open(archive, "rb") as source:
        while True:
            chunk = source.read(part_size)
            if not chunk:
                break
            part = split_dir / f"{archive.name}.{len(parts) + 1:03d}"
            part.write_bytes(chunk)
            print(f"  {part.name}: {part.stat().st_size / 1024**2:.1f} MB")
            parts.append(part)

    # Built from the parts that exist, not a fixed count: a hard-coded list
    # silently rejoins the wrong set the moment the bundle size changes.
    joined = " + ".join(part.name for part in parts)
    (split_dir / "1-CLICK-UNPACK.bat").write_text(
        "@echo off\r\n"
        "echo Combining parts...\r\n"
        f"copy /b {joined} {archive.name} >nul\r\n"
        "echo Extracting...\r\n"
        f'powershell -command "Expand-Archive -Path {archive.name} '
        '-DestinationPath . -Force"\r\n'
        "echo Done. Run DepthWizard\\DepthWizard.exe\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "dist-cpu" / "DepthWizard",
        help="bundle directory produced by PyInstaller",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DIST_DIR / "DepthWizard-Windows.zip",
        help="archive to write",
    )
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    if not source.is_dir():
        parser.error(f"{source} does not exist. Run PyInstaller first.")

    archive = compress(source, arguments.output.resolve())
    size = archive.stat().st_size
    if size <= SHARING_LIMIT_BYTES:
        print(f"Under the {SHARING_LIMIT_BYTES / 1024**3:.0f} GB sharing limit. Send as one file.")
        return 0

    print(f"Over the {SHARING_LIMIT_BYTES / 1024**3:.0f} GB sharing limit; splitting.")
    split(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
