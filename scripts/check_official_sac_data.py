#!/usr/bin/env python
"""Report the current contents of ISRO/SAC's SIH DepthWizard repository."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime


REPOSITORY = "IMG-PROCESS-SAC/SIH-DepthWizard-2026"
WEB_URL = f"https://github.com/{REPOSITORY}"
API_ROOT = f"https://api.github.com/repos/{REPOSITORY}"
DATA_SUFFIXES = {
    ".csv",
    ".dem",
    ".hgt",
    ".jp2",
    ".json",
    ".las",
    ".laz",
    ".npy",
    ".npz",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".xml",
    ".zip",
}


def get_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "DepthWizard-SIH26175/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the official ISRO/SAC SIH26175 reference repository"
    )
    parser.add_argument("--ref", default="main", help="Branch or commit (default: main)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    encoded_ref = urllib.parse.quote(args.ref, safe="")
    commit = get_json(f"{API_ROOT}/commits/{encoded_ref}")
    if not isinstance(commit, dict) or "sha" not in commit:
        raise RuntimeError("Unexpected GitHub commit response")
    sha = str(commit["sha"])
    tree = get_json(f"{API_ROOT}/git/trees/{sha}?recursive=1")
    if not isinstance(tree, dict) or not isinstance(tree.get("tree"), list):
        raise RuntimeError("Unexpected GitHub tree response")

    files = sorted(
        {
            str(item["path"])
            for item in tree["tree"]
            if isinstance(item, dict) and item.get("type") == "blob" and item.get("path")
        }
    )
    likely_data = [
        path
        for path in files
        if any(path.lower().endswith(suffix) for suffix in DATA_SUFFIXES)
    ]
    report = {
        "checked_utc": datetime.now(UTC).isoformat(),
        "repository": WEB_URL,
        "requested_ref": args.ref,
        "commit": sha,
        "tree_truncated": bool(tree.get("truncated")),
        "file_count": len(files),
        "files": files,
        "likely_data_or_document_files": likely_data,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Repository: {WEB_URL}")
        print(f"Checked:    {report['checked_utc']}")
        print(f"Commit:     {sha}")
        print(f"Files:      {len(files)}")
        for path in files:
            print(f"  - {path}")
        if files == ["README.md"]:
            print("No sample imagery, DSM, DEM, LiDAR, metadata, or format files are published yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
