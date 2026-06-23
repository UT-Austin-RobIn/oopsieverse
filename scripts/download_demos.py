#!/usr/bin/env python3
"""
Download OopsieVerse teleop demos from the public Hugging Face dataset into the
local oopsiebench/demos/ folder, preserving the repo structure:

    oopsiebench/demos/robocasa/teleop/<task>_safe.hdf5
    oopsiebench/demos/robocasa/teleop/<task>_unsafe.hdf5
    oopsiebench/demos/behavior1k/teleop/<task>_safe.hdf5
    oopsiebench/demos/behavior1k/teleop/<task>_unsafe.hdf5

Pick which simulator(s) to pull. The script reports how much data will be
downloaded and asks for confirmation first.

Usage:
    python scripts/download_demos.py                       # default: both (all)
    python scripts/download_demos.py --sim robocasa
    python scripts/download_demos.py --sim behavior1k

The dataset is public, so no login/token is required.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "ut-robin-lab/oopsieverse-demos"
REPO_TYPE = "dataset"
SIMS = ("robocasa", "behavior1k")  # top-level folders in the repo

# Download into <repo_root>/oopsiebench/demos so the layout matches the repo.
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "oopsiebench", "demos",
)


def human(nbytes):
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main():
    parser = argparse.ArgumentParser(
        description="Download OopsieVerse teleop demos from Hugging Face.")
    parser.add_argument(
        "--sim", default="both",
        choices=["robocasa", "behavior1k", "both"],
        help="which simulator's data to download (default: both = all)")
    args = parser.parse_args()

    sims = list(SIMS) if args.sim == "both" else [args.sim]
    prefixes = tuple(f"{s}/" for s in sims)

    api = HfApi()
    try:
        entries = api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True)
    except Exception as e:
        sys.exit(f"Could not reach the dataset (online? is it public?): {e}")

    total, count = 0, 0
    for e in entries:
        size = getattr(e, "size", None)  # RepoFile has size; RepoFolder doesn't
        if size is not None and e.path.startswith(prefixes):
            total += size
            count += 1

    if count == 0:
        sys.exit(f"No files found for: {', '.join(sims)}")

    print(f"\nAbout to download {count} file(s), {human(total)} total, "
          f"for [{', '.join(sims)}]")
    print(f"  from: https://huggingface.co/datasets/{REPO_ID}")
    print(f"  into: {LOCAL_DIR}\n")

    resp = input("Proceed with download? [y/N] ").strip().lower()
    if resp not in ("y", "yes"):
        print("Aborted. Nothing was downloaded.")
        return

    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=LOCAL_DIR,
        allow_patterns=[f"{s}/*" for s in sims],
    )
    print(f"\nDone. Files are in: {LOCAL_DIR}")


if __name__ == "__main__":
    main()
