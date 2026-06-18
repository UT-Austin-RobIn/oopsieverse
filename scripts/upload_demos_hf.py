#!/usr/bin/env python3
"""
Upload OopsieVerse teleop demos to a Hugging Face dataset.

Hosts ONLY the teleop HDF5s for BEHAVIOR-1K and RoboCasa, preserving the
folder layout so the dataset repo looks like:

    behavior1k/teleop/<task>.hdf5
    robocasa/teleop/<task>_safe.hdf5
    robocasa/teleop/<task>_unsafe.hdf5

Playback HDF5s are intentionally NOT uploaded (they're large and regenerable).
The HDF5s are stored as-is via Git LFS (the Hub auto-tracks large files), so
they download byte-for-byte identical — no parquet conversion, no data loss.

Usage:
    pip install -U "huggingface_hub>=0.23"
    hf auth login                  # paste a token with WRITE access (once)
    python scripts/upload_demos_hf.py
"""

import os
import sys
from huggingface_hub import HfApi

# ── Config ───────────────────────────────────────────────────────────────
REPO_ID = "arnavbalaji21/oopsieverse_demos"
REPO_TYPE = "dataset"
PRIVATE = False  # set True to keep it private (100 GB free-tier limit applies)

# Local root that contains behavior1k/ and robocasa/ subfolders.
LOCAL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "oopsiebench", "demos",
)

# Only upload the teleop folders (not playback).
ALLOW_PATTERNS = [
    "behavior1k/teleop/*.hdf5",
    "robocasa/teleop/*.hdf5",
]

# Never upload personal / ad-hoc / stub files, even if still present locally.
IGNORE_PATTERNS = [
    "*adhoc_*",                              # ad-hoc robocasa collections
    "*wine_glass*",                          # personal, not an official task
    "behavior1k/teleop/default.hdf5",        # "default" placeholder
    "behavior1k/teleop/firewood.hdf5",       # alias of add_firewood
    "behavior1k/teleop/place_bowl.hdf5",     # empty stub
    "behavior1k/teleop/heat_saucepot_scene.hdf5",  # empty stub variant
]


def main():
    if not os.path.isdir(LOCAL_ROOT):
        sys.exit(f"Local folder not found: {LOCAL_ROOT}")

    # Auth comes from `hf auth login` (cached token) or the HF_TOKEN env var.
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE,
                    private=PRIVATE, exist_ok=True)

    api.upload_folder(
        folder_path=LOCAL_ROOT,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        # Mirror the local teleop folders exactly: prune any remote teleop HDF5
        # that no longer exists locally (e.g. old non-split files).
        delete_patterns=ALLOW_PATTERNS,
        commit_message="Sync BEHAVIOR-1K + RoboCasa teleop demos (safe/unsafe splits)",
    )

    print(f"\nDone. Browse the files at:\n"
          f"  https://huggingface.co/datasets/{REPO_ID}/tree/main")


if __name__ == "__main__":
    main()
