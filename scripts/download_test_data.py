#!/usr/bin/env python3
"""Download OopsieVerse test HDF5 demos from Hugging Face into oopsiebench/test_data/."""

from __future__ import annotations

import argparse
import os
import sys

# Set when the dataset is published on Hugging Face Hub.
HF_REPO_ID = os.environ.get("OOPSIEVERSE_HF_DATASET", "UT-Austin-RobIn/oopsieverse-test-data")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DATA_ROOT = os.path.join(_REPO_ROOT, "oopsiebench", "test_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo",
        default=HF_REPO_ID,
        help="Hugging Face dataset repo id (default: env OOPSIEVERSE_HF_DATASET or %(default)s)",
    )
    p.add_argument(
        "--dest",
        default=TEST_DATA_ROOT,
        help="Local directory for downloaded files (default: oopsiebench/test_data)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Install huggingface_hub first:\n"
            "  pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.dest, exist_ok=True)
    print(f"Downloading test data from {args.repo} -> {args.dest}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.dest,
        local_dir_use_symlinks=False,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
