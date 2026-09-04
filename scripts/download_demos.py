#!/usr/bin/env python3
"""
Download OopsieVerse teleop demos from the public Hugging Face dataset into the
local oopsiebench/demos/ folder, preserving the repo structure:

    oopsiebench/demos/robocasa/teleop/<task>_safe.hdf5
    oopsiebench/demos/robocasa/teleop/<task>_unsafe.hdf5
    oopsiebench/demos/behavior1k/teleop/<task>_safe.hdf5
    oopsiebench/demos/behavior1k/teleop/<task>_unsafe.hdf5

Paper live-feedback study demos (``--paper-demos``):

    oopsiebench/demos/paper_demos/teleop_data/<task>/{without,with,all_data}.hdf5

Pick which simulator(s) to pull, or the paper-demos set. The script reports how
much data will be downloaded and asks for confirmation first.

Usage:
    python scripts/download_demos.py                       # default: both sims
    python scripts/download_demos.py --sim robocasa
    python scripts/download_demos.py --sim behavior1k
    python scripts/download_demos.py --paper-demos

The dataset is public, so no login/token is required.
"""

import argparse
import os
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "ut-robin-lab/oopsieverse-demos"
REPO_TYPE = "dataset"
SIMS = ("robocasa", "behavior1k")  # top-level folders in the repo
PAPER_HF_PREFIX = "paper_experiments/teleop_data"

# Download into <repo_root>/oopsiebench/demos so the layout matches the repo.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(REPO_ROOT, "oopsiebench", "demos")
PAPER_LOCAL_DIR = os.path.join(LOCAL_DIR, "paper_demos")


def human(nbytes):
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def _count_matching(entries, prefixes):
    total, count = 0, 0
    for e in entries:
        size = getattr(e, "size", None)  # RepoFile has size; RepoFolder doesn't
        if size is not None and e.path.startswith(prefixes):
            total += size
            count += 1
    return count, total


def _download_sims(sims):
    prefixes = tuple(f"{s}/" for s in sims)

    api = HfApi()
    try:
        entries = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True))
    except Exception as e:
        sys.exit(f"Could not reach the dataset (online? is it public?): {e}")

    count, total = _count_matching(entries, prefixes)
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


def _download_paper_demos():
    """Download paper_experiments/teleop_data → oopsiebench/demos/paper_demos/teleop_data."""
    prefix = PAPER_HF_PREFIX + "/"

    api = HfApi()
    try:
        entries = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True))
    except Exception as e:
        sys.exit(f"Could not reach the dataset (online? is it public?): {e}")

    count, total = _count_matching(entries, (prefix,))
    if count == 0:
        sys.exit(f"No files found under: {PAPER_HF_PREFIX}/")

    dest_teleop = os.path.join(PAPER_LOCAL_DIR, "teleop_data")
    print(f"\nAbout to download {count} file(s), {human(total)} total, "
          f"for [paper-demos]")
    print(f"  from: https://huggingface.co/datasets/{REPO_ID}/{PAPER_HF_PREFIX}")
    print(f"  into: {dest_teleop}\n")

    resp = input("Proceed with download? [y/N] ").strip().lower()
    if resp not in ("y", "yes"):
        print("Aborted. Nothing was downloaded.")
        return

    os.makedirs(PAPER_LOCAL_DIR, exist_ok=True)
    # snapshot_download preserves HF paths under local_dir, so we get
    # paper_demos/paper_experiments/teleop_data/... then promote teleop_data.
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=PAPER_LOCAL_DIR,
        allow_patterns=[f"{PAPER_HF_PREFIX}/*"],
    )

    staged = os.path.join(PAPER_LOCAL_DIR, "paper_experiments", "teleop_data")
    if not os.path.isdir(staged):
        sys.exit(f"Download finished but expected path missing: {staged}")

    if os.path.exists(dest_teleop):
        shutil.rmtree(dest_teleop)
    shutil.move(staged, dest_teleop)

    # Remove empty paper_experiments/ and any HF cache metadata under it.
    paper_exp = os.path.join(PAPER_LOCAL_DIR, "paper_experiments")
    if os.path.isdir(paper_exp):
        shutil.rmtree(paper_exp)

    print(f"\nDone. Files are in: {dest_teleop}")


def main():
    parser = argparse.ArgumentParser(
        description="Download OopsieVerse teleop demos from Hugging Face.")
    parser.add_argument(
        "--sim", default="both",
        choices=["robocasa", "behavior1k", "both"],
        help="which simulator's data to download (default: both = all); "
             "ignored when --paper-demos is set")
    parser.add_argument(
        "--paper-demos",
        action="store_true",
        help="download paper live-feedback study demos into "
             "oopsiebench/demos/paper_demos/teleop_data "
             "(instead of --sim demos)")
    args = parser.parse_args()

    if args.paper_demos:
        _download_paper_demos()
        return

    sims = list(SIMS) if args.sim == "both" else [args.sim]
    _download_sims(sims)


if __name__ == "__main__":
    main()
