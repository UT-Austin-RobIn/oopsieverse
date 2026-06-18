#!/usr/bin/env python3
"""
Quickstart visualizer for BEHAVIOR-1K (OmniGibson) OopsieBench demos.

Interactively browse the teleop demos under
``oopsiebench/demos/behavior1k/teleop/``: pick a task, pick a variant
(safe/unsafe when both exist), and play it back (rendering a health-overlay
video). After each playback it shows the size of the generated playback HDF5
and asks whether to keep or delete it. Loops until you quit.

Run with the BEHAVIOR-1K env active (needs an NVIDIA GPU + OmniGibson):
    conda activate oopsieverse_b1k
    python scripts/quickstart_b1k.py
    python scripts/quickstart_b1k.py --hires    # full-res render (default is 256x256)

If you don't have the demos locally yet:
    python scripts/download_demos.py --sim behavior1k
"""

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEOP_DIR = os.path.join(ROOT, "oopsiebench", "demos", "behavior1k", "teleop")
PLAYBACK_DIR = os.path.join(ROOT, "oopsiebench", "demos", "behavior1k", "playback")
PLAYBACK_SCRIPT = os.path.join(ROOT, "scripts", "playback_b1k.py")
VIDEO_ROOT = os.path.join(ROOT, "oopsiebench", "demos", "behavior1k", "playback_videos")


def human(nbytes):
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def discover():
    """Map {task: {variant: filepath}} from the teleop HDF5 filenames."""
    tasks = {}
    for fp in sorted(glob.glob(os.path.join(TELEOP_DIR, "*.hdf5"))):
        name = os.path.basename(fp)[:-5]
        m = re.match(r"^(.*)_(safe|unsafe)$", name)
        task, variant = (m.group(1), m.group(2)) if m else (name, "demo")
        tasks.setdefault(task, {})[variant] = fp
    return tasks


def prompt(msg, valid):
    """Prompt until a valid (lowercased) answer or quit; returns None on quit."""
    while True:
        r = input(msg).strip().lower()
        if r in ("q", "quit", "exit"):
            return None
        if r in valid:
            return r
        print("  invalid choice — try again")


def main():
    ap = argparse.ArgumentParser(description="Interactive BEHAVIOR-1K demo visualizer.")
    ap.add_argument("--hires", action="store_true",
                    help="render at full resolution (default is fast 256x256)")
    args = ap.parse_args()

    if not glob.glob(os.path.join(TELEOP_DIR, "*.hdf5")):
        sys.exit("No teleop demos found in oopsiebench/demos/behavior1k/teleop/.\n"
                 "Get them with:  python scripts/download_demos.py --sim behavior1k")

    tasks = discover()
    names = sorted(tasks)

    print("=" * 64)
    print("  BEHAVIOR-1K demo visualizer   (type 'q' at any prompt to quit)")
    print("=" * 64)

    while True:
        print("\nAvailable tasks:")
        for i, t in enumerate(names, 1):
            print(f"  {i:2d}. {t}  ({'/'.join(sorted(tasks[t]))})")

        sel = input("\nPick a task number (or 'q' to quit): ").strip().lower()
        if sel in ("q", "quit", "exit", ""):
            break
        if not sel.isdigit() or not (1 <= int(sel) <= len(names)):
            print("  invalid selection")
            continue
        task = names[int(sel) - 1]
        variants = tasks[task]

        # Pick a variant (skip if only one exists — many b1k tasks are single-file).
        if len(variants) == 1:
            variant = next(iter(variants))
        else:
            choice = prompt("  [s]afe or [u]nsafe? ", {"s", "u", "safe", "unsafe"})
            if choice is None:
                continue
            variant = "safe" if choice in ("s", "safe") else "unsafe"

        teleop = variants[variant]
        os.makedirs(PLAYBACK_DIR, exist_ok=True)
        tag = task if variant == "demo" else f"{task}_{variant}"
        out = os.path.join(PLAYBACK_DIR, f"{tag}.hdf5")
        vdir = os.path.join(VIDEO_ROOT, tag)

        print(f"\n  Playing back '{task}' ({variant}) "
              f"{'(full-res)' if args.hires else '(256x256)'} "
              f"— this renders a video and may take a bit ...\n")
        cmd = [sys.executable, PLAYBACK_SCRIPT,
               "--task_name", task,
               "--source_hdf5_path", teleop,
               "--playback_hdf5_path", out,
               "--video_dir", vdir,
               "--playback", "--visualize", "--compute_metrics"]
        if not args.hires:
            cmd.append("--low_resolution")
        if subprocess.call(cmd) != 0:
            print("\n  Playback failed (see output above). Back to menu.")
            continue

        # Point the user at the video(s).
        vids = sorted(glob.glob(os.path.join(vdir, "*health_overlay*.mp4")))
        if vids:
            print("\n  Health-overlay video(s):")
            for v in vids:
                print(f"    {v}")

        # Keep or delete the (potentially large) playback HDF5.
        if os.path.exists(out):
            print(f"\n  Generated playback file: {out}")
            print(f"  Size: {human(os.path.getsize(out))}")
            keep = prompt("  Keep this playback HDF5? [y/n]  (n = delete it): ",
                          {"y", "n", "yes", "no"})
            if keep in ("n", "no"):
                os.remove(out)
                print("  Deleted.")
            else:
                print("  Kept.")

    print("\nDone. Bye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone. Bye!")
