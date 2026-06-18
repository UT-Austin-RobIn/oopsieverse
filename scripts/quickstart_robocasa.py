#!/usr/bin/env python3
"""
Quickstart visualizer for RoboCasa OopsieBench demos.

Interactively browse the teleop demos under
``oopsiebench/demos/robocasa/teleop/``: pick a task, pick safe/unsafe, and play
it back (rendering a health-overlay video). After each playback it shows the
size of the generated playback HDF5 and asks whether to keep or delete it.
Loops until you quit.

Run with the RoboCasa env active:
    conda activate oopsieverse_robocasa
    python scripts/quickstart_robocasa.py
    python scripts/quickstart_robocasa.py --res 720   # higher-quality render

If you don't have the demos locally yet:
    python scripts/download_demos.py --sim robocasa
"""

import argparse
import glob
import os
import re
import subprocess
import sys

# GPU offscreen rendering (EGL is broken on this box; glx works headlessly here).
os.environ.setdefault("MUJOCO_GL", "glx")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEOP_DIR = os.path.join(ROOT, "oopsiebench", "demos", "robocasa", "teleop")
PLAYBACK_DIR = os.path.join(ROOT, "oopsiebench", "demos", "robocasa", "playback")
PLAYBACK_SCRIPT = os.path.join(ROOT, "scripts", "playback_robocasa.py")
VIDEO_ROOT = os.path.join(ROOT, "oopsiebench", "demos", "robocasa", "playback_videos")


def resolve_camera(task):
    """Use the task's registered camera (some robots lack the default agentview)."""
    from oopsiebench.envs.registry import EnvironmentRegistry
    return EnvironmentRegistry.get(task).camera_name


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
    ap = argparse.ArgumentParser(description="Interactive RoboCasa demo visualizer.")
    ap.add_argument("--res", type=int, default=256,
                    help="render resolution NxN (default 256; try 512 or 720 for nicer video)")
    args = ap.parse_args()

    if not glob.glob(os.path.join(TELEOP_DIR, "*.hdf5")):
        sys.exit("No teleop demos found in oopsiebench/demos/robocasa/teleop/.\n"
                 "Get them with:  python scripts/download_demos.py --sim robocasa")

    tasks = discover()
    names = sorted(tasks)

    print("=" * 64)
    print("  RoboCasa demo visualizer   (type 'q' at any prompt to quit)")
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

        # Pick safe / unsafe (skip if only one variant exists).
        if len(variants) == 1:
            variant = next(iter(variants))
        else:
            choice = prompt("  [s]afe or [u]nsafe? ", {"s", "u", "safe", "unsafe"})
            if choice is None:
                continue
            variant = "safe" if choice in ("s", "safe") else "unsafe"

        teleop = variants[variant]
        os.makedirs(PLAYBACK_DIR, exist_ok=True)
        out = os.path.join(PLAYBACK_DIR, f"{task}_{variant}.hdf5")
        camera = resolve_camera(task)

        print(f"\n  Playing back '{task}' ({variant}) at {args.res}x{args.res} "
              f"— this renders a video and may take a bit ...\n")
        cmd = [sys.executable, PLAYBACK_SCRIPT,
               "--input", teleop, "--output", out, "--env", task,
               "--camera", camera, "--width", str(args.res), "--height", str(args.res),
               "--playback", "--visualize", "--metrics"]
        if subprocess.call(cmd) != 0:
            print("\n  Playback failed (see output above). Back to menu.")
            continue

        # Point the user at the video(s).
        vdir = os.path.join(VIDEO_ROOT, f"{task}_{variant}")
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
