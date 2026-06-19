#!/usr/bin/env python3
"""Generate JPG thumbnails for OopsieBench demo MP4s under docs/assets/videos/."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEOS_ROOT = _REPO_ROOT / "docs" / "assets" / "videos"
THUMBS_ROOT = VIDEOS_ROOT / "thumbnails"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--width",
        type=int,
        default=240,
        help="Thumbnail width in pixels (default: 240)",
    )
    p.add_argument(
        "--seek",
        type=float,
        default=1.0,
        help="Seconds into the clip for the frame grab (default: 1.0)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when the JPG already exists",
    )
    return p.parse_args()


def thumb_path_for_mp4(mp4: Path) -> Path:
    rel = mp4.relative_to(VIDEOS_ROOT)
    return THUMBS_ROOT / rel.with_suffix(".jpg")


def generate_thumbnail(mp4: Path, out_jpg: Path, *, width: int, seek: float) -> bool:
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(seek),
        "-i",
        str(mp4),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        "3",
        str(out_jpg),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  failed: {mp4.name} ({exc})", file=sys.stderr)
        return False


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 1

    mp4s = sorted(VIDEOS_ROOT.rglob("*.mp4"))
    if not mp4s:
        print(f"No MP4s under {VIDEOS_ROOT}")
        return 0

    ok = skipped = failed = 0
    for mp4 in mp4s:
        if "thumbnails" in mp4.parts:
            continue
        out = thumb_path_for_mp4(mp4)
        if out.is_file() and not args.force:
            skipped += 1
            continue
        print(f"  {mp4.relative_to(_REPO_ROOT)} -> {out.relative_to(_REPO_ROOT)}")
        if generate_thumbnail(mp4, out, width=args.width, seek=args.seek):
            ok += 1
        else:
            failed += 1

    print(f"Done: {ok} generated, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
