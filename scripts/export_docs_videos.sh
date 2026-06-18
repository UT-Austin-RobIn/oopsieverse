#!/usr/bin/env bash
# Render health-overlay videos from playback HDF5s and copy them into the docs
# folder using the <task>_<split>.mp4 naming convention.
#
# For each task and split (safe/unsafe) it:
#   1. runs `playback_b1k.py --visualize` on teleop_demos/<split>/<task>_playback.hdf5
#      into a per-(task,split) video dir (so safe/unsafe don't overwrite each other),
#   2. copies the resulting demo_0_health_overlay_video.mp4 to
#      docs/assets/videos/behavior1k/<task>_<split>.mp4
#
# Usage (from repo root or anywhere):
#   bash scripts/export_docs_videos.sh                       # default task list below
#   bash scripts/export_docs_videos.sh nav_to_table pick_egg # explicit task list
#
# Requires conda env: oopsieverse

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Tasks to export: CLI args override the default list.
if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=(nav_to_table pick_egg open_drawer pour_water wipe_counter)
fi

DEMOS_ROOT="${TELEOP_DEMOS_ROOT:-${REPO_ROOT}/teleop_demos}"
VIDEO_ROOT="${REPO_ROOT}/demos/behavior1k/playback_videos"
DOCS_DIR="${REPO_ROOT}/docs/assets/videos/behavior1k"
OVERLAY_NAME="demo_0_health_overlay_video.mp4"   # 1 demo per playback file

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found on PATH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate oopsieverse

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$DOCS_DIR"

FAILED=()

export_one() {
  local task="$1"
  local split="$2"
  local playback_path="${DEMOS_ROOT}/${split}/${task}_playback.hdf5"
  local video_dir="${VIDEO_ROOT}/${task}_${split}"
  local label="${task}/${split}"

  echo ""
  echo "================================================================================"
  echo "[${label}] playback: ${playback_path}"
  echo "================================================================================"

  if [[ ! -f "$playback_path" ]]; then
    echo "[${label}] SKIP: playback file missing" >&2
    FAILED+=("${label} (missing playback)")
    return 0
  fi

  if ! python scripts/playback_b1k.py \
      --task_name "$task" \
      --playback_hdf5_path "$playback_path" \
      --visualize \
      --video_dir "$video_dir"; then
    echo "[${label}] FAILED: visualize" >&2
    FAILED+=("${label} (visualize)")
    return 0
  fi

  local src="${video_dir}/${OVERLAY_NAME}"
  local dst="${DOCS_DIR}/${task}_${split}.mp4"
  if [[ ! -f "$src" ]]; then
    echo "[${label}] FAILED: expected overlay video not found at ${src}" >&2
    FAILED+=("${label} (no overlay video)")
    return 0
  fi

  cp "$src" "$dst"
  echo "[${label}] OK -> ${dst}"
}

for task in "${TASKS[@]}"; do
  for split in safe unsafe; do
    export_one "$task" "$split"
  done
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Finished with issues (${#FAILED[@]}):"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi

echo "All videos exported to ${DOCS_DIR}"
echo "Reminder: these .mp4s are Git LFS-tracked — run 'git add' then 'git lfs status' to confirm."
