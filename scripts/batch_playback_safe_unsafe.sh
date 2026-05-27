#!/usr/bin/env bash
# Replay all teleop HDF5s under teleop_demos/{safe,unsafe}/ at 256×256.
# Deletes any existing *_playback.hdf5 in the same folder before each run.
#
# Usage (from repo root or anywhere):
#   bash scripts/batch_playback_safe_unsafe.sh
#
# Override demo root:
#   TELEOP_DEMOS_ROOT=/path/to/teleop_demos bash scripts/batch_playback_safe_unsafe.sh
#
# Requires conda env: oopsieverse

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEMOS_ROOT="${TELEOP_DEMOS_ROOT:-${REPO_ROOT}/teleop_demos}"

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found on PATH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate oopsieverse

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

FAILED=()

run_playback() {
  local split="$1"
  local teleop_path="$2"
  local task_name
  task_name="$(basename "$teleop_path" .hdf5)"

  if [[ "$task_name" == *_playback ]]; then
    return 0
  fi

  if [[ ! -f "$teleop_path" ]]; then
    echo "[${split}/${task_name}] SKIP: teleop file missing: ${teleop_path}" >&2
    FAILED+=("${split}/${task_name} (missing teleop)")
    return 0
  fi

  local playback_path="${teleop_path%.hdf5}_playback.hdf5"
  local label="${split}/${task_name}"

  echo ""
  echo "================================================================================"
  echo "[${label}] teleop:    ${teleop_path}"
  echo "[${label}] playback:  ${playback_path}"
  echo "================================================================================"

  if [[ -f "$playback_path" ]]; then
    echo "[${label}] removing existing playback file"
    rm -f "$playback_path"
  fi

  if ! python scripts/playback_b1k.py \
    --task_name "$task_name" \
    --playback \
    --low_resolution \
    --collect_hdf5_path "$teleop_path" \
    --playback_hdf5_path "$playback_path"; then
    echo "[${label}] FAILED" >&2
    FAILED+=("$label")
    return 0
  fi

  echo "[${label}] OK"
}

echo "Demo root: ${DEMOS_ROOT}"

for split in safe unsafe; do
  dir="${DEMOS_ROOT}/${split}"
  if [[ ! -d "$dir" ]]; then
    echo "WARN: missing directory ${dir}, skipping" >&2
    continue
  fi

  shopt -s nullglob
  teleop_files=("${dir}"/*.hdf5)
  shopt -u nullglob

  if [[ ${#teleop_files[@]} -eq 0 ]]; then
    echo "WARN: no .hdf5 files in ${dir}" >&2
    continue
  fi

  for teleop_path in "${teleop_files[@]}"; do
    run_playback "$split" "$teleop_path"
  done
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Finished with failures (${#FAILED[@]}):"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi

echo "All playback runs completed successfully."
