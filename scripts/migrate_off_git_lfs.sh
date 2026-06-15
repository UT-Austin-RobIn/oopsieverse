#!/usr/bin/env bash
# Migrate oopsieverse off Git LFS:
#   - docs/assets/videos/*.mp4  -> regular git blobs (~46 MB total)
#   - oopsiebench/test_data/*.hdf5 -> removed from git (download from Hugging Face)
#
# Run from repo root after committing .gitattributes / .gitignore updates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> 1/4 Unblock pulls while LFS is over budget (one-time, any machine):"
echo "    GIT_LFS_SKIP_SMUDGE=1 git pull"
echo

echo "==> 2/4 Stop tracking HDF5 in git (files stay on disk; ignored via .gitignore)"
if git ls-files 'oopsiebench/test_data/**/*.hdf5' | grep -q .; then
  git rm -r --cached oopsiebench/test_data/behavior1k oopsiebench/test_data/robocasa 2>/dev/null || true
  git rm --cached oopsiebench/test_data/**/*.hdf5 2>/dev/null || true
fi

echo "==> 3/4 Re-store doc videos as regular git blobs (requires real MP4 files on disk)"
missing=0
while IFS= read -r f; do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f"
    missing=1
  elif head -c 8 "$f" | grep -q 'version https://git-lfs.github.com'; then
    echo "  LFS pointer (need real file): $f"
    missing=1
  fi
done < <(git ls-files 'docs/assets/videos/**/*.mp4')

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Fix missing/pointer MP4s before committing (copy from a machine with LFS cache,"
  echo "re-render, or restore from backup). nav_to_table_* may need to be re-added."
  exit 1
fi

git add --renormalize docs/assets/videos/

echo "==> 4/4 Stage config and review"
git add .gitattributes .gitignore scripts/download_test_data.py scripts/migrate_off_git_lfs.sh
git status

cat <<'EOF'

Next steps:
  1. Upload oopsiebench/test_data/**/*.hdf5 to Hugging Face (dataset repo).
  2. Set OOPSIEVERSE_HF_DATASET if the repo id differs from UT-Austin-RobIn/oopsieverse-test-data
  3. Commit: git commit -m "Drop Git LFS; store doc videos in git, HDF5 on Hugging Face"
  4. Push to origin/main

Optional (rewrites history — coordinate with collaborators):
  git lfs migrate export --include="docs/assets/videos/**" --everything
  git push --force-with-lease

After push, clones no longer need Git LFS:
  git clone https://github.com/UT-Austin-RobIn/oopsieverse.git
  python scripts/download_test_data.py   # for test HDF5
EOF
