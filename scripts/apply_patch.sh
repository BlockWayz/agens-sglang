#!/usr/bin/env bash
# Check out upstream SGLang at the pinned commit (SGLANG_COMMIT) and apply the Agens patch.
#
#   scripts/apply_patch.sh [DEST_DIR]        # default: ./sglang
#
# Then install it into an environment that matches lmsysorg/sglang:v0.5.16 (torch 2.11, CUDA 12.9,
# sglang-kernel 0.4.5, FlashInfer 0.6.x) with a Rust toolchain on PATH:
#   pip install -e DEST_DIR/python
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dest="${1:-sglang}"
commit="$(tr -d '[:space:]' < "$here/SGLANG_COMMIT")"
repo="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"

if [ -e "$dest/.git" ]; then
  echo "error: $dest is already a git checkout" >&2
  exit 1
fi
git init -q "$dest"
cd "$dest"
git remote add origin "$repo"
git fetch -q --depth 1 origin "$commit"
git checkout -q FETCH_HEAD
git apply --whitespace=nowarn "$here/patches/agens-volundr.patch"
echo "SGLang $commit + agens-volundr.patch -> $(pwd)"
