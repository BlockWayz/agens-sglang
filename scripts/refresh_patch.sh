#!/usr/bin/env bash
# Regenerate patches/agens-volundr.patch from a working tree created by apply_patch.sh (after
# editing it). Only python/ is part of the patch.
#
#   scripts/refresh_patch.sh SGLANG_DIR
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
src="${1:?usage: refresh_patch.sh SGLANG_DIR}"
commit="$(tr -d '[:space:]' < "$here/SGLANG_COMMIT")"
cd "$src"
find python -name '__pycache__' -type d -prune -exec rm -rf {} +
git add -N python
git diff "$commit" -- python > "$here/patches/agens-volundr.patch"
git diff --stat "$commit" -- python | tail -1
