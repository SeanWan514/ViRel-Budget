#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/virel_external"
SOURCE="$ROOT/FastV"
TARGET="$ROOT/FastV-random"
EXPECTED_COMMIT="f95102a10acf31416f17ccf21b311c00d14ef49b"
PROJECT_ROOT="${VIREL_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PATCH="$PROJECT_ROOT/patches/fastv_random_pruning.patch"

test -d "$SOURCE/.git"
test -f "$PATCH"
actual_commit="$(git -C "$SOURCE" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "Pinned FastV source mismatch: expected $EXPECTED_COMMIT, got $actual_commit" >&2
  exit 2
fi

if [[ ! -d "$TARGET/.git" ]]; then
  git clone --local "$SOURCE" "$TARGET"
fi
git -C "$TARGET" checkout --detach "$EXPECTED_COMMIT"
git -C "$TARGET" reset --hard "$EXPECTED_COMMIT"
git -C "$TARGET" clean -fd
git -C "$TARGET" apply --check "$PATCH"
git -C "$TARGET" apply "$PATCH"

mkdir -p "$TARGET/virel_patch_metadata"
python3 - "$TARGET/virel_patch_metadata/random_pruning.json" "$PATCH" "$EXPECTED_COMMIT" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

output, patch, commit = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
output.write_text(
    json.dumps(
        {
            "base_commit": commit,
            "patch": str(patch),
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "selection": "uniform without replacement over the 576 visual-token positions",
            "seed": "SHA-256-derived from run seed, sample ID, image path, method, and budget",
            "comparison_boundary": "same FastV mask location, aggregation layer, retained-token count, model, and prompt",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

git -C "$TARGET" diff --check
echo "Prepared isolated real-random FastV tree at $TARGET"
cat "$TARGET/virel_patch_metadata/random_pruning.json"
