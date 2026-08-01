#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_DIR="${1:-/workspace/virel_external}"
mkdir -p "$EXTERNAL_DIR"
cd "$EXTERNAL_DIR"

clone_or_update() {
  local name="$1"
  local url="$2"
  if [[ -d "$name/.git" ]]; then
    git -C "$name" pull --ff-only || true
  else
    git clone --depth 1 "$url" "$name"
  fi
}

clone_or_update FastV https://github.com/chenllliang/FastV
clone_or_update SCOPE https://github.com/kinredon/SCOPE
clone_or_update LLaVA-PruMerge https://github.com/42Shawn/LLaVA-PruMerge

echo "External pruning repositories are available under $EXTERNAL_DIR"
