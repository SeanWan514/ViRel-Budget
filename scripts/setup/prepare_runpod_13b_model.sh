#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${VIREL_FASTV_PYTHON:-/workspace/virel_envs/fastv_official/bin/python}"
MODEL_ID="liuhaotian/llava-v1.5-13b"
MODEL_DIR="/workspace/models/llava-v1.5-13b"
MIN_FREE_GB=35

test -x "$PYTHON_BIN"
mkdir -p /workspace/models
free_gb="$(df -Pk /workspace | awk 'NR==2 {print int($4/1024/1024)}')"
if (( free_gb < MIN_FREE_GB )); then
  echo "Need at least ${MIN_FREE_GB} GiB free on /workspace before downloading 13B; found ${free_gb} GiB." >&2
  exit 2
fi

"$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

model_id, model_dir = sys.argv[1], Path(sys.argv[2])
info = HfApi().model_info(model_id)
snapshot_download(
    repo_id=model_id,
    revision=info.sha,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
required = ["config.json", "tokenizer.model"]
missing = [name for name in required if not (model_dir / name).exists()]
shards = sorted(model_dir.glob("pytorch_model-*.bin"))
if missing or not shards:
    raise SystemExit(f"Incomplete model snapshot: missing={missing}, shards={len(shards)}")
files = []
total = 0
for path in sorted(value for value in model_dir.rglob("*") if value.is_file()):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    size = path.stat().st_size
    total += size
    files.append({"path": str(path.relative_to(model_dir)), "bytes": size, "sha256": digest})
manifest = {
    "model_id": model_id,
    "revision": info.sha,
    "local_dir": str(model_dir),
    "total_bytes": total,
    "file_count": len(files),
    "files": files,
}
(model_dir / "virel_model_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({key: manifest[key] for key in ("model_id", "revision", "total_bytes", "file_count")}, indent=2))
PY

echo "LLaVA-1.5-13B snapshot is pinned and ready at $MODEL_DIR"
