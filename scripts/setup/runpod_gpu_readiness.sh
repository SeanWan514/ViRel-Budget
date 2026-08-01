#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[virel-gpu] Repo: $PWD"
echo "[virel-gpu] Python:"
python3 --version

echo "[virel-gpu] CUDA/Torch:"
python3 - <<'PY'
import json
try:
    import torch
    print(json.dumps({
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))
except Exception as exc:
    print(json.dumps({"torch_error": f"{type(exc).__name__}: {exc}"}, indent=2))
PY

echo "[virel-gpu] nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,power.draw,power.limit --format=csv,noheader,nounits

echo "[virel-gpu] Disk:"
df -h /workspace .

echo "[virel-gpu] Source hash:"
find virel_budget configs scripts tests -type f \( -name '*.py' -o -name '*.json' -o -name '*.sh' \) -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sha256sum
