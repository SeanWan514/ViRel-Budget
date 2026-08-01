#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-/workspace/virel_envs/smolvlm_dense}"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 is required for a reproducible RunPod SmolVLM environment." >&2
  exit 3
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python3.10 -m venv "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  "numpy<2" pandas pillow "transformers>=4.50" "accelerate>=0.34" \
  safetensors sentencepiece protobuf requests pyyaml regex codecarbon
python -m pip install -e .

python - <<'PY'
import sys
import torch
import transformers
import codecarbon

print("python", sys.version)
print("torch", torch.__version__, torch.cuda.is_available())
print("transformers", transformers.__version__)
print("codecarbon", getattr(codecarbon, "__version__", "unknown"))
print("smolvlm dense environment imports ok")
PY
