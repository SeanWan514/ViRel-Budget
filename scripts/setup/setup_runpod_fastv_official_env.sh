#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-/workspace/virel_envs/fastv_official}"
EXTERNAL_DIR="${2:-/workspace/virel_external}"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 is required for FastV's tokenizers/transformers dependency band." >&2
  exit 3
fi

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepare_external_pruning_baselines.sh" "$EXTERNAL_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python3.10 -m venv "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  "numpy<2" pandas pillow "accelerate==0.21.0" shortuuid "sentencepiece==0.1.99" \
  protobuf "einops==0.6.1" "einops-exts==0.0.4" "timm==0.6.13" \
  requests pyyaml regex safetensors codecarbon
python -m pip install -e "$EXTERNAL_DIR/FastV/src/transformers"
python -m pip install -e "$EXTERNAL_DIR/FastV/src/LLaVA" --no-deps

python - <<'PY'
import sys
import torch
import tokenizers
import transformers
import codecarbon
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

print("python", sys.version)
print("torch", torch.__version__, torch.cuda.is_available())
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("codecarbon", getattr(codecarbon, "__version__", "unknown"))
print("llava official source imports ok")
PY
