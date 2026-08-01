#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-/workspace/virel_envs/scope_official}"
EXTERNAL_DIR="${2:-/workspace/virel_external}"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 is required for SCOPE's tokenizers/transformers dependency band." >&2
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
  "numpy<2" pandas pillow "accelerate==0.34.2" shortuuid "sentencepiece==0.1.99" \
  protobuf "einops==0.6.1" "einops-exts==0.0.4" "timm==0.6.13" \
  requests pyyaml regex safetensors "tokenizers==0.15.1" "transformers==4.37.2" "matplotlib==3.9.2" codecarbon
python -m pip install -e "$EXTERNAL_DIR/SCOPE/LLaVA" --no-deps

python - "$EXTERNAL_DIR" <<'PY'
import sys
import types
from pathlib import Path

external_dir = Path(sys.argv[1])
sys.path.insert(0, str(external_dir / "SCOPE"))
package = types.ModuleType("llava")
package.__path__ = [str(external_dir / "SCOPE" / "LLaVA" / "llava")]
package.__file__ = str(external_dir / "SCOPE" / "LLaVA" / "llava" / "__init__.py")
sys.modules.setdefault("llava", package)
sys.modules.setdefault("submodular_function", types.ModuleType("submodular_function"))
sys.modules.setdefault("submodular_optimizer", types.ModuleType("submodular_optimizer"))

import torch
import tokenizers
import transformers
import codecarbon
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from scope import SCOPE

print("python", sys.version)
print("torch", torch.__version__, torch.cuda.is_available())
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("codecarbon", getattr(codecarbon, "__version__", "unknown"))
print("llava and scope official sources import ok")
PY
