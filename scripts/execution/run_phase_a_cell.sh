#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_phase_a_cell.sh <phase-A-config.json>" >&2
  exit 2
fi

CONFIG="$1"
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 3
fi

readarray -t CONFIG_VALUES < <(python3 - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
print(cfg["backend"]["name"])
print(cfg["outputs"]["dir"])
print(",".join(cfg["pruning"]["methods"]))
print(cfg["backend"]["model_path"])
PY
)
BACKEND="${CONFIG_VALUES[0]}"
OUT_DIR="${CONFIG_VALUES[1]}"
METHODS="${CONFIG_VALUES[2]}"
MODEL_PATH="${CONFIG_VALUES[3]}"

case "$BACKEND" in
  fastv_llava)
    RUN_PYTHON="${VIREL_FASTV_PYTHON:-/workspace/virel_envs/fastv_official/bin/python}"
    ;;
  scope_llava)
    RUN_PYTHON="${VIREL_SCOPE_PYTHON:-/workspace/virel_envs/scope_official/bin/python}"
    ;;
  *)
    echo "Unsupported Phase-A backend: $BACKEND" >&2
    exit 4
    ;;
esac

if [[ ! -x "$RUN_PYTHON" ]]; then
  echo "Missing backend environment Python: $RUN_PYTHON" >&2
  exit 5
fi
if [[ "$MODEL_PATH" == /workspace/* && ! -d "$MODEL_PATH" ]]; then
  echo "Missing locally pinned model: $MODEL_PATH" >&2
  exit 6
fi
if [[ "$METHODS" == *random* && ! -d /workspace/virel_external/FastV-random ]]; then
  echo "Random-pruning patch is not installed; run prepare_runpod_random_pruning.sh." >&2
  exit 7
fi

ATTEMPT="$(python3 - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]) / "telemetry_attempts"
root.mkdir(parents=True, exist_ok=True)
nums = [int(p.name.split("_")[-1]) for p in root.glob("attempt_*") if p.name.split("_")[-1].isdigit()]
print(max(nums, default=0) + 1)
PY
)"
MONITOR_DIR="$OUT_DIR/telemetry_attempts/attempt_$(printf '%03d' "$ATTEMPT")"
CODECARBON_DIR="$MONITOR_DIR/codecarbon"

CODECARBON_PYTHON="$RUN_PYTHON" bash scripts/run_with_codecarbon.sh \
  "$CODECARBON_DIR" "$(basename "$OUT_DIR")_attempt_$(printf '%03d' "$ATTEMPT")" \
  bash scripts/run_with_nvidia_monitor.sh "$MONITOR_DIR" \
    "$RUN_PYTHON" scripts/run_resumable_grid.py --config "$CONFIG" --analyze

"$RUN_PYTHON" scripts/merge_telemetry_attempts.py \
  --attempt-root "$OUT_DIR/telemetry_attempts" \
  --output "$OUT_DIR/merged_telemetry/nvidia_smi_samples.csv"

"$RUN_PYTHON" scripts/attribute_nvidia_energy.py \
  --records "$OUT_DIR/records.jsonl" \
  --samples "$OUT_DIR/merged_telemetry/nvidia_smi_samples.csv" \
  --out-json "$OUT_DIR/measured_energy_attribution.json" \
  --out-csv "$OUT_DIR/measured_energy_by_method_budget.csv" \
  --out-records-jsonl "$OUT_DIR/records_with_measured_energy.jsonl"

"$RUN_PYTHON" -m virel_budget.cli analyze-existing \
  --config "$CONFIG" \
  --records "$OUT_DIR/records_with_measured_energy.jsonl" \
  --output-dir "$OUT_DIR/measured_reanalysis" \
  --run-name "$(basename "$OUT_DIR")_measured_reanalysis"

"$RUN_PYTHON" scripts/build_strict_labels_from_run.py \
  --config "$CONFIG" \
  --records "$OUT_DIR/records_with_measured_energy.jsonl" \
  --output-dir "$OUT_DIR/strict_labels"

python3 scripts/audit_phase_a_1200.py --write-report results/phase_a_1200_status.json
