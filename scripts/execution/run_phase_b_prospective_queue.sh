#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
QUEUE_DIR="results/phase_b_queue"
CONTROLLER_ROOT="results/frozen_controllers_phase_a"
DEPLOYMENT="data/prospective_900_frozen/deployment_samples.jsonl"
mkdir -p "$QUEUE_DIR" "$CONTROLLER_ROOT"

if [[ ! -f results/phase_b_expansion_approval.json ]]; then
  echo "Phase-B approval is absent; refusing prospective execution." >&2
  exit 3
fi

python3 - <<'PY'
import json
approval = json.load(open("results/phase_b_expansion_approval.json", encoding="utf-8"))
if approval.get("phase") != "B_expansion" or approval.get("approved") is not True:
    raise SystemExit("Invalid Phase-B approval artifact")
if approval.get("prospective_sample_count") != 900:
    raise SystemExit("Approval is not for the exact 900-case prospective set")
PY

python_for_method() {
  case "$1" in
    scope) printf '%s\n' "${VIREL_SCOPE_PYTHON:-/workspace/virel_envs/scope_official/bin/python}" ;;
    fastv|random) printf '%s\n' "${VIREL_FASTV_PYTHON:-/workspace/virel_envs/fastv_official/bin/python}" ;;
    *) echo "Unknown method: $1" >&2; return 2 ;;
  esac
}

fit_controller() {
  local model="$1"
  local method="$2"
  local out="$CONTROLLER_ROOT/${model}_${method}"
  local artifact="$out/${method}_controller.json"
  local checksum="$out/${method}_controller.sha256"
  if [[ -f "$artifact" && -f "$checksum" ]]; then
    (cd "$out" && sha256sum -c "$(basename "$checksum")")
    return
  fi
  mkdir -p "$out"
  "$(python_for_method "$method")" scripts/fit_frozen_controller.py \
    --samples data/paper_1200/samples.jsonl \
    --labels "results/phase_a_current_llava15_${model}_${method}/strict_labels/safe_budget_labels.csv" \
    --output-dir "$out" \
    --methods "$method" \
    > "$QUEUE_DIR/fit_${model}_${method}.log" 2>&1
  (cd "$out" && sha256sum -c "$(basename "$checksum")")
}

run_controller() {
  local model="$1"
  local method="$2"
  local config="configs/prospective900_llava15_${model}_${method}.json"
  local controller_dir="$CONTROLLER_ROOT/${model}_${method}"
  local out="results/prospective900_execution_llava15_${model}_${method}"
  local manifest="$out/execution_manifest.json"
  if [[ -f "$manifest" ]] && python3 - "$manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if m.get("executed_count") == 900 and m.get("maximum_backend_invocations_per_query") == 1 else 1)
PY
  then
    echo "Skipping complete controller execution: $model/$method"
  else
    local attempt
    attempt="$(python3 - "$out" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]) / "telemetry_attempts"
root.mkdir(parents=True, exist_ok=True)
nums = [int(p.name.split("_")[-1]) for p in root.glob("attempt_*") if p.name.split("_")[-1].isdigit()]
print(max(nums, default=0) + 1)
PY
)"
    local monitor="$out/telemetry_attempts/attempt_$(printf '%03d' "$attempt")"
    local py
    py="$(python_for_method "$method")"
    local -a backend_args=()
    if [[ "$method" == "scope" ]]; then
      backend_args+=(--scope-batched-order)
    fi
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$QUEUE_DIR/controller_${model}_${method}_started_utc.txt"
    set +e
    CODECARBON_PYTHON="$py" bash scripts/run_with_codecarbon.sh \
      "$monitor/codecarbon" "prospective_controller_${model}_${method}_attempt_${attempt}" \
      bash scripts/run_with_nvidia_monitor.sh "$monitor" \
        "$py" scripts/run_frozen_controller.py \
          --config "$config" \
          --controller "$controller_dir/${method}_controller.json" \
          --checksum "$controller_dir/${method}_controller.sha256" \
          --deployment-samples "$DEPLOYMENT" \
          --output-dir "$out" \
          --warmup-samples "$DEPLOYMENT" \
          --warmup-count 3 \
          --resume \
          "${backend_args[@]}" \
      2>&1 | tee "$QUEUE_DIR/controller_${model}_${method}.log"
    local status="${PIPESTATUS[0]}"
    set -e
    if [[ "$status" -ne 0 ]]; then
      echo "Controller attempt failed with status $status: $model/$method" >&2
      return "$status"
    fi
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$QUEUE_DIR/controller_${model}_${method}_completed_utc.txt"
  fi
  local py
  py="$(python_for_method "$method")"
  "$py" scripts/merge_telemetry_attempts.py \
    --attempt-root "$out/telemetry_attempts" \
    --output "$out/merged_telemetry/nvidia_smi_samples.csv"
  "$py" scripts/analyze_greenmm_telemetry.py \
    --telemetry "$out/merged_telemetry/nvidia_smi_samples.csv" \
    --execution-manifest "$out/execution_manifest.json" \
    --executed-actions "$out/executed_actions.jsonl" \
    --output "$out/greenmm_telemetry.json"
}

run_controller_with_progress_retries() {
  local model="$1"
  local method="$2"
  local out="results/prospective900_execution_llava15_${model}_${method}/executed_actions.jsonl"
  local before after
  for retry in 1 2 3 4 5 6; do
    before=0
    [[ -f "$out" ]] && before="$(wc -l < "$out")"
    if run_controller "$model" "$method"; then
      return 0
    fi
    after=0
    [[ -f "$out" ]] && after="$(wc -l < "$out")"
    if [[ "$after" -le "$before" ]]; then
      echo "Refusing retry without durable progress: $model/$method ($before -> $after)" >&2
      return 1
    fi
    echo "Restarting from durable controller progress: $model/$method ($before -> $after)" >&2
  done
  echo "Controller exceeded six progress-making attempts: $model/$method" >&2
  return 1
}

for model_method in \
  "7b fastv" \
  "13b scope" \
  "7b random" \
  "13b fastv" \
  "7b scope" \
  "13b random"
do
  read -r model method <<< "$model_method"
  fit_controller "$model" "$method"
done

# All decisions are planned and executed while prospective labels remain sealed.
for model_method in \
  "7b fastv" \
  "13b scope" \
  "7b random" \
  "13b fastv" \
  "7b scope" \
  "13b random"
do
  read -r model method <<< "$model_method"
  run_controller_with_progress_retries "$model" "$method"
done

if [[ ! -f data/prospective_900_revealed/reveal_manifest.json ]]; then
  /workspace/virel_envs/fastv_official/bin/python scripts/reveal_prospective_labels.py \
    --execution-dir results/prospective900_execution_llava15_7b_fastv \
    --execution-dir results/prospective900_execution_llava15_7b_scope \
    --execution-dir results/prospective900_execution_llava15_7b_random \
    --execution-dir results/prospective900_execution_llava15_13b_fastv \
    --execution-dir results/prospective900_execution_llava15_13b_scope \
    --execution-dir results/prospective900_execution_llava15_13b_random \
    > "$QUEUE_DIR/reveal.log" 2>&1
fi

run_grid() {
  local model="$1"
  local method="$2"
  local label="$3"
  local config="configs/prospective900_llava15_${model}_${method}.json"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$QUEUE_DIR/${label}_started_utc.txt"
  bash scripts/run_phase_a_cell.sh "$config" 2>&1 | tee "$QUEUE_DIR/${label}.log"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$QUEUE_DIR/${label}_completed_utc.txt"
}

reuse_interventions() {
  local target="$1"
  if [[ ! -d "$target/interventions" ]]; then
    mkdir -p "$target"
    cp -al results/prospective900_llava15_7b_fastv/interventions "$target/interventions"
  fi
}

run_grid 7b fastv 01_grid_7b_fastv
reuse_interventions results/prospective900_llava15_13b_scope
run_grid 13b scope 02_grid_13b_scope

reuse_interventions results/prospective900_llava15_7b_random
if [[ ! -s results/prospective900_llava15_7b_random/records_checkpoint.jsonl ]]; then
  /workspace/virel_envs/fastv_official/bin/python scripts/seed_dense_checkpoint.py \
    --source-records results/prospective900_llava15_7b_fastv/records_with_measured_energy.jsonl \
    --target-output-dir results/prospective900_llava15_7b_random \
    --expected-dense-records 3000
fi
run_grid 7b random 03_grid_7b_random

reuse_interventions results/prospective900_llava15_13b_fastv
run_grid 13b fastv 04_grid_13b_fastv
reuse_interventions results/prospective900_llava15_7b_scope
run_grid 7b scope 05_grid_7b_scope

reuse_interventions results/prospective900_llava15_13b_random
if [[ ! -s results/prospective900_llava15_13b_random/records_checkpoint.jsonl ]]; then
  /workspace/virel_envs/fastv_official/bin/python scripts/seed_dense_checkpoint.py \
    --source-records results/prospective900_llava15_13b_fastv/records_with_measured_energy.jsonl \
    --target-output-dir results/prospective900_llava15_13b_random \
    --expected-dense-records 3000
fi
run_grid 13b random 06_grid_13b_random

python3 scripts/audit_phase_b_900.py \
  --write-report "$QUEUE_DIR/final_phase_b_audit.json" \
  --require-complete
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$QUEUE_DIR/queue_completed_utc.txt"
