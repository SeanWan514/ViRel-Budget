#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
QUEUE_DIR="results/phase_a_current_queue"
mkdir -p "$QUEUE_DIR"

run_cell() {
  local config="$1"
  local label="$2"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$QUEUE_DIR/${label}_started_utc.txt"
  bash scripts/run_phase_a_cell.sh "$config" 2>&1 | tee "$QUEUE_DIR/${label}.log"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$QUEUE_DIR/${label}_completed_utc.txt"
}

# Fixed, predeclared interleaving reduces simple model/method run-order
# confounding. Dense reuse is only within the identical FastV environment.
run_cell configs/phase_a_current_llava15_7b_fastv.json 01_7b_fastv
run_cell configs/phase_a_current_llava15_13b_scope.json 02_13b_scope

/workspace/virel_envs/fastv_official/bin/python scripts/seed_dense_checkpoint.py \
  --source-records results/phase_a_current_llava15_7b_fastv/records_with_measured_energy.jsonl \
  --target-output-dir results/phase_a_current_llava15_7b_random
run_cell configs/phase_a_current_llava15_7b_random.json 03_7b_random

run_cell configs/phase_a_current_llava15_13b_fastv.json 04_13b_fastv
run_cell configs/phase_a_current_llava15_7b_scope.json 05_7b_scope

/workspace/virel_envs/fastv_official/bin/python scripts/seed_dense_checkpoint.py \
  --source-records results/phase_a_current_llava15_13b_fastv/records_with_measured_energy.jsonl \
  --target-output-dir results/phase_a_current_llava15_13b_random
run_cell configs/phase_a_current_llava15_13b_random.json 06_13b_random

python3 scripts/audit_phase_a_1200.py \
  --write-report results/phase_a_current_queue/final_phase_a_audit.json \
  --require-complete
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$QUEUE_DIR/queue_completed_utc.txt"
