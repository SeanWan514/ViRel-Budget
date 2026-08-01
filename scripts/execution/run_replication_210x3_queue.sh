#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
QUEUE="results/replication210x3_queue"
mkdir -p "$QUEUE"
touch "$QUEUE/queue.log" "$QUEUE/status.jsonl"

FASTV_PY="${VIREL_FASTV_PYTHON:-/workspace/virel_envs/fastv_official/bin/python}"
SCOPE_PY="${VIREL_SCOPE_PYTHON:-/workspace/virel_envs/scope_official/bin/python}"

python_for_system() {
  case "$1" in
    scope) printf '%s\n' "$SCOPE_PY" ;;
    dense|fastv|random) printf '%s\n' "$FASTV_PY" ;;
    *) return 2 ;;
  esac
}

method_for_system() {
  [[ "$1" == "dense" ]] && printf '%s\n' fastv || printf '%s\n' "$1"
}

run_cell() {
  local seed="$1" model="$2" system="$3"
  local method py config controller_dir out attempt monitor before after
  method="$(method_for_system "$system")"
  py="$(python_for_system "$system")"
  config="configs/replication_210x3/draw${seed}_${model}_${method}.json"
  if [[ "$system" == "dense" ]]; then
    controller_dir="results/replication210x3_frozen_dense/${model}"
  else
    controller_dir="results/frozen_controllers_phase_a/${model}_${method}"
  fi
  out="results/replication210x3/draw_${seed}/${model}_${system}"
  mkdir -p "$out/telemetry_attempts"
  if [[ -f "$out/execution_manifest.json" ]] && "$py" - "$out/execution_manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
raise SystemExit(0 if m.get("executed_count")==210 and m.get("maximum_backend_invocations_per_query")==1 else 1)
PY
  then
    printf '[%s] SKIP complete draw=%s model=%s system=%s\n' "$(date -Is)" "$seed" "$model" "$system" | tee -a "$QUEUE/queue.log"
  else
    attempt="$("$py" - "$out/telemetry_attempts" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True)
n=[int(x.name.rsplit("_",1)[-1]) for x in p.glob("attempt_*") if x.name.rsplit("_",1)[-1].isdigit()]
print(max(n,default=0)+1)
PY
    )"
    monitor="$out/telemetry_attempts/attempt_$(printf '%03d' "$attempt")"
    mkdir -p "$monitor"
    before=0; [[ -f "$out/executed_actions.jsonl" ]] && before="$(wc -l < "$out/executed_actions.jsonl")"
    printf '[%s] START draw=%s model=%s system=%s attempt=%s from=%s\n' "$(date -Is)" "$seed" "$model" "$system" "$attempt" "$before" | tee -a "$QUEUE/queue.log"
    local -a extra=()
    [[ "$system" == "scope" ]] && extra+=(--scope-batched-order)
    set +e
    CODECARBON_PYTHON="$py" bash scripts/run_with_codecarbon.sh \
      "$monitor/codecarbon" "replication210x3_draw${seed}_${model}_${system}_attempt${attempt}" \
      bash scripts/run_with_nvidia_monitor.sh "$monitor" \
        "$py" scripts/run_frozen_controller.py \
          --config "$config" \
          --controller "$controller_dir/${method}_controller.json" \
          --checksum "$controller_dir/${method}_controller.sha256" \
          --deployment-samples "data/replication_210x3/draw_${seed}/deployment_samples.jsonl" \
          --output-dir "$out" \
          --warmup-samples "data/replication_210x3/draw_${seed}/deployment_samples.jsonl" \
          --warmup-count 3 \
          --resume "${extra[@]}" \
      > "$monitor/combined.log" 2>&1
    status=$?
    set -e
    after=0; [[ -f "$out/executed_actions.jsonl" ]] && after="$(wc -l < "$out/executed_actions.jsonl")"
    "$FASTV_PY" - "$QUEUE/status.jsonl" "$seed" "$model" "$system" "$attempt" "$before" "$after" "$status" <<'PY'
import json,sys,time
p,seed,model,system,attempt,before,after,status=sys.argv[1:]
with open(p,"a",encoding="utf-8") as f:
 f.write(json.dumps({"epoch":time.time(),"seed":int(seed),"model":model,"system":system,"attempt":int(attempt),"before":int(before),"after":int(after),"status":int(status)},sort_keys=True)+"\n")
PY
    printf '[%s] END draw=%s model=%s system=%s status=%s progress=%s/%s\n' "$(date -Is)" "$seed" "$model" "$system" "$status" "$after" 210 | tee -a "$QUEUE/queue.log"
    if [[ "$status" -ne 0 ]]; then
      if [[ "$after" -le "$before" ]]; then
        tail -80 "$monitor/combined.log" >&2
        echo "Refusing automatic retry without durable progress" >&2
        return "$status"
      fi
      return 75
    fi
  fi
  "$py" scripts/merge_telemetry_attempts.py \
    --attempt-root "$out/telemetry_attempts" \
    --output "$out/merged_telemetry/nvidia_smi_samples.csv"
  "$py" scripts/analyze_greenmm_telemetry.py \
    --telemetry "$out/merged_telemetry/nvidia_smi_samples.csv" \
    --execution-manifest "$out/execution_manifest.json" \
    --executed-actions "$out/executed_actions.jsonl" \
    --output "$out/greenmm_telemetry.json" >/dev/null
}

run_with_retries() {
  local seed="$1" model="$2" system="$3"
  for try in 1 2 3 4 5 6; do
    set +e
    run_cell "$seed" "$model" "$system"
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then return 0; fi
    [[ "$rc" -eq 75 ]] || return "$rc"
  done
  return 1
}

# Balanced order: each non-dense system occupies a different early position.
for seed in 2101 2102 2103; do
  case "$seed" in
    2101) systems=(dense fastv scope random) ;;
    2102) systems=(scope random dense fastv) ;;
    2103) systems=(random fastv scope dense) ;;
  esac
  for model in 7b 13b; do
    for system in "${systems[@]}"; do
      run_with_retries "$seed" "$model" "$system"
    done
  done
  "$FASTV_PY" scripts/audit_replication_210x3.py --output "$QUEUE/audit_after_draw_${seed}.json" || true
  sha256sum "$QUEUE/audit_after_draw_${seed}.json" > "$QUEUE/audit_after_draw_${seed}.sha256"
done

"$FASTV_PY" scripts/audit_replication_210x3.py --require-complete --output "$QUEUE/final_audit.json"
sha256sum "$QUEUE/final_audit.json" > "$QUEUE/final_audit.sha256"
date -Is > "$QUEUE/queue_completed.txt"
