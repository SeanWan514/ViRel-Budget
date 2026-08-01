#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_with_nvidia_monitor.sh <monitor-output-dir> <command> [args...]

Writes:
  nvidia_smi_samples.csv
  nvidia_smi_summary.json
  command_status.txt
EOF
  exit 2
fi

OUT_DIR="$1"
shift
mkdir -p "$OUT_DIR"

SAMPLES_CSV="$OUT_DIR/nvidia_smi_samples.csv"
SUMMARY_JSON="$OUT_DIR/nvidia_smi_summary.json"
HARDWARE_JSON="$OUT_DIR/hardware_metadata.json"
STATUS_TXT="$OUT_DIR/command_status.txt"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available on this machine." >&2
  exit 3
fi

python3 - "$HARDWARE_JSON" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
query = [
    "nvidia-smi",
    "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
    "--format=csv,noheader,nounits",
]
try:
    raw = subprocess.check_output(query, text=True).strip().splitlines()
    gpus = []
    for idx, line in enumerate(raw):
        parts = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "index": idx,
                "name": parts[0] if len(parts) > 0 else None,
                "uuid": parts[1] if len(parts) > 1 else None,
                "driver_version": parts[2] if len(parts) > 2 else None,
                "memory_total_mb": float(parts[3]) if len(parts) > 3 and parts[3] else None,
                "power_limit_w": float(parts[4]) if len(parts) > 4 and parts[4] else None,
            }
        )
except Exception as exc:
    gpus = []
    error = repr(exc)
else:
    error = None
out.write_text(json.dumps({"gpus": gpus, "metadata_error": error}, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "epoch,timestamp,power_w,utilization_gpu_pct,memory_used_mb,memory_total_mb" > "$SAMPLES_CSV"
(
  nvidia-smi \
    --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits \
    -lms 200 \
    | python3 -c 'import sys, time
for line in sys.stdin:
    print(f"{time.time():.6f},{line.rstrip()}", flush=True)'
) >> "$SAMPLES_CSV" &
MONITOR_PID="$!"

START_EPOCH="$(python3 - <<'PY'
import time
print(time.time())
PY
)"
set +e
"$@"
STATUS="$?"
set -e
END_EPOCH="$(python3 - <<'PY'
import time
print(time.time())
PY
)"

pkill -P "$MONITOR_PID" >/dev/null 2>&1 || true
kill "$MONITOR_PID" >/dev/null 2>&1 || true
wait "$MONITOR_PID" 2>/dev/null || true
printf '%s\n' "$STATUS" > "$STATUS_TXT"

python3 - "$SAMPLES_CSV" "$SUMMARY_JSON" "$START_EPOCH" "$END_EPOCH" "$STATUS" <<'PY'
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

samples_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
start_epoch = float(sys.argv[3])
end_epoch = float(sys.argv[4])
status = int(sys.argv[5])

rows = []
with samples_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            rows.append(
                {
                    "power_w": float(row["power_w"]),
                    "utilization_gpu_pct": float(row["utilization_gpu_pct"]),
                    "memory_used_mb": float(row["memory_used_mb"]),
                    "memory_total_mb": float(row["memory_total_mb"]),
                }
            )
        except Exception:
            continue

duration_s = max(end_epoch - start_epoch, 0.0)
powers = [row["power_w"] for row in rows]
utils = [row["utilization_gpu_pct"] for row in rows]
mem_used = [row["memory_used_mb"] for row in rows]
summary = {
    "command_exit_status": status,
    "duration_s": duration_s,
    "sample_count": len(rows),
    "mean_power_w": statistics.fmean(powers) if powers else None,
    "median_power_w": statistics.median(powers) if powers else None,
    "max_power_w": max(powers) if powers else None,
    "mean_gpu_utilization_pct": statistics.fmean(utils) if utils else None,
    "max_gpu_utilization_pct": max(utils) if utils else None,
    "mean_memory_used_mb": statistics.fmean(mem_used) if mem_used else None,
    "max_memory_used_mb": max(mem_used) if mem_used else None,
    "memory_total_mb": rows[0]["memory_total_mb"] if rows else None,
    "measured_energy_joule": (statistics.fmean(powers) * duration_s) if powers else None,
    "monitor": "nvidia-smi 200ms polling",
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "$STATUS"
