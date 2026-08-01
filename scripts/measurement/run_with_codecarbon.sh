#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  CODECARBON_PYTHON=/path/to/python bash scripts/run_with_codecarbon.sh <codecarbon-output-dir> <project-name> <command> [args...]

Writes:
  emissions.csv
  codecarbon_run_summary.json
  command_status.txt
EOF
  exit 2
fi

OUT_DIR="$1"
PROJECT_NAME="$2"
shift 2
mkdir -p "$OUT_DIR"

PYTHON_BIN="${CODECARBON_PYTHON:-python3}"

"$PYTHON_BIN" - "$OUT_DIR" "$PROJECT_NAME" "$@" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

out_dir = Path(sys.argv[1])
project_name = sys.argv[2]
cmd = sys.argv[3:]
status_path = out_dir / "command_status.txt"
summary_path = out_dir / "codecarbon_run_summary.json"

try:
    from codecarbon import EmissionsTracker
except Exception as exc:
    raise SystemExit(
        "CodeCarbon is required for paper-track green accounting. "
        "Install it in the run environment with: python -m pip install codecarbon"
    ) from exc

tracker = EmissionsTracker(
    project_name=project_name,
    output_dir=str(out_dir),
    output_file="emissions.csv",
    measure_power_secs=1,
    log_level="error",
    save_to_file=True,
)

started = time.time()
tracker.start()
try:
    proc = subprocess.run(cmd)
    status = int(proc.returncode)
finally:
    emissions_kg = tracker.stop()
ended = time.time()

status_path.write_text(f"{status}\n", encoding="utf-8")
summary = {
    "project_name": project_name,
    "command": cmd,
    "command_exit_status": status,
    "duration_s": ended - started,
    "codecarbon_emissions_kgco2e": emissions_kg,
    "emissions_csv": str(out_dir / "emissions.csv"),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(status)
PY
