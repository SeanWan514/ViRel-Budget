from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CELLS = [
    ("7b", "fastv"),
    ("7b", "scope"),
    ("7b", "random"),
    ("13b", "fastv"),
    ("13b", "scope"),
    ("13b", "random"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the sealed prospective 900-case matrix.")
    parser.add_argument("--write-report", default="results/phase_b_queue/final_phase_b_audit.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    deployment = ROOT / "data/prospective_900_frozen/deployment_samples.jsonl"
    revealed = ROOT / "data/prospective_900_revealed/samples.jsonl"
    approval = ROOT / "results/phase_b_expansion_approval.json"
    rows: list[dict[str, Any]] = []
    for model, method in CELLS:
        execution = ROOT / "results" / f"prospective900_execution_llava15_{model}_{method}"
        manifest_path = execution / "execution_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        grid = ROOT / "results" / f"prospective900_llava15_{model}_{method}"
        rows.append(
            {
                "model": model,
                "method": method,
                "execution_count": count_jsonl(execution / "executed_actions.jsonl"),
                "execution_manifest_count": manifest.get("executed_count"),
                "single_action": manifest.get("maximum_backend_invocations_per_query") == 1,
                "resident_backend_replica_count": manifest.get("resident_backend_replica_count"),
                "scope_batched_order": manifest.get("scope_batched_order"),
                "execution_order_rule": manifest.get("execution_order_rule"),
                "labels_hidden_during_execution": manifest.get("gold_or_intervention_labels_loaded")
                is False,
                "grid_record_count": count_jsonl(grid / "records.jsonl"),
                "grid_expected_record_count": 15000,
                "grid_complete": count_jsonl(grid / "records.jsonl") == 15000,
                "strict_labels_present": (grid / "strict_labels/strict_label_manifest.json").exists(),
                "measured_energy_present": (grid / "measured_energy_attribution.json").exists(),
                "telemetry_status_zero": any(
                    path.read_text(encoding="utf-8").strip() == "0"
                    for path in grid.glob("telemetry_attempts/attempt_*/command_status.txt")
                ),
            }
        )
    all_complete = (
        count_jsonl(deployment) == 900
        and count_jsonl(revealed) == 900
        and approval.exists()
        and all(
            row["execution_count"] == 900
            and row["execution_manifest_count"] == 900
            and row["single_action"]
            and row["labels_hidden_during_execution"]
            and row["grid_complete"]
            and row["strict_labels_present"]
            and row["measured_energy_present"]
            and row["telemetry_status_zero"]
            for row in rows
        )
    )
    report = {
        "phase": "B_prospective_900",
        "all_complete": all_complete,
        "prospective_count": count_jsonl(deployment),
        "revealed_count": count_jsonl(revealed),
        "combined_scope": 2100 if all_complete else None,
        "deployment_sha256": sha256(deployment),
        "approval_sha256": sha256(approval) if approval.exists() else None,
        "cells": rows,
    }
    out = ROOT / args.write_report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.require_complete and not all_complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
