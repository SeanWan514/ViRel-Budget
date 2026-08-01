from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.frozen_controller import canonical_json_bytes, sha256_path


SEEDS = (2101, 2102, 2103)
MODELS = ("7b", "13b")
SYSTEMS = ("dense", "fastv", "scope", "random")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", default="results/replication210x3_queue/final_audit.json")
    args = parser.parse_args()
    protocol_path = Path("results/replication210x3_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cells = []
    all_groups: set[str] = set()
    draw_group_overlap = False
    for seed in SEEDS:
        groups = read_jsonl(Path(f"data/replication_210x3/draw_{seed}/source_groups.jsonl"))
        draw_groups = {row["source_group"] for row in groups}
        draw_group_overlap |= bool(all_groups & draw_groups)
        all_groups.update(draw_groups)
        for model in MODELS:
            for system in SYSTEMS:
                out = Path(f"results/replication210x3/draw_{seed}/{model}_{system}")
                actions = read_jsonl(out / "executed_actions.jsonl")
                manifest = json.loads((out / "execution_manifest.json").read_text(encoding="utf-8")) if (out / "execution_manifest.json").exists() else {}
                green = json.loads((out / "greenmm_telemetry.json").read_text(encoding="utf-8")) if (out / "greenmm_telemetry.json").exists() else {}
                attempts = sorted((out / "telemetry_attempts").glob("attempt_*"))
                codecarbon = [
                    attempt / "codecarbon" / "emissions.csv"
                    for attempt in attempts
                    if (attempt / "codecarbon" / "emissions.csv").exists()
                ]
                cells.append({
                    "seed": seed,
                    "model": model,
                    "system": system,
                    "actions": len(actions),
                    "expected_actions": 210,
                    "manifest_complete": manifest.get("executed_count") == 210,
                    "single_invocation": all(row.get("backend_invocation_count") == 1 for row in actions),
                    "unique_samples": len({row.get("sample_id") for row in actions}),
                    "nvidia_telemetry": bool(green),
                    "codecarbon_attempts": len(codecarbon),
                    "plan_sha256": manifest.get("planned_actions_sha256"),
                })
    complete = (
        not draw_group_overlap
        and len(all_groups) == 630
        and len(cells) == 24
        and all(
            cell["actions"] == 210
            and cell["manifest_complete"]
            and cell["single_invocation"]
            and cell["unique_samples"] == 210
            and cell["nvidia_telemetry"]
            and cell["codecarbon_attempts"] >= 1
            for cell in cells
        )
    )
    report = {
        "artifact_type": "virel_replication_210x3_completion_audit",
        "complete": complete,
        "protocol_sha256": sha256_path(protocol_path),
        "draw_count": 3,
        "unique_queries": 630,
        "unique_source_groups": len(all_groups),
        "draw_group_overlap": draw_group_overlap,
        "cell_count": len(cells),
        "total_actions": sum(cell["actions"] for cell in cells),
        "expected_total_actions": 5040,
        "cells": cells,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_complete and not complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
