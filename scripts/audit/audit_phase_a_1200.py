from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CELLS = [
    ("7b", "fastv", "configs/phase_a_current_llava15_7b_fastv.json", "results/phase_a_current_llava15_7b_fastv"),
    ("7b", "scope", "configs/phase_a_current_llava15_7b_scope.json", "results/phase_a_current_llava15_7b_scope"),
    ("7b", "random", "configs/phase_a_current_llava15_7b_random.json", "results/phase_a_current_llava15_7b_random"),
    ("13b", "fastv", "configs/phase_a_current_llava15_13b_fastv.json", "results/phase_a_current_llava15_13b_fastv"),
    ("13b", "scope", "configs/phase_a_current_llava15_13b_scope.json", "results/phase_a_current_llava15_13b_scope"),
    ("13b", "random", "configs/phase_a_current_llava15_13b_random.json", "results/phase_a_current_llava15_13b_random"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the common 1,200-case Phase-A matrix.")
    parser.add_argument("--write-report", default="results/phase_a_1200_status.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    dataset = ROOT / "data/paper_1200/samples.jsonl"
    dataset_count = count_jsonl(dataset)
    cells: list[dict[str, Any]] = []
    for model, method, config_name, output_name in CELLS:
        config_path = ROOT / config_name
        output = ROOT / output_name
        config = json.loads(config_path.read_text(encoding="utf-8"))
        records = output / "records.jsonl"
        if not records.exists():
            legacy_canonical = output / "records_with_measured_energy_optionfix.jsonl"
            if legacy_canonical.exists():
                records = legacy_canonical
        completion = output / "grid_completion.json"
        record_count = count_jsonl(records)
        # 400 POPE*3 + 400 MMStar*3 + 400 VCF*4 = 4,000
        # intervention evaluations; dense + four budgets = 20,000 records.
        expected = 20_000
        complete = record_count == expected
        cells.append(
            {
                "model": model,
                "method": method,
                "config": config_name,
                "config_sha256": sha256(config_path),
                "output": output_name,
                "record_count": record_count,
                "expected_record_count": expected,
                "complete": complete,
                "completion_manifest_present": completion.exists(),
                "strict_labels_present": (output / "strict_labels").exists(),
                "records_path": str(records.relative_to(ROOT)) if records.exists() else None,
                "seed": config["seed"],
                "dataset_path": config["dataset"]["path"],
                "budgets": config["pruning"]["budget_schedule"],
            }
        )

    controlled = (
        dataset_count == 1200
        and {cell["seed"] for cell in cells} == {13}
        and {tuple(cell["budgets"]) for cell in cells} == {(64, 128, 256, 432)}
        and {cell["dataset_path"] for cell in cells} == {"data/paper_1200/samples.jsonl"}
    )
    report = {
        "phase": "A_common_1200",
        "dataset_count": dataset_count,
        "dataset_sha256": sha256(dataset) if dataset.exists() else None,
        "controlled_design": controlled,
        "complete_cells": sum(cell["complete"] for cell in cells),
        "required_cells": len(cells),
        "all_complete": controlled and all(cell["complete"] for cell in cells),
        "expansion_gate_open": controlled and all(cell["complete"] for cell in cells),
        "cells": cells,
        "note": "A complete matrix is necessary but not sufficient: the scientific go/no-go also uses strict-safety and measured-efficiency results.",
    }
    out = ROOT / args.write_report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.require_complete and not report["all_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
