#!/usr/bin/env python3
"""Fail-fast checks for the compact reviewer-facing release."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def require(relative: str) -> Path:
    path = ROOT / relative
    if not path.exists():
        raise AssertionError(f"missing required artifact: {relative}")
    return path


def main() -> None:
    required = [
        "README.md",
        "REPRODUCIBILITY.md",
        "virel_budget/controller.py",
        "data/manifests/development/samples.jsonl",
        "data/manifests/prospective/samples.jsonl",
        "results/prospective/prospective_analysis.json",
        "results/replication/summary/replication_aggregate_results.csv",
        "results/carbon/whole_experiment_carbon_summary.json",
        "results/pareto/final_greenmm_analysis.json",
        "results/integrity/final_experiment_closeout_integrity.json",
    ]
    for relative in required:
        require(relative)

    forbidden_suffixes = {".tex", ".drawio"}
    forbidden_names = {"private_blind_mapping.csv", ".DS_Store"}
    offenders = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in forbidden_suffixes or path.name in forbidden_names)
    ]
    if offenders:
        raise AssertionError(f"private or draft artifacts present: {offenders}")

    carbon = json.loads(
        require("results/carbon/whole_experiment_carbon_summary.json").read_text()
    )
    totals = carbon["totals"]
    assert totals["runs"] == 66.0
    assert 4.81 < totals["measured_gpu_energy_kwh"] < 4.83

    with require(
        "results/replication/summary/replication_aggregate_results.csv"
    ).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    strong = {(row["model"], row["method"]) for row in rows if row["strong_green_replication"] == "True"}
    assert strong == {("7b", "scope"), ("13b", "scope")}
    assert all(int(row["n"]) == 630 for row in rows)

    development_count = sum(
        1 for line in require("data/manifests/development/samples.jsonl").open() if line.strip()
    )
    prospective_count = sum(
        1 for line in require("data/manifests/prospective/samples.jsonl").open() if line.strip()
    )
    assert development_count == 1200
    assert prospective_count == 900

    print("release verification: PASS")
    print(f"development={development_count}, prospective={prospective_count}, replication_cells={len(rows)}")
    print(f"strong_green_cells={sorted(strong)}")


if __name__ == "__main__":
    main()
