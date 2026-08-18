#!/usr/bin/env python3
"""Fail-fast checks for the compact reviewer-facing release."""

from __future__ import annotations

import csv
import hashlib
import json
import re
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
        "results/feature_analysis/development_features_and_folds.csv",
        "results/feature_analysis/feature_ablation_by_controller.csv",
        "results/feature_analysis/feature_ablation_cross_controller_summary.json",
        "scripts/analysis/run_feature_ablation.py",
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

    forbidden_directories = ["results/qualitative"]
    present_directories = [name for name in forbidden_directories if (ROOT / name).exists()]
    if present_directories:
        raise AssertionError(f"non-redistributable directories present: {present_directories}")

    text_suffixes = {".csv", ".json", ".jsonl", ".md", ".py", ".toml", ".cff", ".txt"}
    private_patterns = {
        "absolute user path": re.compile(r"/Users/|/home/[^/]+/"),
        "anonymous-review remnant": re.compile(r"anonymous|4open\.science", re.IGNORECASE),
        "secret material": re.compile(
            r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|(?:api[_-]?key|password)\s*[:=]\s*['\"][^'\"]+",
            re.IGNORECASE,
        ),
    }
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in text_suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in private_patterns.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(ROOT)}")
    if findings:
        raise AssertionError(f"release hygiene failures: {findings}")

    oversized = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.stat().st_size > 10 * 1024 * 1024
    ]
    if oversized:
        raise AssertionError(f"unexpected files larger than 10 MiB: {oversized}")

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

    with require("results/feature_analysis/development_features_and_folds.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        feature_rows = list(csv.DictReader(handle))
    assert len(feature_rows) == 1200
    assert sorted({int(row["fold"]) for row in feature_rows}) == [0, 1, 2, 3, 4]
    feature_manifest = json.loads(
        require("results/feature_analysis/feature_ablation_run_manifest.json").read_text()
    )
    feature_digest = hashlib.sha256(
        require("results/feature_analysis/development_features_and_folds.csv").read_bytes()
    ).hexdigest()
    assert feature_manifest["development_features_sha256"] == feature_digest

    with require("results/feature_analysis/feature_ablation_by_controller.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        feature_results = list(csv.DictReader(handle))
    assert len(feature_results) == 42
    assert all(row["risk_limit_met"] == "True" for row in feature_results)

    print("release verification: PASS")
    print(f"development={development_count}, prospective={prospective_count}, replication_cells={len(rows)}")
    print(f"strong_green_cells={sorted(strong)}")
    print(f"feature_rows={len(feature_rows)}, feature_policy_evaluations={len(feature_results)}")


if __name__ == "__main__":
    main()
