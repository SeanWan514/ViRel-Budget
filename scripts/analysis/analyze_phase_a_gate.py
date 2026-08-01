from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BUDGETS = (64, 128, 256, 432)
MODELS = ("7b", "13b")
METHODS = ("fastv", "scope", "random")


def _canonical(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cell(model: str, method: str) -> Path:
    return ROOT / "results" / f"phase_a_current_llava15_{model}_{method}"


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _paired_bootstrap(
    left: dict[str, bool],
    right: dict[str, bool],
    *,
    iterations: int = 5000,
    seed: int = 13,
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    deltas = [int(left[sample_id]) - int(right[sample_id]) for sample_id in ids]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        draws.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    draws.sort()
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return {
        "n_paired": len(ids),
        "difference": observed,
        "bootstrap_95pct_ci": [lower, upper],
        "left_only_safe": sum(left[sample_id] and not right[sample_id] for sample_id in ids),
        "right_only_safe": sum(right[sample_id] and not left[sample_id] for sample_id in ids),
        "bootstrap_iterations": iterations,
        "seed": seed,
    }


def _strict_summary(model: str, method: str) -> tuple[list[dict[str, Any]], dict[int, dict[str, bool]]]:
    rows = _read_csv(_cell(model, method) / "strict_labels" / "safe_budget_labels.csv")
    output: list[dict[str, Any]] = []
    by_budget: dict[int, dict[str, bool]] = {}
    for budget in BUDGETS:
        selected = [row for row in rows if int(row["budget"]) == budget]
        reliant = [row for row in selected if row["dense_reliant"] == "True"]
        dense_correct = [row for row in selected if row["dense_correct"] == "True"]
        correct_reliant = [
            row
            for row in selected
            if row["dense_reliant"] == "True" and row["dense_correct"] == "True"
        ]
        reference = [row["reference_safe"] == "True" for row in reliant]
        answer_fidelity = [row["answer_fidelity"] == "True" for row in selected]
        gold_safe = [row["combined_gold_safe"] == "True" for row in correct_reliant]
        output.append(
            {
                "model": model,
                "method": method,
                "budget": budget,
                "n_full": len(selected),
                "answer_fidelity_full": _rate(answer_fidelity),
                "dense_correct_coverage": len(dense_correct) / len(selected),
                "dense_reliant_coverage": len(reliant) / len(selected),
                "dense_correct_reliant_coverage": len(correct_reliant) / len(selected),
                "n_dense_reliant": len(reliant),
                "strict_reference_safe_rate": _rate(reference),
                "strict_reference_safe_95pct_ci": _wilson(sum(reference), len(reference)),
                "n_dense_correct_reliant": len(correct_reliant),
                "combined_gold_safe_rate": _rate(gold_safe),
                "combined_gold_safe_95pct_ci": _wilson(sum(gold_safe), len(gold_safe)),
            }
        )
        by_budget[budget] = {
            row["sample_id"]: row["reference_safe"] == "True" for row in reliant
        }
    return output, by_budget


def _energy_summary(model: str, method: str) -> list[dict[str, Any]]:
    rows = _read_csv(_cell(model, method) / "measured_energy_by_method_budget.csv")
    indexed = {(row["method"], row["budget"]): row for row in rows}
    if method == "random":
        # Random reused the identical FastV dense calls. Its seeded dense rows were
        # intentionally outside this cell's telemetry window.
        dense_rows = _read_csv(_cell(model, "fastv") / "measured_energy_by_method_budget.csv")
        dense = next(row for row in dense_rows if row["method"] == "dense")
        dense_source = f"{model}_fastv_dense_reused"
    else:
        dense = indexed[("dense", "full")]
        dense_source = f"{model}_{method}_environment_matched_dense"
    dense_energy = float(dense["mean_measured_energy_per_call_joule"])
    dense_latency = float(dense["mean_latency_ms"])
    output = []
    for budget in BUDGETS:
        row = indexed[(method, str(budget))]
        energy = float(row["mean_measured_energy_per_call_joule"])
        latency = float(row["mean_latency_ms"])
        output.append(
            {
                "model": model,
                "method": method,
                "budget": budget,
                "n_measured_calls": int(row["n_measured_calls"]),
                "mean_energy_per_call_joule": energy,
                "dense_energy_per_call_joule": dense_energy,
                "energy_reduction_vs_dense": 1 - energy / dense_energy,
                "mean_latency_ms": latency,
                "dense_latency_ms": dense_latency,
                "latency_reduction_vs_dense": 1 - latency / dense_latency,
                "dense_reference_source": dense_source,
            }
        )
    return output


def _dense_original_answers(model: str, method: str) -> dict[str, str]:
    rows = _read_jsonl(_cell(model, method) / "records_with_measured_energy.jsonl")
    answers: dict[str, set[str]] = {}
    for row in rows:
        if row["method"] != "dense":
            continue
        answers.setdefault(row["sample_id"], set()).add(_canonical(row["answer"]))
    inconsistent = {sample_id: values for sample_id, values in answers.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"{model}/{method}: dense original answer varies across interventions")
    return {sample_id: next(iter(values)) for sample_id, values in answers.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Scientific Gate-A analysis for the six-cell matrix.")
    parser.add_argument("--output-dir", default="results/phase_a_gate")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    completion = json.loads(
        (ROOT / "results/phase_a_current_queue/final_phase_a_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if not completion.get("all_complete"):
        raise ValueError("Gate A cannot be evaluated before all six Phase-A cells complete")

    strict_rows: list[dict[str, Any]] = []
    safety_maps: dict[tuple[str, str, int], dict[str, bool]] = {}
    energy_rows: list[dict[str, Any]] = []
    monotonicity: list[dict[str, Any]] = []
    for model in MODELS:
        for method in METHODS:
            summaries, maps = _strict_summary(model, method)
            strict_rows.extend(summaries)
            for budget, values in maps.items():
                safety_maps[(model, method, budget)] = values
            energy_rows.extend(_energy_summary(model, method))
            audit = json.loads(
                (_cell(model, method) / "strict_labels/budget_monotonicity_audit.json").read_text(
                    encoding="utf-8"
                )
            )[method]
            monotonicity.append(
                {
                    "model": model,
                    "method": method,
                    "n": audit["n"],
                    "monotonic_rate": audit["monotonic_rate"],
                    "nonmonotonic_count": audit["nonmonotonic_count"],
                    "pattern_counts": audit["pattern_counts"],
                }
            )

    comparisons = []
    for model in MODELS:
        for method in ("fastv", "scope"):
            for budget in BUDGETS:
                comparison = _paired_bootstrap(
                    safety_maps[(model, method, budget)],
                    safety_maps[(model, "random", budget)],
                    seed=13 + budget + (0 if model == "7b" else 1000),
                )
                comparisons.append(
                    {"model": model, "method": method, "baseline": "random", "budget": budget, **comparison}
                )

    dense_agreement = []
    for model in MODELS:
        fastv = _dense_original_answers(model, "fastv")
        scope = _dense_original_answers(model, "scope")
        ids = sorted(set(fastv) & set(scope))
        agreements = sum(fastv[sample_id] == scope[sample_id] for sample_id in ids)
        dense_agreement.append(
            {
                "model": model,
                "n_common": len(ids),
                "exact_normalized_agreements": agreements,
                "agreement_rate": agreements / len(ids),
                "disagreement_count": len(ids) - agreements,
            }
        )

    strict_index = {
        (row["model"], row["method"], row["budget"]): row for row in strict_rows
    }
    energy_index = {
        (row["model"], row["method"], row["budget"]): row for row in energy_rows
    }
    comparison_index = {
        (row["model"], row["method"], row["budget"]): row for row in comparisons
    }

    integrity_pass = bool(completion["all_complete"]) and all(
        (_cell(model, method) / "strict_labels/strict_label_manifest.json").exists()
        and (_cell(model, method) / "measured_energy_attribution.json").exists()
        for model in MODELS
        for method in METHODS
    )
    scope_random_advantage_pass = all(
        comparison_index[(model, "scope", 64)]["difference"] >= 0.05
        for model in MODELS
    )
    informed_reliability_pass = all(
        max(
            strict_index[(model, method, 432)]["strict_reference_safe_rate"]
            for method in ("fastv", "scope")
        )
        >= 0.85
        for model in MODELS
    )
    cross_model_pass = all(
        strict_index[(model, "scope", 64)]["strict_reference_safe_rate"] >= 0.55
        and strict_index[(model, "fastv", 432)]["strict_reference_safe_rate"] >= 0.90
        for model in MODELS
    )
    measured_efficiency_pass = all(
        energy_index[(model, "scope", 64)]["energy_reduction_vs_dense"] >= 0.10
        and energy_index[(model, "scope", 64)]["latency_reduction_vs_dense"] >= 0.10
        for model in MODELS
    )
    gate_checks = {
        "six_cell_integrity": integrity_pass,
        "scope_beats_random_by_at_least_5pp_at_budget_64_in_both_models": scope_random_advantage_pass,
        "at_least_one_informed_method_reaches_85pct_strict_safety_at_budget_432_in_both_models": informed_reliability_pass,
        "cross_model_consistency_floor_is_met": cross_model_pass,
        "scope_budget64_reduces_measured_energy_and_latency_by_at_least_10pct_in_both_models": measured_efficiency_pass,
    }
    approved = all(gate_checks.values())

    limitations = [
        "Phase A is development evidence; it cannot establish prospective controller performance.",
        "Random dense energy is inherited from the identical FastV dense run because reused dense calls were not re-executed.",
        "FastV energy improvements are small or inconsistent; Green claims must be method- and hardware-specific.",
        "Strict safety is intervention-defined reliance preservation, not universal semantic-grounding certification.",
        "Non-monotonic budget patterns require budget-specific safety prediction rather than assuming that more tokens always repair behavior.",
    ]
    decision = {
        "phase": "A_common_1200_scientific_gate",
        "approved_for_frozen_phase_b": approved,
        "gate_checks": gate_checks,
        "interpretation": (
            "Proceed to the sealed prospective phase. Phase A supports a risk-efficiency-frontier "
            "hypothesis, especially for SCOPE, but does not by itself prove a deployable green controller."
            if approved
            else "Do not spend GPU time on Phase B until the failed gate checks are resolved or the claim is narrowed."
        ),
        "limitations": limitations,
        "gate_status_note": (
            "The qualitative gate dimensions were frozen before Phase A. Numeric pass thresholds "
            "were operationalized during the development-stage audit and are not confirmatory endpoints."
        ),
        "completion_audit_sha256": _sha256(
            ROOT / "results/phase_a_current_queue/final_phase_a_audit.json"
        ),
        "artifacts": {
            "strict_summary_csv": "strict_safety_summary.csv",
            "random_comparison_csv": "paired_random_comparison.csv",
            "energy_summary_csv": "measured_efficiency_summary.csv",
            "monotonicity_csv": "monotonicity_summary.csv",
            "dense_agreement_json": "dense_environment_agreement.json",
        },
    }

    _write_csv(output / "strict_safety_summary.csv", strict_rows)
    _write_csv(output / "paired_random_comparison.csv", comparisons)
    _write_csv(output / "measured_efficiency_summary.csv", energy_rows)
    _write_csv(output / "monotonicity_summary.csv", monotonicity)
    (output / "dense_environment_agreement.json").write_text(
        json.dumps(dense_agreement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "gate_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase-A Scientific Gate",
        "",
        f"**Decision:** {'PASS — proceed to the sealed prospective phase.' if approved else 'DO NOT PROCEED.'}",
        "",
        "## Gate checks",
        "",
        *[f"- {'PASS' if value else 'FAIL'}: {name.replace('_', ' ')}" for name, value in gate_checks.items()],
        "",
        "## Claim boundary",
        "",
        decision["interpretation"],
        "",
        "Phase A remains development evidence. The prospective 900-case execution—not this gate—determines whether the final claim is a deployable controller, a risk–efficiency frontier, or an offline audit.",
        "",
        "## Mandatory limitations",
        "",
        *[f"- {item}" for item in limitations],
        "",
    ]
    (output / "PHASE_A_GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if approved else 2


if __name__ == "__main__":
    raise SystemExit(main())
