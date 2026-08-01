from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.measurement.attribute_nvidia_energy import (  # noqa: E402
    _attribute_call,
    _read_power_samples,
)
from virel_budget.frozen_controller import canonical_json_bytes  # noqa: E402
from virel_budget.metrics import exact_match  # noqa: E402


MODELS = ("7b", "13b")
METHODS = ("fastv", "scope", "random")
BUDGETS = (64, 128, 256, 432)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def bootstrap_energy_reduction(
    controller: dict[str, float],
    dense: dict[str, float],
    *,
    seed: int,
    iterations: int = 5000,
) -> dict[str, Any]:
    ids = sorted(set(controller) & set(dense))
    observed = 1 - statistics.fmean(controller[i] for i in ids) / statistics.fmean(
        dense[i] for i in ids
    )
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [ids[rng.randrange(len(ids))] for _ in ids]
        draws.append(
            1
            - statistics.fmean(controller[i] for i in sampled)
            / statistics.fmean(dense[i] for i in sampled)
        )
    draws.sort()
    return {
        "n_paired": len(ids),
        "energy_reduction": observed,
        "bootstrap_95pct_ci": [
            draws[int(0.025 * (len(draws) - 1))],
            draws[int(0.975 * (len(draws) - 1))],
        ],
        "bootstrap_iterations": iterations,
        "seed": seed,
    }


def cell_dir(model: str, method: str) -> Path:
    return ROOT / "results" / f"prospective900_llava15_{model}_{method}"


def execution_dir(model: str, method: str) -> Path:
    return ROOT / "results" / f"prospective900_execution_llava15_{model}_{method}"


def strict_index(model: str, method: str) -> dict[tuple[str, int], dict[str, str]]:
    rows = read_csv(cell_dir(model, method) / "strict_labels/safe_budget_labels.csv")
    return {(row["sample_id"], int(row["budget"])): row for row in rows}


def original_calls(model: str, method: str) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}

    def ingest(source_method: str, accepted_methods: set[str]) -> None:
        directory = cell_dir(model, source_method)
        records = read_jsonl(directory / "records.jsonl")
        samples = _read_power_samples(directory / "merged_telemetry/nvidia_smi_samples.csv")
        for record in records:
            if str(record["method"]) not in accepted_methods:
                continue
            budget = str(record["budget"])
            key = (record["sample_id"], budget)
            if key in output:
                continue
            call = (record.get("metadata") or {}).get("original_backend") or {}
            energy = _attribute_call(call, samples)["measured_energy_joule"]
            start = call.get("start_epoch")
            end = call.get("end_epoch")
            if energy is None or start is None or end is None:
                raise ValueError(f"Missing original-call telemetry for {model}/{method}/{key}")
            output[key] = {
                "energy_joule": float(energy),
                "latency_ms": max((float(end) - float(start)) * 1000.0, 0.0),
                "answer": record["answer"],
                "is_correct": bool(record["is_correct"]),
            }

    if method == "random":
        ingest("fastv", {"dense"})
        ingest("random", {"random"})
    else:
        ingest(method, {"dense", method})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the sealed prospective 900-case experiment.")
    parser.add_argument("--output-dir", default="results/phase_b_analysis")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    audit = json.loads(
        (ROOT / "results/phase_b_queue/final_phase_b_audit.json").read_text(encoding="utf-8")
    )
    if audit.get("all_complete") is not True:
        raise ValueError("Phase-B analysis requires the complete six-cell prospective audit")
    revealed = {
        row["sample_id"]: row
        for row in read_jsonl(ROOT / "data/prospective_900_revealed/samples.jsonl")
    }
    if len(revealed) != 900:
        raise ValueError("Revealed prospective population is not exactly 900")

    controller_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    for model in MODELS:
        for method in METHODS:
            strict = strict_index(model, method)
            calls = original_calls(model, method)
            actions = read_jsonl(execution_dir(model, method) / "executed_actions.jsonl")
            telemetry = json.loads(
                (execution_dir(model, method) / "greenmm_telemetry.json").read_text(
                    encoding="utf-8"
                )
            )
            energy_by_id = {
                row["sample_id"]: float(row["gpu_energy_joule"])
                for row in telemetry["per_action"]
            }
            dense_energy = {
                sample_id: calls[(sample_id, "full")]["energy_joule"] for sample_id in revealed
            }
            unsafe = 0
            accepted = 0
            correct = 0
            supported_correct = 0
            selected_tokens = []
            action_energy: dict[str, float] = {}
            action_latency: dict[str, float] = {}
            for action in actions:
                sample_id = action["sample_id"]
                selected = action["selected_budget"]
                is_dense = selected == "dense"
                row = strict[(sample_id, BUDGETS[-1])] if is_dense else strict[(sample_id, int(selected))]
                task_correct = exact_match(action["answer"], revealed[sample_id]["answer"])
                unsafe_acceptance = (not is_dense) and row["reference_safe"] != "True"
                is_supported_correct = (
                    row["dense_correct"] == "True" and row["dense_reliant"] == "True"
                    if is_dense
                    else row["combined_gold_safe"] == "True"
                )
                unsafe += int(unsafe_acceptance)
                accepted += int(not is_dense)
                correct += int(task_correct)
                supported_correct += int(is_supported_correct)
                selected_tokens.append(576 if is_dense else int(selected))
                action_energy[sample_id] = energy_by_id[sample_id]
                action_latency[sample_id] = float(action["model_latency_ms"])
                decision_records.append(
                    {
                        "model": model,
                        "method": method,
                        "sample_id": sample_id,
                        "dataset": action["dataset"],
                        "selected_budget": selected,
                        "task_correct": task_correct,
                        "unsafe_acceptance": unsafe_acceptance,
                        "supported_correct": is_supported_correct,
                        "gpu_energy_joule": energy_by_id[sample_id],
                        "latency_ms": float(action["model_latency_ms"]),
                    }
                )
            energy_bootstrap = bootstrap_energy_reduction(
                action_energy,
                dense_energy,
                seed=13 + (0 if model == "7b" else 100) + METHODS.index(method),
            )
            mean_energy = statistics.fmean(action_energy.values())
            mean_dense_energy = statistics.fmean(dense_energy.values())
            unsafe_ci = wilson(unsafe, len(actions))
            controller_rows.append(
                {
                    "model": model,
                    "method": method,
                    "n": len(actions),
                    "dense_avoidance_rate": accepted / len(actions),
                    "selected_budget_distribution": dict(
                        Counter(str(row["selected_budget"]) for row in actions)
                    ),
                    "mean_selected_visual_tokens": statistics.fmean(selected_tokens),
                    "token_reduction_vs_dense": 1 - statistics.fmean(selected_tokens) / 576,
                    "unsafe_acceptance_rate_full": unsafe / len(actions),
                    "unsafe_acceptance_95pct_ci": unsafe_ci,
                    "unsafe_acceptance_count": unsafe,
                    "prospective_risk_limit_met_point_estimate": unsafe / len(actions) <= 0.05,
                    "prospective_risk_limit_met_95pct_upper": unsafe_ci[1] <= 0.05,
                    "unconditional_accuracy": correct / len(actions),
                    "unconditional_accuracy_95pct_ci": wilson(correct, len(actions)),
                    "supported_correct_rate_full": supported_correct / len(actions),
                    "supported_correct_95pct_ci": wilson(supported_correct, len(actions)),
                    "mean_call_window_gpu_energy_joule": mean_energy,
                    "mean_dense_call_window_gpu_energy_joule": mean_dense_energy,
                    "energy_reduction_vs_dense": energy_bootstrap["energy_reduction"],
                    "energy_reduction_bootstrap_95pct_ci": energy_bootstrap["bootstrap_95pct_ci"],
                    "mean_model_latency_ms": statistics.fmean(action_latency.values()),
                    "mean_dense_latency_ms": statistics.fmean(
                        calls[(sample_id, "full")]["latency_ms"] for sample_id in revealed
                    ),
                    "correct_answers_per_kwh": (correct / len(actions)) * 3_600_000 / mean_energy,
                    "supported_correct_answers_per_kwh": (
                        (supported_correct / len(actions)) * 3_600_000 / mean_energy
                    ),
                    "complete_inference_block_gpu_energy_joule_per_query": telemetry[
                        "complete_execution_inference_window_joule_per_query"
                    ],
                    "execution_resumed_from_count": telemetry["resumed_from_count"],
                    "mean_controller_cpu_latency_ms": telemetry["mean_controller_cpu_latency_ms"],
                }
            )

            for budget in BUDGETS:
                rows = [strict[(sample_id, budget)] for sample_id in revealed]
                reliant = [row for row in rows if row["dense_reliant"] == "True"]
                correct_reliant = [
                    row
                    for row in rows
                    if row["dense_correct"] == "True" and row["dense_reliant"] == "True"
                ]
                energies = [calls[(sample_id, str(budget))]["energy_joule"] for sample_id in revealed]
                latencies = [calls[(sample_id, str(budget))]["latency_ms"] for sample_id in revealed]
                accuracy_count = sum(
                    calls[(sample_id, str(budget))]["is_correct"] for sample_id in revealed
                )
                reference_safe_count = sum(row["reference_safe"] == "True" for row in reliant)
                supported_count = sum(
                    row["combined_gold_safe"] == "True" for row in correct_reliant
                )
                fixed_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "budget": budget,
                        "n": 900,
                        "unconditional_accuracy": accuracy_count / 900,
                        "strict_reference_safe_rate_dense_reliant": (
                            reference_safe_count / len(reliant) if reliant else float("nan")
                        ),
                        "dense_reliant_n": len(reliant),
                        "supported_correct_rate_full": supported_count / 900,
                        "mean_call_window_gpu_energy_joule": statistics.fmean(energies),
                        "mean_latency_ms": statistics.fmean(latencies),
                        "supported_correct_answers_per_kwh": (
                            (supported_count / 900) * 3_600_000 / statistics.fmean(energies)
                        ),
                    }
                )

    write_csv(output / "prospective_controller_summary.csv", controller_rows)
    write_csv(output / "prospective_fixed_budget_summary.csv", fixed_rows)
    write_csv(output / "prospective_controller_decisions.csv", decision_records)
    conclusion = {
        "phase": "B_prospective_900_analysis",
        "scope": {
            "development": 1200,
            "prospective": 900,
            "combined": 2100,
            "models": list(MODELS),
            "methods": list(METHODS),
        },
        "controller_results": controller_rows,
        "claim_rules": {
            "deployable_green_controller": (
                "Requires prospective 95% unsafe-acceptance upper bound <=5% and positive lower "
                "95% bootstrap bound for measured GPU energy reduction versus dense."
            ),
            "risk_efficiency_frontier": (
                "Use when reliability or efficiency benefits are operating-point dependent and "
                "no single controller dominates dense across all model/method cells."
            ),
            "offline_audit": (
                "Use when prospective deployment control is unsafe or not energy-efficient, while "
                "fixed-budget/oracle results still characterize attainable compression."
            ),
        },
        "measurement_boundary": (
            "Primary paired reductions use GPU call-window energy from 200-ms NVIDIA telemetry. "
            "Controller inference-block energy is also reported; CPU controller latency is reported "
            "separately. These are not full-platform or embodied-carbon measurements."
        ),
    }
    (output / "prospective_analysis.json").write_bytes(canonical_json_bytes(conclusion))

    deployable = [
        row
        for row in controller_rows
        if row["prospective_risk_limit_met_95pct_upper"]
        and row["energy_reduction_bootstrap_95pct_ci"][0] > 0
    ]
    lines = [
        "# Prospective 900-Case Result",
        "",
        f"Complete scope: 1,200 development + 900 sealed prospective = **2,100 queries**.",
        "",
        "## Claim decision",
        "",
        (
            f"{len(deployable)} of 6 model/method controllers meet both the prospective risk and measured-energy criteria."
            if deployable
            else "No model/method controller meets both confirmatory criteria; use the risk–efficiency-frontier or offline-audit framing."
        ),
        "",
        "A cell is called deployable-green only when its prospective unsafe-acceptance 95% upper bound is at most 5% and its paired energy-reduction 95% lower bound is positive.",
        "",
        "## Mandatory boundaries",
        "",
        "- Prospective labels were revealed only after all six one-action executions completed.",
        "- Every deployment query used exactly one backend invocation.",
        "- Intervention grids are evaluation labels, not controller inputs.",
        "- GPU energy is measured device energy, not full-system lifecycle carbon.",
        "- Results must be reported by model and pruning method; pooled averages cannot hide failures.",
        "",
    ]
    (output / "PROSPECTIVE_RESULT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"deployable_green_cells": len(deployable), "rows": len(controller_rows)}, indent=2))


if __name__ == "__main__":
    main()
