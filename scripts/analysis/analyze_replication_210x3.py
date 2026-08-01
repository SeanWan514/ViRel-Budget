from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.frozen_controller import canonical_json_bytes, sha256_path
from virel_budget.metrics import exact_match


SEEDS = (2101, 2102, 2103)
MODELS = ("7b", "13b")
METHODS = ("fastv", "scope", "random")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        return [float("nan"), float("nan")]
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def strict_index(model: str, method: str) -> dict[tuple[str, int], dict[str, str]]:
    output: dict[tuple[str, int], dict[str, str]] = {}
    for root in (
        Path(f"results/phase_a_current_llava15_{model}_{method}/strict_labels/safe_budget_labels.csv"),
        Path(f"results/prospective900_llava15_{model}_{method}/strict_labels/safe_budget_labels.csv"),
    ):
        for row in read_csv(root):
            output[(row["sample_id"], int(row["budget"]))] = row
    return output


def codecarbon_totals(out: Path) -> dict[str, float]:
    values = {"energy_kwh": 0.0, "emissions_kgco2e": 0.0, "duration_s": 0.0, "runs": 0.0}
    for path in sorted((out / "telemetry_attempts").glob("attempt_*/codecarbon/emissions.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        row = rows[-1]
        values["energy_kwh"] += float(row["energy_consumed"])
        values["emissions_kgco2e"] += float(row["emissions"])
        values["duration_s"] += float(row["duration"])
        values["runs"] += 1
    return values


def nvidia_wrapper_totals(out: Path) -> dict[str, float]:
    values = {"energy_joule": 0.0, "duration_s": 0.0, "max_memory_mb": 0.0, "runs": 0.0}
    for path in sorted((out / "telemetry_attempts").glob("attempt_*/nvidia_smi_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("measured_energy_joule") is None:
            continue
        values["energy_joule"] += float(row["measured_energy_joule"])
        values["duration_s"] += float(row["duration_s"])
        values["max_memory_mb"] = max(values["max_memory_mb"], float(row.get("max_memory_used_mb") or 0))
        values["runs"] += 1
    return values


def bootstrap_reduction(
    method: dict[str, float], dense: dict[str, float], *, seed: int, iterations: int = 5000
) -> dict[str, Any]:
    ids = sorted(set(method) & set(dense))
    observed = 1 - statistics.fmean(method[i] for i in ids) / statistics.fmean(dense[i] for i in ids)
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = [rng.choice(ids) for _ in ids]
        values.append(
            1 - statistics.fmean(method[i] for i in sample) / statistics.fmean(dense[i] for i in sample)
        )
    values.sort()
    return {
        "energy_reduction": observed,
        "bootstrap_95pct_ci": [values[int(0.025 * iterations)], values[int(0.975 * iterations) - 1]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="results/replication210x3_queue/final_audit.json")
    parser.add_argument("--output-dir", default="results/replication210x3_analysis")
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--carbon-kg-per-kwh", type=float, default=0.03283)
    parser.add_argument("--cloud-usd-per-hour", type=float, default=2.0)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    if audit.get("complete") is not True:
        raise ValueError("Replication analysis requires a complete final audit")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    draw_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    energy_pool: dict[tuple[str, str], dict[str, float]] = {}
    dense_pool: dict[str, dict[str, float]] = {model: {} for model in MODELS}
    for seed in SEEDS:
        evaluation = {
            row["sample_id"]: row
            for row in read_jsonl(Path(f"data/replication_210x3/draw_{seed}/evaluation_samples.jsonl"))
        }
        for model in MODELS:
            dense_out = Path(f"results/replication210x3/draw_{seed}/{model}_dense")
            dense_actions = {row["sample_id"]: row for row in read_jsonl(dense_out / "executed_actions.jsonl")}
            dense_tel = json.loads((dense_out / "greenmm_telemetry.json").read_text(encoding="utf-8"))
            dense_energy = {row["sample_id"]: float(row["gpu_energy_joule"]) for row in dense_tel["per_action"]}
            dense_pool[model].update({f"{seed}:{key}": value for key, value in dense_energy.items()})
            dense_cc = codecarbon_totals(dense_out)
            dense_nv = nvidia_wrapper_totals(dense_out)
            dense_correct = sum(exact_match(action["answer"], evaluation[sid]["answer"]) for sid, action in dense_actions.items())
            draw_rows.append({
                "seed": seed, "model": model, "system": "dense", "n": 210,
                "dense_avoidance_rate": 0.0, "unsafe_acceptance_rate": 0.0,
                "accuracy": dense_correct / 210,
                "mean_visual_tokens": statistics.fmean(float(row["visual_token_count"]) for row in dense_actions.values()),
                "mean_latency_ms": statistics.fmean(float(row["model_latency_ms"]) for row in dense_actions.values()),
                "mean_call_window_gpu_energy_joule": statistics.fmean(dense_energy.values()),
                "energy_reduction_vs_dense": 0.0,
                "nvidia_wrapper_energy_kwh": dense_nv["energy_joule"] / 3_600_000,
                "nvidia_wrapper_duration_h": dense_nv["duration_s"] / 3600,
                "codecarbon_energy_kwh": dense_cc["energy_kwh"],
                "codecarbon_emissions_kgco2e": dense_cc["emissions_kgco2e"],
                "peak_gpu_memory_mb": dense_nv["max_memory_mb"],
            })
            for method in METHODS:
                out = Path(f"results/replication210x3/draw_{seed}/{model}_{method}")
                actions = read_jsonl(out / "executed_actions.jsonl")
                telemetry = json.loads((out / "greenmm_telemetry.json").read_text(encoding="utf-8"))
                energy = {row["sample_id"]: float(row["gpu_energy_joule"]) for row in telemetry["per_action"]}
                energy_pool.setdefault((model, method), {}).update(
                    {f"{seed}:{key}": value for key, value in energy.items()}
                )
                strict = strict_index(model, method)
                unsafe = correct = supported = accepted = 0
                for action in actions:
                    sid = action["sample_id"]
                    selected = action["selected_budget"]
                    is_dense = selected == "dense"
                    task_correct = exact_match(action["answer"], evaluation[sid]["answer"])
                    label = strict[(sid, 432 if is_dense else int(selected))]
                    unsafe_action = (not is_dense) and label["reference_safe"] != "True"
                    supported_action = (
                        label["dense_correct"] == "True" and label["dense_reliant"] == "True"
                        if is_dense else label["combined_gold_safe"] == "True"
                    )
                    unsafe += int(unsafe_action)
                    correct += int(task_correct)
                    supported += int(supported_action)
                    accepted += int(not is_dense)
                    decisions.append({
                        "seed": seed, "model": model, "method": method,
                        "sample_id": sid, "dataset": action["dataset"],
                        "provenance": evaluation[sid]["metadata"]["replication_provenance"],
                        "selected_budget": selected, "task_correct": task_correct,
                        "unsafe_acceptance": unsafe_action, "supported_correct": supported_action,
                        "gpu_energy_joule": energy[sid], "dense_gpu_energy_joule": dense_energy[sid],
                        "latency_ms": float(action["model_latency_ms"]),
                    })
                cc = codecarbon_totals(out)
                nv = nvidia_wrapper_totals(out)
                draw_rows.append({
                    "seed": seed, "model": model, "system": method, "n": 210,
                    "dense_avoidance_rate": accepted / 210,
                    "unsafe_acceptance_rate": unsafe / 210,
                    "unsafe_acceptance_count": unsafe,
                    "accuracy": correct / 210,
                    "supported_correct_rate": supported / 210,
                    "mean_visual_tokens": statistics.fmean(float(row["visual_token_count"]) for row in actions),
                    "mean_latency_ms": statistics.fmean(float(row["model_latency_ms"]) for row in actions),
                    "mean_call_window_gpu_energy_joule": statistics.fmean(energy.values()),
                    "energy_reduction_vs_dense": 1 - statistics.fmean(energy.values()) / statistics.fmean(dense_energy.values()),
                    "nvidia_wrapper_energy_kwh": nv["energy_joule"] / 3_600_000,
                    "nvidia_wrapper_duration_h": nv["duration_s"] / 3600,
                    "codecarbon_energy_kwh": cc["energy_kwh"],
                    "codecarbon_emissions_kgco2e": cc["emissions_kgco2e"],
                    "peak_gpu_memory_mb": nv["max_memory_mb"],
                })

    aggregate: list[dict[str, Any]] = []
    for model in MODELS:
        dense_draws = [row for row in draw_rows if row["model"] == model and row["system"] == "dense"]
        for method in METHODS:
            rows = [row for row in draw_rows if row["model"] == model and row["system"] == method]
            decisions_cell = [row for row in decisions if row["model"] == model and row["method"] == method]
            unsafe = sum(int(row["unsafe_acceptance"]) for row in decisions_cell)
            energy_boot = bootstrap_reduction(
                energy_pool[(model, method)], dense_pool[model],
                seed=9000 + (0 if model == "7b" else 100) + METHODS.index(method),
            )
            draw_reductions = [float(row["energy_reduction_vs_dense"]) for row in rows]
            mean_energy = statistics.fmean(float(row["mean_call_window_gpu_energy_joule"]) for row in rows)
            accuracy = sum(int(row["task_correct"]) for row in decisions_cell) / len(decisions_cell)
            supported = sum(int(row["supported_correct"]) for row in decisions_cell) / len(decisions_cell)
            central_carbon = mean_energy / 3_600_000 * args.pue * args.carbon_kg_per_kwh
            aggregate.append({
                "model": model, "method": method, "draws": 3, "n": 630,
                "draw_energy_reductions": json.dumps(draw_reductions),
                "all_draw_energy_reductions_positive": all(value > 0 for value in draw_reductions),
                "mean_draw_energy_reduction": statistics.fmean(draw_reductions),
                "std_draw_energy_reduction": statistics.stdev(draw_reductions),
                "pooled_energy_reduction": energy_boot["energy_reduction"],
                "pooled_energy_reduction_95pct_ci": json.dumps(energy_boot["bootstrap_95pct_ci"]),
                "strong_green_replication": (
                    all(value > 0 for value in draw_reductions)
                    and energy_boot["bootstrap_95pct_ci"][0] > 0
                ),
                "unsafe_acceptance_rate": unsafe / 630,
                "unsafe_acceptance_95pct_ci": json.dumps(wilson(unsafe, 630)),
                "risk_upper_below_5pct": wilson(unsafe, 630)[1] <= 0.05,
                "accuracy": accuracy,
                "supported_correct_rate": supported,
                "dense_avoidance_rate": statistics.fmean(float(row["dense_avoidance_rate"]) for row in rows),
                "mean_visual_tokens": statistics.fmean(float(row["mean_visual_tokens"]) for row in rows),
                "mean_latency_ms": statistics.fmean(float(row["mean_latency_ms"]) for row in rows),
                "mean_dense_latency_ms": statistics.fmean(float(row["mean_latency_ms"]) for row in dense_draws),
                "mean_call_window_gpu_energy_joule": mean_energy,
                "mean_dense_call_window_gpu_energy_joule": statistics.fmean(
                    float(row["mean_call_window_gpu_energy_joule"]) for row in dense_draws
                ),
                "estimated_operational_kgco2e_per_query": central_carbon,
                "supported_correct_answers_per_kwh": supported * 3_600_000 / mean_energy,
                "supported_correct_answers_per_kgco2e": supported / central_carbon if central_carbon else None,
                "total_codecarbon_energy_kwh": sum(float(row["codecarbon_energy_kwh"]) for row in rows),
                "total_codecarbon_emissions_kgco2e": sum(float(row["codecarbon_emissions_kgco2e"]) for row in rows),
                "total_nvidia_wrapper_energy_kwh": sum(float(row["nvidia_wrapper_energy_kwh"]) for row in rows),
                "total_cloud_cost_usd_runtime_proxy": sum(float(row["nvidia_wrapper_duration_h"]) for row in rows) * args.cloud_usd_per_hour,
                "peak_gpu_memory_mb": max(float(row["peak_gpu_memory_mb"]) for row in rows),
            })

    gate_cells = []
    for row in aggregate:
        if row["strong_green_replication"] and row["risk_upper_below_5pct"]:
            level = "strong"
        elif row["pooled_energy_reduction"] > 0 and row["risk_upper_below_5pct"]:
            level = "mixed"
        else:
            level = "failed_green_or_risk"
        gate_cells.append({"model": row["model"], "method": row["method"], "replication_level": level})
    gate = {
        "artifact_type": "virel_replication_210x3_gate_r",
        "replication_complete": True,
        "protocol_sha256": sha256_path("results/replication210x3_protocol.json"),
        "completion_audit_sha256": sha256_path(args.audit),
        "cells": gate_cells,
        "strong_cells": [cell for cell in gate_cells if cell["replication_level"] == "strong"],
        "decision": "proceed_to_environmental_statistical_and_paper_analysis",
        "claim_rule": "Only strong cells support workload-replicated green deployment; mixed cells are workload-sensitive.",
    }
    breakdown_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for method in METHODS:
            cell = [row for row in decisions if row["model"] == model and row["method"] == method]
            for dimension in ("dataset", "provenance"):
                for value in sorted({str(row[dimension]) for row in cell}):
                    subset = [row for row in cell if str(row[dimension]) == value]
                    mean_energy = statistics.fmean(float(row["gpu_energy_joule"]) for row in subset)
                    mean_dense = statistics.fmean(float(row["dense_gpu_energy_joule"]) for row in subset)
                    breakdown_rows.append({
                        "model": model,
                        "method": method,
                        "dimension": dimension,
                        "value": value,
                        "n": len(subset),
                        "accuracy": sum(int(row["task_correct"]) for row in subset) / len(subset),
                        "supported_correct_rate": sum(int(row["supported_correct"]) for row in subset) / len(subset),
                        "unsafe_acceptance_rate": sum(int(row["unsafe_acceptance"]) for row in subset) / len(subset),
                        "mean_gpu_energy_joule": mean_energy,
                        "energy_reduction_vs_dense": 1 - mean_energy / mean_dense,
                        "mean_latency_ms": statistics.fmean(float(row["latency_ms"]) for row in subset),
                    })
    write_csv(output / "replication_draw_results.csv", draw_rows)
    write_csv(output / "replication_decisions.csv", decisions)
    write_csv(output / "replication_aggregate_results.csv", aggregate)
    write_csv(output / "replication_dataset_provenance_breakdown.csv", breakdown_rows)
    (output / "gate_r.json").write_bytes(canonical_json_bytes(gate))
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
