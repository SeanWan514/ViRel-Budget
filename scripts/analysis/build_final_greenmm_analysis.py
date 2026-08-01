from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_phase_b_results import _attribute_call, _read_power_samples
from virel_budget.frozen_controller import canonical_json_bytes


PARAMETERS = {"500m": 500_000_000, "7b": 7_000_000_000, "13b": 13_000_000_000}
PUE = 1.2
CARBON = 0.03283
CLOUD_USD_H = 2.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dense_development(model: str) -> dict[str, Any]:
    root = Path(f"results/phase_a_current_llava15_{model}_fastv")
    samples = _read_power_samples(root / "merged_telemetry/nvidia_smi_samples.csv")
    calls: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(root / "records.jsonl"):
        if record["method"] != "dense" or str(record["budget"]) != "full":
            continue
        sid = record["sample_id"]
        if sid in calls:
            continue
        call = (record.get("metadata") or {}).get("original_backend") or {}
        energy = _attribute_call(call, samples)["measured_energy_joule"]
        calls[sid] = {
            "correct": bool(record["is_correct"]),
            "energy": float(energy),
            "latency": (float(call["end_epoch"]) - float(call["start_epoch"])) * 1000,
        }
    if len(calls) != 1200:
        raise ValueError(f"Dense {model} development calls: {len(calls)}/1200")
    monitor = json.loads((root / "telemetry_attempts/attempt_001/nvidia_smi_summary.json").read_text(encoding="utf-8"))
    return {
        "model_scale": model,
        "model": f"LLaVA-1.5-{model.upper()}",
        "system": "dense",
        "population": "common_development_1200",
        "n": 1200,
        "parameters": PARAMETERS[model],
        "accuracy": sum(row["correct"] for row in calls.values()) / 1200,
        "supported_correct_rate": None,
        "mean_latency_ms": statistics.fmean(row["latency"] for row in calls.values()),
        "mean_gpu_energy_joule": statistics.fmean(row["energy"] for row in calls.values()),
        "peak_gpu_memory_mb": float(monitor["max_memory_used_mb"]),
    }


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    energy = float(row["mean_gpu_energy_joule"])
    accuracy = float(row["accuracy"])
    latency = float(row["mean_latency_ms"])
    params = float(row["parameters"])
    supported = row.get("supported_correct_rate")
    emissions = energy / 3_600_000 * PUE * CARBON
    row.update({
        "throughput_queries_per_second": 1000 / latency,
        "accuracy_per_billion_parameters": accuracy / (params / 1e9),
        "correct_answers_per_kwh": accuracy * 3_600_000 / energy,
        "correct_answers_per_kgco2e": accuracy / emissions,
        "supported_correct_answers_per_kwh": (
            float(supported) * 3_600_000 / energy if supported not in (None, "") else None
        ),
        "estimated_kgco2e_per_query": emissions,
        "estimated_gco2e_per_1000_queries": emissions * 1_000_000,
        "cloud_cost_usd_per_1000_queries_runtime_proxy": latency / 1000 * 1000 / 3600 * CLOUD_USD_H,
    })
    return row


def mark_pairwise_pareto(rows: list[dict[str, Any]], performance: str, resource: str, key: str) -> None:
    for target in rows:
        target[key] = not any(
            other is not target
            and float(other[performance]) >= float(target[performance])
            and float(other[resource]) <= float(target[resource])
            and (
                float(other[performance]) > float(target[performance])
                or float(other[resource]) < float(target[resource])
            )
            for other in rows
            if other.get(performance) not in (None, "") and other.get(resource) not in (None, "")
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    out = Path("results/final_greenmm_analysis")
    out.mkdir(parents=True, exist_ok=True)
    smol = json.loads((out / "smolvlm500m_dense_audit.json").read_text(encoding="utf-8"))
    dense_rows = [
        enrich({
            "model_scale": "500m",
            "model": "SmolVLM-500M-Instruct",
            "system": "dense",
            "population": "common_development_1200",
            "n": 1200,
            "parameters": PARAMETERS["500m"],
            "accuracy": smol["unconditional_accuracy"],
            "supported_correct_rate": None,
            "mean_latency_ms": smol["mean_original_call_latency_ms"],
            "mean_gpu_energy_joule": smol["mean_original_call_gpu_energy_joule"],
            "peak_gpu_memory_mb": smol["peak_gpu_memory_mb"],
            "scope_note": "Dense SLM verification; complete inference with disclosed post-inference analysis failure.",
        }),
        enrich(dense_development("7b")),
        enrich(dense_development("13b")),
    ]
    for performance in ("accuracy",):
        for resource in ("mean_gpu_energy_joule", "mean_latency_ms", "peak_gpu_memory_mb"):
            mark_pairwise_pareto(dense_rows, performance, resource, f"pareto_{performance}_vs_{resource}")
    write_csv(out / "dense_model_scale_pareto.csv", dense_rows)

    replication = read_csv(Path("results/replication210x3_analysis/replication_aggregate_results.csv"))
    controller_rows = []
    for row in replication:
        model = row["model"]
        controller_rows.append(enrich({
            "model_scale": model,
            "model": f"LLaVA-1.5-{model.upper()}",
            "system": row["method"],
            "population": "replication_630",
            "n": 630,
            "parameters": PARAMETERS[model],
            "accuracy": float(row["accuracy"]),
            "supported_correct_rate": float(row["supported_correct_rate"]),
            "mean_latency_ms": float(row["mean_latency_ms"]),
            "mean_gpu_energy_joule": float(row["mean_call_window_gpu_energy_joule"]),
            "peak_gpu_memory_mb": float(row["peak_gpu_memory_mb"]),
            "dense_avoidance_rate": float(row["dense_avoidance_rate"]),
            "unsafe_acceptance_rate": float(row["unsafe_acceptance_rate"]),
            "pooled_energy_reduction": float(row["pooled_energy_reduction"]),
            "strong_green_replication": row["strong_green_replication"] == "True",
        }))
    for performance in ("accuracy", "supported_correct_rate"):
        for resource in ("mean_gpu_energy_joule", "mean_latency_ms", "peak_gpu_memory_mb"):
            mark_pairwise_pareto(controller_rows, performance, resource, f"pareto_{performance}_vs_{resource}")
    write_csv(out / "controller_resource_pareto.csv", controller_rows)
    report = {
        "artifact_type": "virel_final_greenmm_pareto_analysis",
        "dense_model_scale_points": dense_rows,
        "controller_points": controller_rows,
        "assumptions": {
            "pue": PUE,
            "carbon_intensity_kgco2e_per_kwh": CARBON,
            "cloud_usd_per_hour": CLOUD_USD_H,
        },
        "boundaries": [
            "The dense 500M/7B/13B frontier uses the common 1,200 development cases.",
            "The controller frontier uses the three 210-case replication draws.",
            "Pairwise Pareto frontiers are reported separately by objective pair; no universal scalar score is imposed.",
            "Carbon is a location/PUE estimate derived from measured GPU energy, not direct carbon measurement.",
        ],
    }
    (out / "final_greenmm_analysis.json").write_bytes(canonical_json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
