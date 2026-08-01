from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.frozen_controller import canonical_json_bytes


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def last_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def family(path: str) -> str:
    if "replication210x3_smoke" in path:
        return "replication_smoke"
    if "replication210x3" in path:
        return "replication_210x3"
    if "robustness10_" in path:
        return "legacy_120x3"
    if "phase_a_current" in path:
        return "phase_a_1200"
    if "prospective900_execution" in path:
        return "phase_b_controller_execution"
    if "prospective900_llava" in path:
        return "phase_b_evaluation_grid"
    if "feature_pilot" in path:
        return "feature_pilot"
    if "frozen_controller_gpu_smoke" in path:
        return "controller_smoke"
    if "verification_smolvlm" in path:
        return "smolvlm_verification"
    if "paper1200_llava" in path:
        return "original_1200"
    return "other_recorded"


def codecarbon_for_summary(summary: Path) -> Path | None:
    candidates = [
        summary.parent / "codecarbon" / "emissions.csv",
        summary.parent.parent / "codecarbon" / "emissions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def recovered_failed_smoke(root: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    """Recover the documented failed smoke attempt when only its combined log survived."""
    path = root / "replication210x3_smoke/failed_attempts/13b_scope_initial/combined.log"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")

    def captured(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    energy_j = captured(r'"measured_energy_joule":\s*([0-9.eE+-]+)')
    duration_s = captured(r'"duration_s":\s*([0-9.eE+-]+)')
    emissions = captured(r'"codecarbon_emissions_kgco2e":\s*([0-9.eE+-]+)')
    if energy_j is None or duration_s is None:
        return None
    gpu_kwh = energy_j / 3_600_000.0
    central_facility = gpu_kwh * args.central_pue
    return {
        "run_id": hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16],
        "family": "replication_smoke_failed",
        "nvidia_summary": str(path),
        "measurement_tier": "tier2_recovered_from_preserved_combined_log",
        "duration_s": duration_s,
        "duration_h": duration_s / 3600.0,
        "mean_power_w": energy_j / duration_s,
        "max_power_w": None,
        "mean_gpu_utilization_pct": None,
        "max_memory_used_mb": None,
        "measured_gpu_energy_kwh": gpu_kwh,
        "codecarbon_available": emissions is not None,
        "codecarbon_path": str(path),
        "codecarbon_energy_kwh": emissions / args.central_carbon_kg_per_kwh if emissions else None,
        "codecarbon_gpu_energy_kwh": None,
        "codecarbon_cpu_energy_kwh": None,
        "codecarbon_ram_energy_kwh": None,
        "codecarbon_emissions_kgco2e_pue1": emissions,
        "codecarbon_country": "Iceland",
        "codecarbon_region": "",
        "codecarbon_version": "3.2.9",
        "facility_gpu_energy_kwh_central": central_facility,
        "gpu_emissions_kgco2e_low": gpu_kwh * args.low_pue * args.low_carbon_kg_per_kwh,
        "gpu_emissions_kgco2e_central": central_facility * args.central_carbon_kg_per_kwh,
        "gpu_emissions_kgco2e_high": gpu_kwh * args.high_pue * args.high_carbon_kg_per_kwh,
        "cloud_cost_usd_runtime_proxy": duration_s / 3600.0 * args.cloud_usd_per_hour,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/whole_experiment_carbon")
    parser.add_argument("--central-pue", type=float, default=1.2)
    parser.add_argument("--low-pue", type=float, default=1.1)
    parser.add_argument("--high-pue", type=float, default=1.4)
    parser.add_argument("--central-carbon-kg-per-kwh", type=float, default=0.03283)
    parser.add_argument("--low-carbon-kg-per-kwh", type=float, default=0.015)
    parser.add_argument("--high-carbon-kg-per-kwh", type=float, default=0.08)
    parser.add_argument("--cloud-usd-per-hour", type=float, default=2.0)
    args = parser.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    excluded = []
    for path in sorted(root.rglob("nvidia_smi_summary.json")):
        relative = str(path)
        # This directory is a known duplicate transfer artifact, not an independent run.
        if "_stray_flattened_pull_" in relative:
            excluded.append({"path": relative, "reason": "duplicate transfer artifact"})
            continue
        record = read_json(path)
        energy_j = number(record.get("measured_energy_joule"))
        duration_s = number(record.get("duration_s"))
        if energy_j is None or duration_s is None:
            excluded.append({"path": relative, "reason": "missing measured energy or duration"})
            continue
        cc_path = codecarbon_for_summary(path)
        cc = last_csv(cc_path) if cc_path else {}
        gpu_kwh = energy_j / 3_600_000.0
        central_facility = gpu_kwh * args.central_pue
        row = {
            "run_id": hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16],
            "family": family(relative),
            "nvidia_summary": relative,
            "measurement_tier": "tier1_direct_gpu",
            "duration_s": duration_s,
            "duration_h": duration_s / 3600.0,
            "mean_power_w": number(record.get("mean_power_w")),
            "max_power_w": number(record.get("max_power_w")),
            "mean_gpu_utilization_pct": number(record.get("mean_gpu_utilization_pct")),
            "max_memory_used_mb": number(record.get("max_memory_used_mb")),
            "measured_gpu_energy_kwh": gpu_kwh,
            "codecarbon_available": bool(cc),
            "codecarbon_path": str(cc_path) if cc_path else "",
            "codecarbon_energy_kwh": number(cc.get("energy_consumed")),
            "codecarbon_gpu_energy_kwh": number(cc.get("gpu_energy")),
            "codecarbon_cpu_energy_kwh": number(cc.get("cpu_energy")),
            "codecarbon_ram_energy_kwh": number(cc.get("ram_energy")),
            "codecarbon_emissions_kgco2e_pue1": number(cc.get("emissions")),
            "codecarbon_country": cc.get("country_name", ""),
            "codecarbon_region": cc.get("region", ""),
            "codecarbon_version": cc.get("codecarbon_version", ""),
            "facility_gpu_energy_kwh_central": central_facility,
            "gpu_emissions_kgco2e_low": gpu_kwh * args.low_pue * args.low_carbon_kg_per_kwh,
            "gpu_emissions_kgco2e_central": central_facility * args.central_carbon_kg_per_kwh,
            "gpu_emissions_kgco2e_high": gpu_kwh * args.high_pue * args.high_carbon_kg_per_kwh,
            "cloud_cost_usd_runtime_proxy": duration_s / 3600.0 * args.cloud_usd_per_hour,
        }
        rows.append(row)
    failed_smoke = recovered_failed_smoke(root, args)
    if failed_smoke:
        rows.append(failed_smoke)

    fields = list(rows[0]) if rows else []
    with (out / "experiment_run_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_family: dict[str, dict[str, float]] = {}
    for row in rows:
        agg = by_family.setdefault(row["family"], {
            "runs": 0,
            "duration_h": 0.0,
            "measured_gpu_energy_kwh": 0.0,
            "gpu_emissions_kgco2e_low": 0.0,
            "gpu_emissions_kgco2e_central": 0.0,
            "gpu_emissions_kgco2e_high": 0.0,
            "cloud_cost_usd_runtime_proxy": 0.0,
            "codecarbon_runs": 0,
            "codecarbon_energy_kwh": 0.0,
        })
        agg["runs"] += 1
        for key in (
            "duration_h", "measured_gpu_energy_kwh", "gpu_emissions_kgco2e_low",
            "gpu_emissions_kgco2e_central", "gpu_emissions_kgco2e_high",
            "cloud_cost_usd_runtime_proxy",
        ):
            agg[key] += float(row[key])
        if row["codecarbon_available"]:
            agg["codecarbon_runs"] += 1
            agg["codecarbon_energy_kwh"] += float(row["codecarbon_energy_kwh"] or 0)
    total = {
        key: sum(float(value.get(key, 0)) for value in by_family.values())
        for key in (
            "runs", "duration_h", "measured_gpu_energy_kwh",
            "gpu_emissions_kgco2e_low", "gpu_emissions_kgco2e_central",
            "gpu_emissions_kgco2e_high", "cloud_cost_usd_runtime_proxy",
            "codecarbon_runs", "codecarbon_energy_kwh",
        )
    }
    report = {
        "artifact_type": "virel_whole_experiment_carbon_ledger",
        "scope": "All unique NVIDIA wrapper summaries recoverable in the repository, plus explicitly tiered preserved failed-run evidence.",
        "totals": total,
        "by_family": by_family,
        "excluded": excluded,
        "assumptions": {
            "hardware": "one NVIDIA RTX PRO 6000 Blackwell Server Edition unless per-run metadata states otherwise",
            "region": "RunPod EUR-IS-1; CodeCarbon geolocation reports Iceland/capital region",
            "central_pue": args.central_pue,
            "pue_sensitivity": [args.low_pue, args.high_pue],
            "central_carbon_intensity_kgco2e_per_kwh": args.central_carbon_kg_per_kwh,
            "carbon_intensity_sensitivity_kgco2e_per_kwh": [
                args.low_carbon_kg_per_kwh, args.high_carbon_kg_per_kwh
            ],
            "cloud_price_usd_per_hour": args.cloud_usd_per_hour,
        },
        "boundaries": [
            "Direct NVIDIA figures are GPU-wrapper energy and include model load/warm-up when the wrapper surrounded them.",
            "CodeCarbon is available only for instrumented runs and reports PUE=1.0 by default.",
            "CodeCarbon detected host-wide CPU/RAM resources larger than the pod allocation; its CPU/RAM totals are secondary estimates, not direct pod measurements.",
            "The location/PUE-adjusted GPU estimate is the consistent whole-program primary footprint.",
            "Pod idle time outside recorded wrappers, embodied carbon, networking, and storage are excluded unless separately reconstructed.",
            "One failed replication smoke attempt is Tier 2 because its numeric NVIDIA and CodeCarbon summaries survive only in the preserved combined log.",
            "The cloud-cost field is a runtime proxy, not a billing statement.",
        ],
    }
    (out / "whole_experiment_carbon_summary.json").write_bytes(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
