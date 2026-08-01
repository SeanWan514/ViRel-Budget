from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path("results/verification_smolvlm500m_dense_paper1200")


def main() -> int:
    rows = [
        json.loads(line)
        for line in (ROOT / "records_with_measured_energy_optionfix.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    first_by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        first_by_sample.setdefault(str(row["sample_id"]), row)
    complete_inference = (
        len(rows) == 4000
        and len(first_by_sample) == 1200
        and {str(row["method"]) for row in rows} == {"dense"}
        and {str(row["budget"]) for row in rows} == {"full"}
    )
    monitor = json.loads((ROOT / "hardware_monitor/nvidia_smi_summary.json").read_text(encoding="utf-8"))
    codecarbon = list(csv.DictReader((ROOT / "codecarbon/emissions.csv").open(encoding="utf-8")))[-1]
    command_status = int((ROOT / "exit_status.txt").read_text(encoding="utf-8").strip())
    audit = {
        "artifact_type": "virel_smolvlm500m_dense_pareto_audit",
        "model": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "parameter_count": 500_000_000,
        "scope": "Dense-only secondary model-size/Pareto verification on the common 1,200 development cases.",
        "records": len(rows),
        "unique_samples": len(first_by_sample),
        "complete_inference_records": complete_inference,
        "post_inference_command_status": command_status,
        "post_inference_failure": (
            "The dense inference grid completed; legacy report generation then indexed an empty "
            "pruning budget schedule. Reanalysis artifacts were generated from the complete records."
        ),
        "unconditional_accuracy": sum(bool(row["is_correct"]) for row in first_by_sample.values()) / 1200,
        "mean_original_call_latency_ms": statistics.fmean(
            float(row["measured_latency_ms"]) for row in first_by_sample.values()
        ),
        "mean_original_call_gpu_energy_joule": statistics.fmean(
            float(row["measured_energy_joule"]) for row in first_by_sample.values()
        ),
        "mean_visual_tokens": statistics.fmean(
            float(row["token_count"]) for row in first_by_sample.values()
        ),
        "wrapper_duration_s": float(monitor["duration_s"]),
        "wrapper_gpu_energy_kwh": float(monitor["measured_energy_joule"]) / 3_600_000,
        "peak_gpu_memory_mb": float(monitor["max_memory_used_mb"]),
        "codecarbon_energy_kwh": float(codecarbon["energy_consumed"]),
        "codecarbon_emissions_kgco2e_pue1": float(codecarbon["emissions"]),
        "codecarbon_country": codecarbon["country_name"],
        "codecarbon_version": codecarbon["codecarbon_version"],
        "claim_boundary": (
            "Valid as a dense SLM resource/performance point. It is not a ViRel controller, "
            "pruning-method comparison, or prospective 900-case result."
        ),
    }
    out = Path("results/final_greenmm_analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "smolvlm500m_dense_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if complete_inference else 1


if __name__ == "__main__":
    raise SystemExit(main())
