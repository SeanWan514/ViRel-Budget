from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.frozen_controller import canonical_json_bytes


def _integrate(samples: list[tuple[float, float]], start: float, end: float) -> float:
    points = [(epoch, watts) for epoch, watts in samples if start <= epoch <= end]
    before = [item for item in samples if item[0] < start]
    after = [item for item in samples if item[0] > end]
    if before:
        points.insert(0, (start, before[-1][1]))
    if after:
        points.append((end, after[0][1]))
    if len(points) < 2:
        raise ValueError(f"Insufficient telemetry samples in [{start}, {end}]")
    energy = 0.0
    for (left_t, left_w), (right_t, right_w) in zip(points, points[1:]):
        clipped_left = max(left_t, start)
        clipped_right = min(right_t, end)
        if clipped_right > clipped_left:
            energy += (clipped_right - clipped_left) * (left_w + right_w) / 2.0
    return energy


def _read_samples(path: Path) -> list[tuple[float, float]]:
    output = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.append((float(row["epoch"]), float(row["power_w"])))
    return sorted(output)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate telemetry over frozen execution phase boundaries.")
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--executed-actions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    samples = _read_samples(Path(args.telemetry))
    manifest = json.loads(Path(args.execution_manifest).read_text(encoding="utf-8"))
    actions = _read_jsonl(Path(args.executed_actions))
    if manifest["executed_count"] != len(actions):
        raise ValueError("Execution manifest/action count mismatch")
    inference_start = float(manifest["inference_started_epoch"])
    inference_end = float(manifest["inference_ended_epoch"])
    inference_energy = _integrate(samples, inference_start, inference_end)
    resumed_from_count = int(manifest.get("resumed_from_count", 0))
    executed_this_attempt = int(
        manifest.get("executed_this_attempt_count", len(actions) - resumed_from_count)
    )
    per_action = []
    for action in actions:
        energy = _integrate(
            samples,
            float(action["call_started_epoch"]),
            float(action["call_ended_epoch"]),
        )
        per_action.append(
            {
                "sample_id": action["sample_id"],
                "selected_budget": action["selected_budget"],
                "gpu_energy_joule": energy,
            }
        )
    call_energy = sum(row["gpu_energy_joule"] for row in per_action)
    controller_latencies = [float(row["controller_latency_ms"]) for row in actions]
    output = {
        "boundary": "GPU device energy integrated over frozen single-action inference window",
        "sample_count": len(actions),
        "final_attempt_inference_window_energy_joule": inference_energy,
        "final_attempt_query_count": executed_this_attempt,
        "final_attempt_inference_window_joule_per_query": (
            inference_energy / executed_this_attempt if executed_this_attempt else None
        ),
        "resumed_from_count": resumed_from_count,
        "complete_execution_block_energy_available": resumed_from_count == 0,
        "complete_execution_inference_window_energy_joule": (
            inference_energy if resumed_from_count == 0 else None
        ),
        "complete_execution_inference_window_joule_per_query": (
            inference_energy / len(actions) if resumed_from_count == 0 else None
        ),
        "summed_call_window_energy_joule": call_energy,
        "summed_call_window_joule_per_query": call_energy / len(actions),
        "mean_controller_cpu_latency_ms": sum(controller_latencies) / len(controller_latencies),
        "telemetry_interval_target_ms": 200,
        "per_action": per_action,
        "caution": (
            "Per-action integration is noisy for short calls. A complete block-level value is "
            "available only for executions completed in one attempt; resumed executions use the "
            "summed call-window value across merged attempts."
        ),
    }
    Path(args.output).write_bytes(canonical_json_bytes(output))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
