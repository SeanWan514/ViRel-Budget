from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.config import load_config, resolve_path
from virel_budget.datasets.jsonl import write_jsonl
from virel_budget.images import materialize_interventions
from virel_budget.pipeline import (
    _available_interventions_for_sample,
    _evaluate_one,
    _load_samples,
    _make_backend,
    _resolve_intervention_specs,
    analyze_records,
)
from virel_budget.schema import EvalRecord


def _key(record: EvalRecord) -> tuple[str, str, str, str]:
    return record.sample_id, record.method, str(record.budget), record.intervention


def _load_checkpoint(path: Path) -> list[EvalRecord]:
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.append(EvalRecord(**json.loads(line)))
    keys = [_key(record) for record in output]
    if len(keys) != len(set(keys)):
        raise ValueError("Checkpoint contains duplicate evaluation keys")
    return output


def _append(path: Path, record: EvalRecord) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a restart-safe dense/pruning intervention grid.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    guard = config.get("execution_guard")
    if guard:
        guard_path = resolve_path(config, guard["path"])
        if guard_path is None or not guard_path.exists():
            raise PermissionError(
                f"Execution is gated. Missing approval artifact: {guard_path}"
            )
        approval = json.loads(guard_path.read_text(encoding="utf-8"))
        if (
            approval.get("phase") != guard["required_phase"]
            or approval.get("approved") is not guard["required_approved"]
        ):
            raise PermissionError(
                f"Execution approval is invalid for {guard['required_phase']}: {guard_path}"
            )
    output = resolve_path(config, config["outputs"]["dir"])
    assert output is not None
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "records_checkpoint.jsonl"
    records = _load_checkpoint(checkpoint)
    completed = {_key(record) for record in records}
    samples = _load_samples(config)
    intervention_specs = _resolve_intervention_specs(config)
    interventions = materialize_interventions(
        samples,
        intervention_specs,
        output / "interventions",
        int(config["seed"]),
        reuse_existing=bool(config.get("reuse_existing_interventions", False)),
    )
    sample_interventions = {
        sample.sample_id: _available_interventions_for_sample(
            sample, interventions, intervention_specs
        )
        for sample in samples
    }
    backend = _make_backend(config)
    original_cache: dict[tuple[str, str, str], Any] = {}
    dense_budget = config["pruning"].get("dense_budget", "full")
    try:
        for sample in samples:
            for intervention_name, intervention in sample_interventions[sample.sample_id]:
                key = (sample.sample_id, "dense", str(dense_budget), intervention_name)
                if key in completed:
                    continue
                record = _evaluate_one(
                    config,
                    sample,
                    backend,
                    "dense",
                    dense_budget,
                    intervention_name,
                    intervention.path,
                    dense_vem=None,
                    seed=int(config["seed"]),
                    original_cache=original_cache,
                )
                record = EvalRecord(
                    **{
                        **record.__dict__,
                        "dense_vem": record.vem,
                        "reliance_retention": 1.0 if abs(record.vem) > 1e-9 else None,
                        "delta_vem": 0.0,
                    }
                )
                records.append(record)
                completed.add(key)
                _append(checkpoint, record)
                print(json.dumps({"phase": "dense", "completed_records": len(records), "key": key}), flush=True)

        dense_vem = {
            (record.sample_id, record.intervention): record.vem
            for record in records
            if record.method == "dense"
        }
        for method in config["pruning"]["methods"]:
            for budget in config["pruning"]["budget_schedule"]:
                for sample in samples:
                    for intervention_name, intervention in sample_interventions[sample.sample_id]:
                        key = (sample.sample_id, str(method), str(budget), intervention_name)
                        if key in completed:
                            continue
                        record = _evaluate_one(
                            config,
                            sample,
                            backend,
                            str(method),
                            budget,
                            intervention_name,
                            intervention.path,
                            dense_vem=dense_vem[(sample.sample_id, intervention_name)],
                            seed=int(config["seed"]),
                            original_cache=original_cache,
                        )
                        records.append(record)
                        completed.add(key)
                        _append(checkpoint, record)
                        print(
                            json.dumps(
                                {
                                    "phase": "pruned",
                                    "method": method,
                                    "budget": budget,
                                    "completed_records": len(records),
                                    "key": key,
                                }
                            ),
                            flush=True,
                        )
    finally:
        backend.close()

    expected = sum(len(sample_interventions[sample.sample_id]) for sample in samples) * (
        1 + len(config["pruning"]["methods"]) * len(config["pruning"]["budget_schedule"])
    )
    if len(records) != expected:
        raise RuntimeError(f"Grid incomplete: expected {expected} records, found {len(records)}")
    records_path = output / "records.jsonl"
    write_jsonl(records_path, [asdict(record) for record in records])
    completion = {
        "status": "complete",
        "sample_count": len(samples),
        "record_count": len(records),
        "expected_record_count": expected,
        "checkpoint": str(checkpoint),
        "config": str(Path(args.config)),
    }
    (output / "grid_completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.analyze:
        analyze_records(config, records, output)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
