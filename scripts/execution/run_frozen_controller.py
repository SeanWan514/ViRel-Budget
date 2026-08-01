from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.config import load_config
from virel_budget.datasets.deployment_jsonl import load_deployment_samples
from virel_budget.datasets.jsonl import write_jsonl
from virel_budget.frozen_controller import (
    FrozenBudgetController,
    canonical_json_bytes,
    sha256_path,
)
from virel_budget.pipeline import _make_backend


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the frozen one-action ViRel controller.")
    parser.add_argument("--config", required=True, help="Backend configuration JSON.")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--checksum", required=True)
    parser.add_argument("--deployment-samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup-samples")
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial execution after verifying its frozen plan and completed prefix.",
    )
    parser.add_argument(
        "--scope-batched-order",
        action="store_true",
        help=(
            "For SCOPE only, execute frozen dense actions before frozen pruned actions. "
            "Decisions remain unchanged; this avoids reversing SCOPE's process-global wrapper."
        ),
    )
    return parser.parse_args()


def _controller_input(sample: Any) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "question": sample.question,
        "options": sample.options,
    }


def _cuda_synchronize() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_executed_prefix(
    executed: list[dict[str, Any]],
    planned: list[dict[str, Any]],
) -> None:
    if len(executed) > len(planned):
        raise ValueError("Executed checkpoint is longer than the frozen plan")
    planned_by_id = {str(row["sample_id"]): row for row in planned}
    if len(planned_by_id) != len(planned):
        raise ValueError("Frozen plan contains duplicate sample IDs")
    seen: set[str] = set()
    for index, row in enumerate(executed):
        sample_id = str(row.get("sample_id"))
        if sample_id in seen:
            raise ValueError(f"Duplicate executed sample ID: {sample_id}")
        if sample_id not in planned_by_id:
            raise ValueError(f"Executed sample is absent from frozen plan: {sample_id}")
        seen.add(sample_id)
        plan = planned_by_id[sample_id]
        for key in ("sample_id", "dataset", "selected_budget", "method"):
            if row.get(key) != plan.get(key):
                raise ValueError(
                    f"Executed checkpoint diverges from frozen plan at row {index}, field {key}"
                )
        if int(row.get("backend_invocation_count", 0)) != 1:
            raise ValueError(f"Invalid invocation count in executed checkpoint row {index}")


def _validate_recomputed_plan(
    existing: list[dict[str, Any]],
    recomputed: list[dict[str, Any]],
) -> None:
    if len(existing) != len(recomputed):
        raise ValueError("Existing frozen plan length differs from recomputed decisions")
    decision_fields = (
        "sample_id",
        "dataset",
        "selected_budget",
        "probabilities",
        "method",
    )
    for index, (left, right) in enumerate(zip(existing, recomputed, strict=True)):
        for key in decision_fields:
            if left.get(key) != right.get(key):
                raise ValueError(
                    f"Existing frozen plan differs from recomputed decision at row {index}, field {key}"
                )


def main() -> None:
    args = _arguments()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    controller = FrozenBudgetController.load(args.controller, args.checksum)
    samples = load_deployment_samples(args.deployment_samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No deployment samples loaded")
    if args.warmup_count and not args.warmup_samples:
        raise ValueError("--warmup-samples is required when --warmup-count is nonzero")
    method = str(controller.artifact["method"])
    configured_methods = [str(value) for value in config["pruning"]["methods"]]
    if configured_methods != [method]:
        raise ValueError(
            f"Backend config methods {configured_methods} do not match frozen controller {method}"
        )
    if args.scope_batched_order and method != "scope":
        raise ValueError("--scope-batched-order is valid only for a SCOPE controller")

    planning_started_epoch = time.time()
    planned: list[dict[str, Any]] = []
    for sample in samples:
        decision = controller.decide(_controller_input(sample))
        planned.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "selected_budget": decision.selected_budget,
                "probabilities": decision.probabilities,
                "controller_latency_ms": decision.controller_latency_ms,
                "method": method,
            }
        )
    planned_path = output_dir / "planned_actions_before_inference.jsonl"
    if planned_path.exists():
        existing_plan = _read_jsonl(planned_path)
        _validate_recomputed_plan(existing_plan, planned)
        # Keep the originally frozen planning latency and byte-level checksum.
        planned = existing_plan
    else:
        write_jsonl(planned_path, planned)
    plan_checksum = sha256_path(planned_path)
    plan_checksum_path = output_dir / "planned_actions_before_inference.sha256"
    expected_checksum_text = f"{plan_checksum}  {planned_path.name}\n"
    if plan_checksum_path.exists():
        if plan_checksum_path.read_text(encoding="utf-8") != expected_checksum_text:
            raise ValueError("Existing frozen-plan checksum file is inconsistent")
    else:
        plan_checksum_path.write_text(expected_checksum_text, encoding="utf-8")
    planning_ended_epoch = time.time()
    backend_load_started_epoch = None
    backend_load_ended_epoch = None
    warmup_started_epoch = None
    warmup_ended_epoch = None
    inference_started_epoch = None
    inference_ended_epoch = None
    resumed_from_count = 0
    executed_this_attempt_count = 0
    if args.dry_run:
        executed: list[dict[str, Any]] = []
    else:
        executed_path = output_dir / "executed_actions.jsonl"
        executed = _read_jsonl(executed_path)
        _validate_executed_prefix(executed, planned)
        resumed_from_count = len(executed)
        backend_load_started_epoch = time.time()
        backend = _make_backend(config)
        backend_load_ended_epoch = time.time()

        if args.warmup_count:
            warmup_samples = load_deployment_samples(args.warmup_samples)
            if not warmup_samples:
                raise ValueError("No warm-up samples loaded")
            warmup_started_epoch = time.time()
            for index in range(args.warmup_count):
                warmup_sample = warmup_samples[index % len(warmup_samples)]
                warmup_decision = controller.decide(_controller_input(warmup_sample))
                # SCOPE installs a process-global pruning wrapper and cannot return
                # to dense mode. A batched dense-before-SCOPE execution must
                # therefore warm up in dense mode before any pruned action.
                selected = "dense" if args.scope_batched_order else warmup_decision.selected_budget
                backend.score_options(
                    warmup_sample,
                    warmup_sample.image_path,
                    "dense" if selected == "dense" else method,
                    "full" if selected == "dense" else int(selected),
                    int(config["seed"]),
                )
            _cuda_synchronize()
            warmup_ended_epoch = time.time()
        _cuda_synchronize()
        inference_started_epoch = time.time()
        try:
            completed_ids = {str(row["sample_id"]) for row in executed}
            execution_pairs = [
                (sample, plan)
                for sample, plan in zip(samples, planned, strict=True)
                if sample.sample_id not in completed_ids
            ]
            if args.scope_batched_order:
                execution_pairs.sort(key=lambda pair: pair[1]["selected_budget"] != "dense")
            for sample, plan in execution_pairs:
                selected = plan["selected_budget"]
                inference_method = "dense" if selected == "dense" else method
                inference_budget = "full" if selected == "dense" else int(selected)
                # Exactly one backend invocation is permitted for each query.
                _cuda_synchronize()
                call_started_epoch = time.time()
                result = backend.score_options(
                    sample,
                    sample.image_path,
                    inference_method,
                    inference_budget,
                    int(config["seed"]),
                )
                _cuda_synchronize()
                call_ended_epoch = time.time()
                executed_row = {
                    **plan,
                    "answer": result.answer,
                    "executed_method": inference_method,
                    "executed_budget": inference_budget,
                    "model_latency_ms": float(result.measured_latency_ms or result.latency_ms),
                    "visual_token_count": int(result.token_count),
                    "backend_invocation_count": 1,
                    "call_started_epoch": call_started_epoch,
                    "call_ended_epoch": call_ended_epoch,
                }
                _append_durable(executed_path, executed_row)
                executed.append(executed_row)
                executed_this_attempt_count += 1
        finally:
            inference_ended_epoch = time.time()
            backend.close()
    manifest = {
        "artifact_type": "virel_prospective_single_action_execution",
        "dry_run": bool(args.dry_run),
        "sample_count": len(samples),
        "executed_count": len(executed),
        "resumed_from_count": resumed_from_count,
        "executed_this_attempt_count": executed_this_attempt_count,
        "maximum_backend_invocations_per_query": 0 if args.dry_run else 1,
        "resident_backend_replica_count": 0 if args.dry_run else 1,
        "scope_batched_order": bool(args.scope_batched_order),
        "execution_order_rule": (
            "frozen_dense_actions_then_frozen_scope_actions"
            if args.scope_batched_order
            else "frozen_manifest_order"
        ),
        "controller_sha256": sha256_path(args.controller),
        "controller_checksum_file_sha256": sha256_path(args.checksum),
        "deployment_samples_sha256": sha256_path(args.deployment_samples),
        "planned_actions_sha256": plan_checksum,
        "backend_config_sha256": sha256_path(args.config),
        "gold_or_intervention_labels_loaded": False,
        "planning_started_epoch": planning_started_epoch,
        "planning_ended_epoch": planning_ended_epoch,
        "backend_load_started_epoch": backend_load_started_epoch,
        "backend_load_ended_epoch": backend_load_ended_epoch,
        "warmup_count": int(args.warmup_count),
        "warmup_samples_sha256": (
            sha256_path(args.warmup_samples) if args.warmup_samples else None
        ),
        "warmup_started_epoch": warmup_started_epoch,
        "warmup_ended_epoch": warmup_ended_epoch,
        "inference_started_epoch": inference_started_epoch,
        "inference_ended_epoch": inference_ended_epoch,
        "python": platform.python_version(),
        "completed_epoch": time.time(),
    }
    (output_dir / "execution_manifest.json").write_bytes(canonical_json_bytes(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
