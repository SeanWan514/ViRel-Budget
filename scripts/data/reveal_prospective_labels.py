from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.datasets.jsonl import write_jsonl  # noqa: E402
from virel_budget.frozen_controller import canonical_json_bytes, sha256_path  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reveal prospective labels only after all six single-action executions complete."
    )
    parser.add_argument("--frozen-dir", default="data/prospective_900_frozen")
    parser.add_argument("--execution-dir", action="append", required=True)
    parser.add_argument("--output-dir", default="data/prospective_900_revealed")
    args = parser.parse_args()
    if len(args.execution_dir) != 6:
        raise ValueError("Exactly six model/pruner execution directories are required before reveal")
    frozen = Path(args.frozen_dir)
    deployment_path = frozen / "deployment_samples.jsonl"
    labels_path = frozen / "hidden_evaluation_labels.jsonl"
    expected_sha = sha256_path(deployment_path)
    execution_audit = []
    for name in args.execution_dir:
        directory = Path(name)
        manifest = json.loads((directory / "execution_manifest.json").read_text(encoding="utf-8"))
        actions = read_jsonl(directory / "executed_actions.jsonl")
        if manifest.get("dry_run") or manifest.get("executed_count") != 900:
            raise ValueError(f"Incomplete or dry-run execution: {directory}")
        if manifest.get("maximum_backend_invocations_per_query") != 1:
            raise ValueError(f"Non-single-action execution: {directory}")
        if manifest.get("deployment_samples_sha256") != expected_sha:
            raise ValueError(f"Deployment manifest mismatch: {directory}")
        if len(actions) != 900 or len({row["sample_id"] for row in actions}) != 900:
            raise ValueError(f"Executed action set is incomplete or duplicated: {directory}")
        execution_audit.append(
            {
                "directory": str(directory),
                "manifest_sha256": sha256_path(directory / "execution_manifest.json"),
                "actions_sha256": sha256_path(directory / "executed_actions.jsonl"),
            }
        )
    deployment = read_jsonl(deployment_path)
    hidden = {row["sample_id"]: row for row in read_jsonl(labels_path)}
    if set(hidden) != {row["sample_id"] for row in deployment}:
        raise ValueError("Frozen deployment/hidden-label sample IDs differ")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite revealed dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in deployment:
        label = hidden[row["sample_id"]]
        image = (frozen / row["image_path"]).resolve()
        counterfactual = label.get("counterfactual_image_path")
        rows.append(
            {
                **row,
                "image_path": os.path.relpath(image, output.resolve()),
                "answer": label["gold_answer"],
                "counterfactual_image_path": (
                    os.path.relpath((frozen / counterfactual).resolve(), output.resolve())
                    if counterfactual
                    else None
                ),
                "metadata": label.get("original_metadata", row.get("metadata", {})),
            }
        )
    samples = output / "samples.jsonl"
    write_jsonl(samples, rows)
    manifest = {
        "status": "revealed_after_six_complete_single_action_executions",
        "sample_count": len(rows),
        "source_frozen_manifest_sha256": sha256_path(frozen / "frozen_manifest.json"),
        "samples_sha256": sha256_path(samples),
        "execution_audit": execution_audit,
    }
    (output / "reveal_manifest.json").write_bytes(canonical_json_bytes(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
