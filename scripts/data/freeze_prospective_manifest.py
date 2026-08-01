from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.datasets.jsonl import load_jsonl_samples, write_jsonl
from virel_budget.frozen_controller import canonical_json_bytes, sha256_path


def _image_hash(path: Path) -> str:
    return sha256_path(path)


def _question_key(question: str) -> str:
    return " ".join(question.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze group-isolated prospective inputs and separate hidden labels before inference."
    )
    parser.add_argument("--development", default="data/paper_1200/samples.jsonl")
    parser.add_argument("--prospective", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    development_path = Path(args.development)
    prospective_path = Path(args.prospective)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty frozen manifest directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    development = load_jsonl_samples(development_path, dataset_name="development")
    prospective = load_jsonl_samples(prospective_path, dataset_name="prospective")
    development_images = {_image_hash(sample.image_path) for sample in development}
    development_question_counts = Counter(_question_key(sample.question) for sample in development)
    prospective_question_counts = Counter(_question_key(sample.question) for sample in prospective)
    seen_ids: set[str] = set()
    deployment_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    group_components: list[dict[str, str]] = []
    overlap: list[dict[str, Any]] = []
    allowed_template_question_overlaps = 0

    for sample in prospective:
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate prospective sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        image_hash = _image_hash(sample.image_path)
        question_key = _question_key(sample.question)
        reasons = []
        if image_hash in development_images:
            reasons.append("image_sha256")
        question_is_template = (
            development_question_counts[question_key] >= 5
            or prospective_question_counts[question_key] >= 5
        )
        if development_question_counts[question_key] and not question_is_template:
            reasons.append("normalized_exact_question")
        elif development_question_counts[question_key] and question_is_template:
            allowed_template_question_overlaps += 1
        if reasons:
            overlap.append({"sample_id": sample.sample_id, "reasons": reasons})
            continue
        safe_metadata = {}
        option_map = sample.metadata.get("option_map")
        if isinstance(option_map, dict):
            safe_metadata["option_map"] = option_map
        deployment_rows.append(
            {
                "sample_id": sample.sample_id,
                "split": "prospective",
                "dataset": sample.dataset,
                "image_path": os.path.relpath(sample.image_path.resolve(), output_dir.resolve()),
                "question": sample.question,
                "options": sample.options,
                "metadata": safe_metadata,
            }
        )
        hidden_rows.append(
            {
                "sample_id": sample.sample_id,
                "gold_answer": sample.answer,
                "counterfactual_image_path": (
                    os.path.relpath(sample.counterfactual_image_path.resolve(), output_dir.resolve())
                    if sample.counterfactual_image_path else None
                ),
                "original_metadata": sample.metadata,
            }
        )
        group_components.append(
            {
                "sample_id": sample.sample_id,
                "image_sha256": image_hash,
                "normalized_question_sha256": hashlib.sha256(question_key.encode()).hexdigest(),
                "question_is_template": str(question_is_template).lower(),
            }
        )
    if overlap:
        (output_dir / "rejected_development_overlap.json").write_bytes(canonical_json_bytes(overlap))
        raise ValueError(
            f"Prospective isolation failed for {len(overlap)} cases; see rejected_development_overlap.json"
        )
    if not deployment_rows:
        raise ValueError("No prospective cases remain")

    parent = {row["sample_id"]: row["sample_id"] for row in group_components}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    image_first: dict[str, str] = {}
    question_first: dict[str, str] = {}
    for row in group_components:
        sample_id = row["sample_id"]
        image_hash = row["image_sha256"]
        question_hash = row["normalized_question_sha256"]
        if image_hash in image_first:
            union(sample_id, image_first[image_hash])
        else:
            image_first[image_hash] = sample_id
        if row["question_is_template"] != "true":
            if question_hash in question_first:
                union(sample_id, question_first[question_hash])
            else:
                question_first[question_hash] = sample_id
    group_rows = [
        {
            **row,
            "source_group": hashlib.sha256(find(row["sample_id"]).encode()).hexdigest(),
        }
        for row in group_components
    ]

    deployment_path = output_dir / "deployment_samples.jsonl"
    labels_path = output_dir / "hidden_evaluation_labels.jsonl"
    groups_path = output_dir / "source_groups.jsonl"
    write_jsonl(deployment_path, deployment_rows)
    write_jsonl(labels_path, hidden_rows)
    write_jsonl(groups_path, group_rows)
    manifest = {
        "status": "frozen_before_inference",
        "sample_count": len(deployment_rows),
        "source_group_count": len({row["source_group"] for row in group_rows}),
        "development_sample_count": len(development),
        "isolation_rules": [
            "exact image SHA-256",
            "normalized exact non-template question (frequency below 5)"
        ],
        "allowed_template_question_overlaps": allowed_template_question_overlaps,
        "deployment_contains_gold_labels": False,
        "checksums": {
            "source_development": sha256_path(development_path),
            "source_prospective": sha256_path(prospective_path),
            "deployment_samples": sha256_path(deployment_path),
            "hidden_evaluation_labels": sha256_path(labels_path),
            "source_groups": sha256_path(groups_path),
        },
    }
    manifest_path = output_dir / "frozen_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (output_dir / "frozen_manifest.sha256").write_text(
        f"{sha256_path(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
