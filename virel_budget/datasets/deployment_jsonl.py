from __future__ import annotations

import json
from pathlib import Path

from virel_budget.frozen_controller import FORBIDDEN_INPUT_KEYS
from virel_budget.schema import Sample


def load_deployment_samples(path: str | Path) -> list[Sample]:
    source = Path(path)
    samples: list[Sample] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = FORBIDDEN_INPUT_KEYS & set(row)
            if forbidden:
                raise ValueError(
                    f"Deployment row {line_number} contains forbidden fields: {sorted(forbidden)}"
                )
            required = {"sample_id", "dataset", "image_path", "question", "options"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"Deployment row {line_number} is missing {sorted(missing)}")
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = source.parent / image_path
            samples.append(
                Sample(
                    sample_id=str(row["sample_id"]),
                    split=str(row.get("split", "prospective")),
                    dataset=str(row["dataset"]),
                    image_path=image_path,
                    question=str(row["question"]),
                    answer="",
                    options=[str(value) for value in row.get("options", [])],
                    metadata=dict(row.get("metadata") or {}),
                )
            )
    return samples
