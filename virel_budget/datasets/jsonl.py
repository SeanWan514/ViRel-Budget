from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from virel_budget.schema import Sample


def load_jsonl_samples(path: str | Path, dataset_name: str, limit: int | None = None) -> list[Sample]:
    jsonl_path = Path(path)
    samples: list[Sample] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            sample = _sample_from_obj(obj, jsonl_path.parent, dataset_name, line_no)
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    return samples


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sample_from_obj(obj: dict, base_dir: Path, dataset_name: str, line_no: int) -> Sample:
    obj = _repair_sample_obj(obj)
    sample_id = str(obj.get("sample_id") or obj.get("id") or f"row-{line_no}")
    split = str(obj.get("split", "test"))
    image_path = _resolve(base_dir, obj["image_path"])
    cf_path = obj.get("counterfactual_image_path")
    return Sample(
        sample_id=sample_id,
        split=split,
        dataset=str(obj.get("dataset", dataset_name)),
        image_path=image_path,
        question=str(obj["question"]),
        answer=str(obj["answer"]),
        options=[str(x) for x in obj.get("options", [])],
        counterfactual_image_path=_resolve(base_dir, cf_path) if cf_path else None,
        metadata=dict(obj.get("metadata", {})),
    )


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _repair_sample_obj(obj: dict) -> dict:
    dataset = str(obj.get("dataset", "")).lower()
    if dataset != "mmstar":
        return obj
    options = [str(x) for x in obj.get("options", [])]
    option_map = _extract_lettered_options(str(obj.get("question", "")))
    if not option_map and options:
        option_map = {letter: option for letter, option in zip(["A", "B", "C", "D", "E"], options)}
    if not option_map:
        return obj
    repaired = dict(obj)
    metadata = dict(repaired.get("metadata", {}))
    answer_letter = str(repaired.get("answer", "")).strip().upper()
    metadata.setdefault("option_map", option_map)
    if answer_letter in option_map:
        metadata.setdefault("answer_letter", answer_letter)
    if options and all(opt in option_map for opt in options):
        metadata.update(
            {
                "original_answer": repaired.get("answer"),
                "original_options": options,
                "option_map": option_map,
                "option_repair": "mmstar_full_text_from_question",
            }
        )
        repaired["options"] = [option_map[letter] for letter in options]
    elif options:
        repaired["options"] = options
    else:
        repaired["options"] = [option_map[letter] for letter in sorted(option_map)]
    if answer_letter in option_map:
        repaired["answer"] = option_map[answer_letter]
    repaired["metadata"] = metadata
    return repaired


def _extract_lettered_options(question: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if "Choices:" in question:
        option_text = question.split("Choices:", 1)[1]
        pattern = re.compile(r"\(([A-E])\)\s*(.*?)(?=\n\s*\([A-E]\)|$)", re.DOTALL)
        for letter, text in pattern.findall(option_text):
            cleaned = " ".join(text.strip().strip(",").split())
            if cleaned:
                out[letter] = cleaned
        if out:
            return out
    if "Options:" in question:
        option_text = question.split("Options:", 1)[1]
        pattern = re.compile(r"([A-E]):\s*(.*?)(?=,\s*[A-E]:|\n\s*[A-E]:|$)", re.DOTALL)
        for letter, text in pattern.findall(option_text):
            cleaned = " ".join(text.strip().strip(",").split())
            if cleaned:
                out[letter] = cleaned
    return out
