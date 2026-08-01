from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from virel_budget.datasets.jsonl import write_jsonl


DATASET_REGISTRY = {
    "visual_counterfact": {
        "hf_id": "mgolov/Visual-Counterfact",
        "default_config": "default",
        "default_split": "color",
        "role": "core causal-reliance benchmark",
    },
    "visual_counterfact_size": {
        "hf_id": "mgolov/Visual-Counterfact",
        "default_config": "default",
        "default_split": "size",
        "role": "core causal-reliance benchmark with object-size counterfactuals",
    },
    "mmstar": {
        "hf_id": "Lin-Chen/MMStar",
        "default_config": "val",
        "default_split": "val",
        "role": "core vision-indispensable benchmark",
    },
    "pope": {
        "hf_id": "lmms-lab/POPE",
        "default_config": "Full",
        "default_split": "adversarial",
        "role": "core object hallucination benchmark",
    },
}


def prepare_hf_dataset(
    dataset_key: str,
    out_dir: str | Path,
    split: str | None = None,
    limit: int | None = None,
    validation_fraction: float = 0.2,
    seed: int = 13,
) -> Path:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Dataset preparation requires: python3 -m pip install -e '.[research]'") from exc
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset key {dataset_key}. Known: {sorted(DATASET_REGISTRY)}")
    spec = DATASET_REGISTRY[dataset_key]
    selected_config = spec.get("default_config")
    selected_split = split or spec["default_split"]
    root = Path(out_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    if selected_config:
        ds = load_dataset(spec["hf_id"], selected_config, split=selected_split)
    else:
        ds = load_dataset(spec["hf_id"], split=selected_split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for idx, item in enumerate(ds):
        row = _normalize_item(dataset_key, item, idx, images_dir, validation_fraction, seed)
        if row is not None:
            rows.append(row)
    out_path = root / "samples.jsonl"
    write_jsonl(out_path, rows)
    manifest = {
        "dataset_key": dataset_key,
        "hf_id": spec["hf_id"],
        "config": selected_config,
        "split": selected_split,
        "role": spec["role"],
        "n_exported": len(rows),
        "validation_fraction": validation_fraction,
        "schema_note": "Best-effort normalization; inspect samples before paper runs.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _normalize_item(
    dataset_key: str,
    item: dict[str, Any],
    idx: int,
    images_dir: Path,
    validation_fraction: float,
    seed: int,
) -> dict[str, Any] | None:
    image = _find_image(item)
    if image is None:
        return None
    sample_id = str(item.get("id") or item.get("sample_id") or item.get("question_id") or f"{dataset_key}-{idx:06d}")
    image_path = images_dir / f"{sample_id}.png"
    image.convert("RGB").save(image_path)
    question = _first_text(item, ["question", "query", "prompt", "text", "sentence"])
    answer = _first_text(item, ["answer", "label", "gt_answer", "ground_truth", "target", "correct_answer"])
    if dataset_key.startswith("visual_counterfact"):
        object_name = str(item.get("object") or "object")
        if dataset_key == "visual_counterfact":
            question = f"What color is the {object_name} in the image?"
        else:
            question = "What is the main object in the image?"
        answer = _clean_visual_counterfact_answer(str(item.get("correct_answer") or answer or ""))
    if question is None or answer is None:
        return None
    options = _find_options(item)
    if dataset_key.startswith("visual_counterfact"):
        incorrect = _clean_visual_counterfact_answer(str(item.get("incorrect_answer") or ""))
        options = [x for x in [answer, incorrect] if x]
    elif dataset_key == "mmstar" and not options:
        options = _letter_options_from_question(question)
    split = _deterministic_split(sample_id, validation_fraction, seed)
    counterfactual_path = None
    cf_image = _find_image(item, keys=["counterfactual_image", "counterfact_image", "cf_image", "image_counterfactual", "edited_image"])
    if cf_image is not None:
        counterfactual_path = images_dir / f"{sample_id}_counterfactual.png"
        cf_image.convert("RGB").save(counterfactual_path)
    row = {
        "sample_id": sample_id,
        "split": split,
        "dataset": dataset_key,
        "image_path": str(image_path.relative_to(images_dir.parent)),
        "question": question,
        "answer": answer,
        "options": options,
        "metadata": {k: _json_safe(v) for k, v in item.items() if k not in {"image"}},
    }
    if counterfactual_path is not None:
        row["counterfactual_image_path"] = str(counterfactual_path.relative_to(images_dir.parent))
    return row


def _find_image(item: dict[str, Any], keys: list[str] | None = None) -> Image.Image | None:
    search_keys = keys or ["image", "img", "image_1", "input_image"]
    for key in search_keys:
        value = item.get(key)
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict) and "path" in value:
            try:
                return Image.open(value["path"])
            except Exception:
                continue
        if isinstance(value, str) and Path(value).exists():
            try:
                return Image.open(value)
            except Exception:
                continue
    if keys is not None:
        return None
    for value in item.values():
        if isinstance(value, Image.Image):
            return value
    return None


def _first_text(item: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            return str(value)
    return None


def _find_options(item: dict[str, Any]) -> list[str]:
    for key in ["options", "choices", "candidates", "answer_options"]:
        value = item.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, dict):
            return [str(v) for _, v in sorted(value.items())]
    lettered = []
    for key in ["A", "B", "C", "D", "E"]:
        if key in item:
            lettered.append(str(item[key]))
    if lettered:
        return lettered
    answer = _first_text(item, ["answer", "label", "gt_answer", "ground_truth", "target", "correct_answer"])
    if str(answer).lower() in {"yes", "no", "true", "false"}:
        return ["yes", "no"]
    return []


def _clean_visual_counterfact_answer(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        import ast

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except Exception:
            pass
    return value.strip("'\" ")


def _letter_options_from_question(question: str) -> list[str]:
    import re

    letters = re.findall(r"(?:^|[,;\n ])([A-E]):", question)
    deduped = []
    for letter in letters:
        if letter not in deduped:
            deduped.append(letter)
    return deduped


def _deterministic_split(sample_id: str, validation_fraction: float, seed: int) -> str:
    import hashlib

    digest = hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if value < validation_fraction else "test"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return "<image>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)
