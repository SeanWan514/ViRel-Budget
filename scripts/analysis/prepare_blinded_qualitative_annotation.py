from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_image(row: dict[str, Any], parent: Path, key: str) -> Path | None:
    value = row.get(key)
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (parent / path).resolve()


def choose_cases(decisions: list[dict[str, str]]) -> list[dict[str, str]]:
    rng = random.Random(4801)
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in decisions:
        by_sample[row["sample_id"]].append(row)
    categories: list[tuple[str, list[dict[str, str]], int]] = []
    categories.append(("unsafe_acceptance", [r for r in decisions if r["unsafe_acceptance"] == "True"], 10))
    categories.append((
        "safe_aggressive",
        [r for r in decisions if r["supported_correct"] == "True" and r["selected_budget"] in {"64", "128", "256"}],
        8,
    ))
    categories.append((
        "safe_432",
        [r for r in decisions if r["supported_correct"] == "True" and r["selected_budget"] == "432"],
        6,
    ))
    categories.append(("dense_fallback", [r for r in decisions if r["selected_budget"] == "dense"], 8))
    disagreement = []
    for rows in by_sample.values():
        values = {row["unsafe_acceptance"] for row in rows if row["selected_budget"] != "dense"}
        if len(values) > 1:
            disagreement.extend(rows)
    categories.append(("method_disagreement", disagreement, 8))

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for category, candidates, count in categories:
        rng.shuffle(candidates)
        for row in candidates:
            if row["sample_id"] in seen:
                continue
            item = dict(row)
            item["audit_category"] = category
            selected.append(item)
            seen.add(row["sample_id"])
            if sum(x["audit_category"] == category for x in selected) == count:
                break
    # Fill to 48 with diverse remaining cases; the private mapping retains the reason.
    remaining = list(decisions)
    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) == 48:
            break
        if row["sample_id"] in seen:
            continue
        item = dict(row)
        item["audit_category"] = "diversity_fill"
        selected.append(item)
        seen.add(row["sample_id"])
    if len(selected) != 48:
        raise ValueError(f"Could not select 48 unique qualitative cases: {len(selected)}")
    return selected


def intervention_for(row: dict[str, Any], index: int) -> str:
    if row["dataset"].startswith("visual_counterfact") and row.get("counterfactual_image_path"):
        return "counterfactual"
    return ("blur", "constant_gray", "irrelevant")[index % 3]


def materialize_variant(
    original: Path, counterfactual: Path | None, family: str, irrelevant: Path, output: Path
) -> None:
    image = Image.open(original).convert("RGB")
    if family == "counterfactual":
        if counterfactual is None:
            raise ValueError("Counterfactual case lacks counterfactual image")
        variant = Image.open(counterfactual).convert("RGB").resize(image.size)
    elif family == "blur":
        variant = image.filter(ImageFilter.GaussianBlur(radius=8.0))
    elif family == "constant_gray":
        variant = Image.new("RGB", image.size, (128, 128, 128))
    elif family == "irrelevant":
        variant = Image.open(irrelevant).convert("RGB").resize(image.size)
    else:
        raise ValueError(family)
    variant.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/qualitative_annotation_48")
    args = parser.parse_args()
    out = Path(args.output_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    decisions = read_csv(Path("results/phase_b_analysis/prospective_controller_decisions.csv"))
    selected = choose_cases(decisions)
    source_path = Path("data/prospective_900_revealed/samples.jsonl")
    source = {row["sample_id"]: row for row in read_jsonl(source_path)}
    irrelevant = sorted(Path("data/paper_1200/irrelevant").glob("*"))[0]
    public_rows = []
    private_rows = []
    for index, decision in enumerate(selected, 1):
        sample = source[decision["sample_id"]]
        blind_id = f"QA{index:03d}"
        original = resolve_image(sample, source_path.parent, "image_path")
        counterfactual = resolve_image(sample, source_path.parent, "counterfactual_image_path")
        family = intervention_for(sample, index)
        original_out = images / f"{blind_id}_original.jpg"
        variant_out = images / f"{blind_id}_intervened.png"
        shutil.copy2(original, original_out)
        materialize_variant(original, counterfactual, family, irrelevant, variant_out)
        public_rows.append({
            "blind_id": blind_id,
            "dataset": sample["dataset"],
            "question": sample["question"],
            "options": json.dumps(sample.get("options") or [], ensure_ascii=False),
            "intervention_family": family,
            "original_image": str(original_out.relative_to(out)),
            "intervened_image": str(variant_out.relative_to(out)),
            "eligible_intervention_yes_no_uncertain": "",
            "original_answer": "",
            "intervened_answer": "",
            "artifact_present_yes_no_uncertain": "",
            "artifact_reason": "",
            "confidence_1_to_5": "",
            "notes": "",
        })
        private_rows.append({
            "blind_id": blind_id,
            "sample_id": decision["sample_id"],
            "audit_category": decision["audit_category"],
            "model": decision["model"],
            "method": decision["method"],
            "selected_budget": decision["selected_budget"],
            "unsafe_acceptance": decision["unsafe_acceptance"],
            "supported_correct": decision["supported_correct"],
            "gold_answer": sample["answer"],
            "original_sha256": sha256(original_out),
            "intervened_sha256": sha256(variant_out),
        })
    fields = list(public_rows[0])
    for annotator, seed in (("A", 4802), ("B", 4803)):
        rows = list(public_rows)
        random.Random(seed).shuffle(rows)
        with (out / f"annotation_form_{annotator}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (out / "private_blind_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(private_rows[0]))
        writer.writeheader()
        writer.writerows(private_rows)
    instructions = """# Blinded qualitative annotation

Review each original/intervened pair without opening `private_blind_mapping.csv`.

1. Judge whether the intervention is eligible for the question: it should meaningfully remove,
   alter, or replace visual evidence relevant to answering the question.
2. Answer the question independently for the original and intervened image.
3. Mark visible artifacts that could cause a response change for reasons unrelated to the intended
   intervention.
4. Use `uncertain` rather than forcing an answer.
5. Annotators A and B work independently. Resolve disagreements only after both forms are frozen.

The method, model, token budget, controller outcome, and success/failure category are blinded.
"""
    (out / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    manifest = {
        "sample_count": 48,
        "selection_seed": 4801,
        "annotator_order_seeds": {"A": 4802, "B": 4803},
        "forms": {
            "A": sha256(out / "annotation_form_A.csv"),
            "B": sha256(out / "annotation_form_B.csv"),
            "private_mapping": sha256(out / "private_blind_mapping.csv"),
        },
        "blinded_fields": ["sample_id", "model", "method", "selected_budget", "controller_outcome", "audit_category"],
        "required_analysis": ["raw agreement", "Cohen kappa", "eligibility rate", "artifact rate", "resolved adjudication"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
