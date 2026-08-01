from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.datasets.jsonl import load_jsonl_samples, write_jsonl
from virel_budget.frozen_controller import canonical_json_bytes, sha256_path


POPE = ("pope_adversarial", "pope_popular", "pope_random")
VCF = ("visual_counterfact_color", "visual_counterfact_size")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the group-isolated 300+300+300 expansion source.")
    parser.add_argument("--sources", default="data/expansion_sources")
    parser.add_argument("--development", default="data/paper_1200/samples.jsonl")
    parser.add_argument("--output", default="data/prospective_900_source")
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def _stable(seed: int, *parts: object) -> str:
    return hashlib.sha256("|".join([str(seed), *(str(value) for value in parts)]).encode()).hexdigest()


def _image_hash(path: Path) -> str:
    return sha256_path(path)


def _load_sources(root: Path) -> dict[str, list[Any]]:
    datasets = [*POPE, "mmstar", *VCF]
    return {
        dataset: load_jsonl_samples(root / dataset / "samples.jsonl", dataset_name=dataset)
        for dataset in datasets
    }


def _eligible(
    rows: list[Any],
    development_ids: set[str],
    development_images: set[str],
    development_question_counts: Counter[str],
) -> list[Any]:
    output = []
    seen_images = set(development_images)
    for sample in rows:
        if sample.sample_id in development_ids:
            continue
        question_key = " ".join(sample.question.lower().split())
        # Match the freezer: frequent templates are legitimate benchmark
        # structure, while rare/exact question overlap is isolated.
        if 0 < development_question_counts[question_key] < 5:
            continue
        digest = _image_hash(sample.image_path)
        if digest in seen_images:
            continue
        if not sample.image_path.exists():
            continue
        if sample.dataset.startswith("visual_counterfact") and (
            sample.counterfactual_image_path is None or not sample.counterfactual_image_path.exists()
        ):
            continue
        output.append(sample)
        # One selected question per original image gives source-level independence.
        seen_images.add(digest)
    return output


def _take(rows: list[Any], count: int, seed: int, *parts: object) -> list[Any]:
    ordered = sorted(rows, key=lambda sample: _stable(seed, *parts, sample.sample_id))
    if len(ordered) < count:
        raise ValueError(f"Need {count} eligible rows for {parts}, found {len(ordered)}")
    return ordered[:count]


def _take_unique_images(
    rows: list[Any],
    count: int,
    used_hashes: set[str],
    seed: int,
    *parts: object,
) -> list[Any]:
    selected = []
    for sample in sorted(rows, key=lambda value: _stable(seed, *parts, value.sample_id)):
        digest = _image_hash(sample.image_path)
        if digest in used_hashes:
            continue
        selected.append(sample)
        used_hashes.add(digest)
        if len(selected) == count:
            return selected
    raise ValueError(f"Need {count} cross-dataset unique rows for {parts}, found {len(selected)}")


def _select_mmstar(rows: list[Any], seed: int) -> list[Any]:
    strata: dict[str, list[Any]] = defaultdict(list)
    for sample in rows:
        metadata = sample.metadata
        key = str(
            metadata.get("source_l2_category")
            or metadata.get("source_category")
            or "unknown"
        )
        strata[key].append(sample)
    queues = {
        key: sorted(values, key=lambda sample: _stable(seed, "mmstar", key, sample.sample_id))
        for key, values in strata.items()
    }
    selected: list[Any] = []
    # Balanced round-robin avoids failing when one category has one fewer
    # group-isolated case than the ideal equal quota; any shortfall is filled
    # evenly by categories that still have eligible cases.
    while len(selected) < 300:
        progressed = False
        for key in sorted(queues):
            if queues[key] and len(selected) < 300:
                selected.append(queues[key].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(f"Need 300 eligible MMStar rows, found {len(selected)}")
    return selected


def _copy_sample(sample: Any, output: Path) -> dict[str, Any]:
    row = {
        "sample_id": sample.sample_id,
        "split": "prospective",
        "dataset": sample.dataset,
        "question": sample.question,
        "answer": sample.answer,
        "options": sample.options,
        "metadata": sample.metadata,
    }
    for field, source in (
        ("image_path", sample.image_path),
        ("counterfactual_image_path", sample.counterfactual_image_path),
    ):
        if source is None:
            continue
        suffix = source.suffix or ".jpg"
        kind = "counterfactual" if field.startswith("counterfactual") else "original"
        target = output / "images" / sample.dataset / f"{sample.sample_id}_{kind}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        row[field] = os.path.relpath(target, output)
    return row


def main() -> None:
    args = _arguments()
    sources_root = Path(args.sources)
    development_path = Path(args.development)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty expansion directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    development = load_jsonl_samples(development_path, dataset_name="development")
    development_ids = {sample.sample_id for sample in development}
    development_images = {_image_hash(sample.image_path) for sample in development}
    development_question_counts = Counter(
        " ".join(sample.question.lower().split()) for sample in development
    )
    sources = _load_sources(sources_root)
    eligible = {
        dataset: _eligible(
            rows, development_ids, development_images, development_question_counts
        )
        for dataset, rows in sources.items()
    }

    selected = []
    for dataset in POPE:
        # POPE subtypes intentionally reuse a common COCO image pool. Preserve
        # 100 questions per subtype and record shared images as source groups;
        # only development/prospective image overlap is prohibited.
        selected.extend(_take(eligible[dataset], 100, args.seed, "pope", dataset))
    mmstar_selected = _select_mmstar(eligible["mmstar"], args.seed)
    selected.extend(mmstar_selected)
    for dataset in VCF:
        selected.extend(_take(eligible[dataset], 150, args.seed, "vcf", dataset))
    if len(selected) != 900 or len({sample.sample_id for sample in selected}) != 900:
        raise AssertionError("Expansion must contain 900 unique sample IDs")

    rows = [
        _copy_sample(sample, output)
        for sample in sorted(selected, key=lambda value: (value.dataset, value.sample_id))
    ]
    samples_path = output / "samples.jsonl"
    write_jsonl(samples_path, rows)
    manifest = {
        "status": "candidate source for prospective freezing; contains hidden labels",
        "seed": args.seed,
        "sample_count": len(rows),
        "family_counts": {
            "pope": sum(row["dataset"].startswith("pope_") for row in rows),
            "mmstar": sum(row["dataset"] == "mmstar" for row in rows),
            "visual_counterfact": sum(row["dataset"].startswith("visual_counterfact") for row in rows),
        },
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "unique_original_image_hashes": len(
            {_image_hash(output / row["image_path"]) for row in rows}
        ),
        "source_group_note": "Repeated source images within the prospective pool are retained in the same source group; no source image overlaps development.",
        "counterfactual_coverage": {
            "required": sum(row["dataset"].startswith("visual_counterfact") for row in rows),
            "present": sum(bool(row.get("counterfactual_image_path")) for row in rows),
        },
        "development_source_checksum": sha256_path(development_path),
        "samples_sha256": sha256_path(samples_path),
        "source_manifests": {
            dataset: sha256_path(sources_root / dataset / "manifest.json")
            for dataset in sources
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (output / "manifest.sha256").write_text(
        f"{sha256_path(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
