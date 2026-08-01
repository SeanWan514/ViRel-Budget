from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from virel_budget.datasets.jsonl import _repair_sample_obj


POPE_DATASETS = ["pope_adversarial", "pope_popular", "pope_random"]
VCF_DATASETS = ["visual_counterfact_color", "visual_counterfact_size"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a paper-scale balanced 1200-sample ViRel-Budget dataset.")
    parser.add_argument("--data-root", default="data", help="Root containing normalized dataset folders.")
    parser.add_argument("--out", default="data/paper_1200", help="Output directory for balanced dataset.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--pope", type=int, default=400)
    parser.add_argument("--mmstar", type=int, default=400)
    parser.add_argument("--vcf", type=int, default=400)
    parser.add_argument("--irrelevant-pool-size", type=int, default=48)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied_images_dir = out_dir / "images"
    if copied_images_dir.exists():
        shutil.rmtree(copied_images_dir)

    selected: list[dict[str, Any]] = []
    selected.extend(_select_family(data_root, POPE_DATASETS, args.pope, args.seed, out_dir, "pope_subtype"))
    selected.extend(_select_mmstar(data_root / "mmstar" / "samples.jsonl", args.mmstar, args.seed, out_dir))
    selected.extend(_select_family(data_root, VCF_DATASETS, args.vcf, args.seed, out_dir, "vcf_split"))
    selected = sorted(selected, key=lambda row: (str(row["dataset"]), str(row["sample_id"])))

    _write_jsonl(out_dir / "samples.jsonl", selected)
    _build_irrelevant_pool(out_dir, selected, args.irrelevant_pool_size, args.seed)
    manifest = _manifest(selected, args, out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _select_family(
    data_root: Path,
    dataset_names: list[str],
    target: int,
    seed: int,
    out_dir: Path,
    stratum_name: str,
) -> list[dict[str, Any]]:
    groups = {name: _load_jsonl(data_root / name / "samples.jsonl") for name in dataset_names}
    allocation = _balanced_allocation({name: len(rows) for name, rows in groups.items()}, target)
    selected: list[dict[str, Any]] = []
    for name, rows in groups.items():
        chosen = _take(rows, allocation[name], seed, stratum_name, name)
        selected.extend(_copy_paths(chosen, data_root / name, out_dir))
    return selected


def _select_mmstar(path: Path, target: int, seed: int, out_dir: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        key = str(
            metadata.get("source_l2_category")
            or metadata.get("source_category")
            or metadata.get("l2_category")
            or metadata.get("category")
            or "unknown"
        )
        strata[key].append(row)
    allocation = _balanced_allocation({name: len(items) for name, items in strata.items()}, target)
    selected: list[dict[str, Any]] = []
    for name in sorted(strata):
        selected.extend(_take(strata[name], allocation[name], seed, "mmstar_l2", name))
    return _copy_paths(selected, path.parent, out_dir)


def _balanced_allocation(capacity: dict[str, int], target: int) -> dict[str, int]:
    if not capacity:
        raise ValueError("Cannot allocate from empty capacity map")
    if sum(capacity.values()) < target:
        raise ValueError(f"Requested {target} rows, but only {sum(capacity.values())} are available: {capacity}")
    keys = sorted(capacity)
    allocation = {key: 0 for key in keys}
    remaining = int(target)
    while remaining > 0:
        open_keys = [key for key in keys if allocation[key] < capacity[key]]
        if not open_keys:
            raise ValueError(f"Allocation exhausted before reaching target {target}: {allocation}")
        step = max(remaining // len(open_keys), 1)
        progressed = False
        for key in open_keys:
            if remaining <= 0:
                break
            add = min(step, capacity[key] - allocation[key], remaining)
            if add > 0:
                allocation[key] += add
                remaining -= add
                progressed = True
        if not progressed:
            raise RuntimeError(f"Allocation did not progress: {allocation}")
    return allocation


def _take(rows: list[dict[str, Any]], n: int, seed: int, *parts: object) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: _stable_hash(seed, *parts, row.get("sample_id", "")))
    return ordered[:n]


def _copy_paths(rows: list[dict[str, Any]], source_base: Path, out_dir: Path) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        updated = _repair_sample_obj(dict(row))
        for key in ["image_path", "counterfactual_image_path"]:
            value = updated.get(key)
            if not value:
                continue
            path = Path(str(value))
            absolute = path if path.is_absolute() else source_base / path
            if not absolute.exists():
                updated[key] = os.path.relpath(absolute, out_dir)
                continue
            kind = "counterfactual" if key == "counterfactual_image_path" else "original"
            suffix = absolute.suffix or ".jpg"
            target = out_dir / "images" / str(row["dataset"]) / f"{row['sample_id']}_{kind}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(absolute, target)
            updated[key] = os.path.relpath(target, out_dir)
        rewritten.append(updated)
    return rewritten


def _build_irrelevant_pool(out_dir: Path, rows: list[dict[str, Any]], pool_size: int, seed: int) -> None:
    pool_dir = out_dir / "irrelevant"
    pool_dir.mkdir(parents=True, exist_ok=True)
    for old in pool_dir.glob("*"):
        if old.is_file():
            old.unlink()
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    allocation = _balanced_allocation({key: len(items) for key, items in by_dataset.items()}, min(pool_size, len(rows)))
    pool_rows: list[dict[str, Any]] = []
    for dataset, items in by_dataset.items():
        pool_rows.extend(_take(items, allocation[dataset], seed, "irrelevant_pool", dataset))
    for idx, row in enumerate(sorted(pool_rows, key=lambda item: (str(item["dataset"]), str(item["sample_id"])))):
        src = out_dir / row["image_path"]
        suffix = src.suffix or ".jpg"
        if src.exists():
            shutil.copy2(src, pool_dir / f"irrelevant_{idx:03d}{suffix}")


def _manifest(rows: list[dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    missing_paths = []
    counterfactual_missing = []
    for row in rows:
        image_path = out_dir / row["image_path"]
        if not image_path.exists():
            missing_paths.append({"sample_id": row["sample_id"], "path": row["image_path"]})
        if str(row["dataset"]).startswith("visual_counterfact"):
            cf = row.get("counterfactual_image_path")
            if not cf or not (out_dir / str(cf)).exists():
                counterfactual_missing.append(row["sample_id"])
    metadata = [row.get("metadata") or {} for row in rows]
    return {
        "dataset_name": "paper_1200_balanced",
        "seed": args.seed,
        "n_exported": len(rows),
        "targets": {"pope": args.pope, "mmstar": args.mmstar, "visual_counterfact": args.vcf},
        "by_family": {
            "pope": sum(1 for row in rows if str(row["dataset"]).startswith("pope_")),
            "mmstar": sum(1 for row in rows if row["dataset"] == "mmstar"),
            "visual_counterfact": sum(1 for row in rows if str(row["dataset"]).startswith("visual_counterfact")),
        },
        "by_dataset": _count(row["dataset"] for row in rows),
        "by_split": _count(row["split"] for row in rows),
        "pope_internal_balance": _count(row["dataset"] for row in rows if str(row["dataset"]).startswith("pope_")),
        "visual_counterfact_internal_balance": _count(
            row["dataset"] for row in rows if str(row["dataset"]).startswith("visual_counterfact")
        ),
        "mmstar_category_balance": _count(
            item.get("source_category") or item.get("category") or "unknown"
            for row, item in zip(rows, metadata)
            if row["dataset"] == "mmstar"
        ),
        "mmstar_l2_category_balance": _count(
            item.get("source_l2_category") or item.get("l2_category") or "unknown"
            for row, item in zip(rows, metadata)
            if row["dataset"] == "mmstar"
        ),
        "visual_counterfact_counterfactual_coverage": {
            "n_visual_counterfact": sum(1 for row in rows if str(row["dataset"]).startswith("visual_counterfact")),
            "missing_counterfactual_paths": len(counterfactual_missing),
        },
        "path_audit": {
            "missing_image_paths": len(missing_paths),
            "missing_counterfactual_sample_ids": counterfactual_missing[:20],
            "missing_image_examples": missing_paths[:20],
        },
        "irrelevant_pool_size": len(list((out_dir / "irrelevant").glob("*"))),
        "schema_note": "Raw-balanced before dense-reliant filtering; report post-filter counts and macro-averages after model runs.",
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _count(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _stable_hash(seed: int, *parts: object) -> str:
    return hashlib.sha256("|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
