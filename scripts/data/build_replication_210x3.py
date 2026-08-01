from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.frozen_controller import FORBIDDEN_INPUT_KEYS, canonical_json_bytes, sha256_path


SEEDS = (2101, 2102, 2103)
QUOTAS = {
    ("development", "mmstar"): 40,
    ("development", "pope_adversarial"): 14,
    ("development", "pope_popular"): 13,
    ("development", "pope_random"): 13,
    ("development", "visual_counterfact_color"): 20,
    ("development", "visual_counterfact_size"): 20,
    ("prospective", "mmstar"): 30,
    ("prospective", "pope_adversarial"): 10,
    ("prospective", "pope_popular"): 10,
    ("prospective", "pope_random"): 10,
    ("prospective", "visual_counterfact_color"): 15,
    ("prospective", "visual_counterfact_size"): 15,
}
MODELS = ("7b", "13b")
METHODS = ("fastv", "scope", "random")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def image_group(row: dict[str, Any], source_parent: Path) -> str:
    image = (source_parent / str(row["image_path"])).resolve()
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    return f"{row['dataset']}::{digest}"


def rewrite_paths(row: dict[str, Any], source_parent: Path, target_parent: Path) -> dict[str, Any]:
    out = copy.deepcopy(row)
    for key in ("image_path", "counterfactual_image_path"):
        value = out.get(key)
        if value:
            resolved = (source_parent / str(value)).resolve()
            if not resolved.exists():
                raise FileNotFoundError(resolved)
            out[key] = os.path.relpath(resolved, target_parent)
    return out


def select_draws(pool: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    used_groups: set[str] = set()
    draws: list[tuple[int, list[dict[str, Any]]]] = []
    for seed in SEEDS:
        rng = random.Random(seed)
        selected: list[dict[str, Any]] = []
        for key, quota in QUOTAS.items():
            candidates = [
                row for row in pool
                if (row["_provenance"], row["dataset"]) == key and row["_source_group"] not in used_groups
            ]
            rng.shuffle(candidates)
            chosen: list[dict[str, Any]] = []
            local_groups: set[str] = set()
            for row in candidates:
                if row["_source_group"] in local_groups:
                    continue
                chosen.append(row)
                local_groups.add(row["_source_group"])
                if len(chosen) == quota:
                    break
            if len(chosen) != quota:
                raise ValueError(f"Insufficient group-isolated rows for seed={seed}, stratum={key}: {len(chosen)}/{quota}")
            selected.extend(chosen)
        rng.shuffle(selected)
        groups = {row["_source_group"] for row in selected}
        if len(groups) != len(selected):
            raise AssertionError("A draw contains repeated source-image groups")
        used_groups.update(groups)
        draws.append((seed, selected))
    return draws


def config_for_draw(base: dict[str, Any], seed: int, model: str, method: str, samples: Path) -> dict[str, Any]:
    out = copy.deepcopy(base)
    run_name = f"replication210x3_draw{seed}_llava15_{model}_{method}"
    out["run_name"] = run_name
    out["seed"] = 13
    out["dataset"]["name"] = f"replication210x3_draw{seed}"
    out["dataset"]["path"] = str(samples)
    out["dataset"]["validation_split"] = "replication"
    out["dataset"]["test_split"] = "replication"
    out["dataset"]["limit"] = None
    out["outputs"] = {"dir": f"results/{run_name}"}
    out.pop("execution_guard", None)
    out["replication_protocol"] = {
        "artifact": "results/replication210x3_protocol.json",
        "draw_seed": seed,
        "sample_count": 210,
        "controller_refit": False,
        "labels_used_during_execution": False,
    }
    return out


def make_dense_controller(source: Path, output_dir: Path, model: str) -> dict[str, str]:
    artifact = json.loads(source.read_text(encoding="utf-8"))
    artifact["global_threshold"] = 2.0
    artifact["calibration"] = {
        "purpose": "Matched fixed-dense replication control",
        "selected_action_distribution": {"dense": 210},
    }
    artifact["development_only"] = True
    artifact["replication_dense_control"] = True
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fastv_controller.json"
    path.write_bytes(canonical_json_bytes(artifact))
    checksum = sha256_path(path)
    checksum_path = output_dir / "fastv_controller.sha256"
    checksum_path.write_text(f"{checksum}  {path.name}\n", encoding="utf-8")
    return {"controller": str(path), "checksum": str(checksum_path), "sha256": checksum, "model": model}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data/replication_210x3")
    parser.add_argument("--config-root", default="configs/replication_210x3")
    args = parser.parse_args()
    root = Path(args.out_root)
    config_root = Path(args.config_root)
    root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    sources = [
        ("development", Path("data/paper_1200/samples.jsonl")),
        ("prospective", Path("data/prospective_900_revealed/samples.jsonl")),
    ]
    pool: list[dict[str, Any]] = []
    for provenance, path in sources:
        for row in read_jsonl(path):
            item = copy.deepcopy(row)
            item["_provenance"] = provenance
            item["_source_parent"] = str(path.parent)
            item["_source_group"] = image_group(item, path.parent)
            pool.append(item)
    if len(pool) != 2100 or len({row["sample_id"] for row in pool}) != 2100:
        raise ValueError("Combined candidate pool must contain exactly 2,100 unique sample IDs")

    draw_entries = []
    for seed, rows in select_draws(pool):
        draw_dir = root / f"draw_{seed}"
        samples_path = draw_dir / "deployment_samples.jsonl"
        evaluation_path = draw_dir / "evaluation_samples.jsonl"
        clean = []
        groups = []
        for row in rows:
            source_parent = Path(row.pop("_source_parent"))
            provenance = row.pop("_provenance")
            group = row.pop("_source_group")
            rewritten = rewrite_paths(row, source_parent, draw_dir)
            rewritten["split"] = "replication"
            rewritten.setdefault("metadata", {})["replication_provenance"] = provenance
            rewritten["metadata"]["replication_source_group"] = group
            clean.append(rewritten)
            groups.append({"sample_id": row["sample_id"], "source_group": group, "provenance": provenance})
        deployment = [
            {key: value for key, value in row.items() if key not in FORBIDDEN_INPUT_KEYS}
            for row in clean
        ]
        write_jsonl(samples_path, deployment)
        write_jsonl(evaluation_path, clean)
        write_jsonl(draw_dir / "source_groups.jsonl", groups)
        counts = Counter((row["metadata"]["replication_provenance"], row["dataset"]) for row in clean)
        manifest = {
            "seed": seed,
            "sample_count": len(clean),
            "deployment_samples_sha256": sha256_path(samples_path),
            "evaluation_samples_sha256": sha256_path(evaluation_path),
            "source_groups_sha256": sha256_path(draw_dir / "source_groups.jsonl"),
            "unique_source_groups": len(set(item["source_group"] for item in groups)),
            "counts": {f"{a}/{b}": n for (a, b), n in sorted(counts.items())},
            "sample_ids": [row["sample_id"] for row in clean],
        }
        (draw_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        draw_entries.append({**manifest, "manifest_sha256": sha256_path(draw_dir / "manifest.json")})

        for model in MODELS:
            for method in METHODS:
                base_path = Path(f"configs/prospective900_llava15_{model}_{method}.json")
                base = json.loads(base_path.read_text(encoding="utf-8"))
                config = config_for_draw(base, seed, model, method, samples_path)
                out = config_root / f"draw{seed}_{model}_{method}.json"
                out.write_bytes(canonical_json_bytes(config))

    dense = {}
    for model in MODELS:
        source = Path(f"results/frozen_controllers_phase_a/{model}_fastv/fastv_controller.json")
        dense[model] = make_dense_controller(source, Path(f"results/replication210x3_frozen_dense/{model}"), model)

    protocol = {
        "artifact_type": "virel_replication_210x3_frozen_protocol",
        "purpose": "Three group-disjoint stratified 10% replication draws over the complete 2,100-query pool.",
        "candidate_pool": {
            "total": 2100,
            "development": 1200,
            "prospective": 900,
            "source_files": [str(path) for _, path in sources],
        },
        "seeds": list(SEEDS),
        "draws": draw_entries,
        "quotas_per_draw": {f"{a}/{b}": n for (a, b), n in QUOTAS.items()},
        "models": list(MODELS),
        "methods": list(METHODS),
        "matched_dense_controls": dense,
        "execution": {
            "controller_refit": False,
            "one_backend_invocation_per_query": True,
            "strict_labels_not_loaded_by_execution": True,
            "nvidia_smi_interval_ms": 200,
            "codecarbon_interval_s": 1,
            "warmup_queries_per_block": 3,
            "complete_primary_inferences": 5040,
        },
        "gate_r": {
            "required_for_completion": [
                "24/24 execution blocks complete",
                "5,040/5,040 actions durable",
                "one backend invocation per action",
                "NVIDIA and CodeCarbon telemetry present",
                "no sample or source group shared across draws",
            ],
            "strong_green_replication": "all three paired draw energy reductions positive and pooled paired 95% CI lower bound > 0",
            "mixed_green_replication": "pooled point estimate positive but at least one draw non-positive or pooled 95% CI includes 0",
            "failed_green_replication": "pooled point estimate non-positive",
        },
    }
    protocol_path = Path("results/replication210x3_protocol.json")
    protocol_path.write_bytes(canonical_json_bytes(protocol))
    Path("results/replication210x3_protocol.sha256").write_text(
        f"{sha256_path(protocol_path)}  {protocol_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
