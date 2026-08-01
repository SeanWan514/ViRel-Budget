from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from virel_budget.config import load_config
from virel_budget.datasets.jsonl import write_jsonl
from virel_budget.datasets.hf_prepare import DATASET_REGISTRY, prepare_hf_dataset
from virel_budget.pipeline import analyze_records, run_experiment
from virel_budget.schema import EvalRecord


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="virel-budget")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a ViRel-Budget experiment config.")
    run_p.add_argument("--config", required=True, help="Path to JSON config.")

    analyze_p = sub.add_parser("analyze-existing", help="Recompute policy/reporting artifacts from an existing records.jsonl.")
    analyze_p.add_argument("--config", required=True, help="Config used for thresholds, policy, and dataset split names.")
    analyze_p.add_argument("--records", required=True, help="Existing records.jsonl produced by a prior run.")
    analyze_p.add_argument("--output-dir", required=True, help="Directory for regenerated analysis artifacts.")
    analyze_p.add_argument("--run-name", default=None, help="Optional run_name override for regenerated reports.")

    smoke_p = sub.add_parser("make-smoke-data", help="Create tiny local smoke images and JSONL.")
    smoke_p.add_argument("--out", default="data/smoke", help="Output directory.")

    hf_p = sub.add_parser("prepare-hf-dataset", help="Normalize a Hugging Face benchmark into ViRel-Budget JSONL.")
    hf_p.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY), help="Dataset key.")
    hf_p.add_argument("--out", required=True, help="Output directory.")
    hf_p.add_argument("--split", default=None, help="Dataset split override.")
    hf_p.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    hf_p.add_argument("--validation-fraction", type=float, default=0.2)
    hf_p.add_argument("--seed", type=int, default=13)

    args = parser.parse_args(argv)
    if args.command == "run":
        config = load_config(args.config)
        result = run_experiment(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "analyze-existing":
        config = load_config(args.config)
        if args.run_name:
            config["run_name"] = args.run_name
        config["outputs"] = dict(config.get("outputs", {}))
        config["outputs"]["dir"] = args.output_dir
        records = _load_records(Path(args.records))
        result = analyze_records(config, records, Path(args.output_dir))
        print(json.dumps({"run_name": config["run_name"], "output_dir": args.output_dir, "n_records": len(records), **result}, indent=2, sort_keys=True))
        return 0
    if args.command == "make-smoke-data":
        make_smoke_data(Path(args.out))
        print(f"Wrote smoke dataset to {args.out}")
        return 0
    if args.command == "prepare-hf-dataset":
        out = prepare_hf_dataset(
            args.dataset,
            args.out,
            split=args.split,
            limit=args.limit,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        print(f"Wrote normalized dataset to {out}")
        return 0
    raise ValueError(args.command)


def make_smoke_data(out_dir: Path) -> None:
    images_dir = out_dir / "images"
    irrelevant_dir = out_dir / "irrelevant"
    images_dir.mkdir(parents=True, exist_ok=True)
    irrelevant_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("vcf_val_red_square", "validation", "red", "What color is the central square?", "red", ["red", "blue", "green"]),
        ("vcf_val_blue_circle", "validation", "blue", "What color is the central circle?", "blue", ["red", "blue", "green"]),
        ("vcf_test_green_square", "test", "green", "What color is the central square?", "green", ["red", "blue", "green"]),
        ("vcf_test_red_circle", "test", "red", "What color is the central circle?", "red", ["red", "blue", "green"]),
    ]
    rows = []
    for sample_id, split, color, question, answer, options in specs:
        path = images_dir / f"{sample_id}.png"
        _draw_shape(path, color=color, circle="circle" in sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "dataset": "smoke_vcf_style",
                "image_path": str(path.relative_to(out_dir)),
                "question": question,
                "answer": answer,
                "options": options,
                "metadata": {"purpose": "offline smoke only; not paper evidence"},
            }
        )
    _draw_shape(irrelevant_dir / "irrelevant_yellow_triangle.png", color="yellow", triangle=True)
    _draw_shape(irrelevant_dir / "irrelevant_gray_bar.png", color="gray", bar=True)
    write_jsonl(out_dir / "samples.jsonl", rows)


def _draw_shape(path: Path, color: str, circle: bool = False, triangle: bool = False, bar: bool = False) -> None:
    palette = {
        "red": (220, 40, 40),
        "blue": (40, 90, 220),
        "green": (35, 170, 80),
        "yellow": (230, 205, 50),
        "gray": (120, 120, 120),
    }
    img = Image.new("RGB", (192, 192), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    fill = palette[color]
    if triangle:
        draw.polygon([(96, 35), (40, 150), (152, 150)], fill=fill)
    elif bar:
        draw.rectangle((40, 82, 152, 110), fill=fill)
    elif circle:
        draw.ellipse((48, 48, 144, 144), fill=fill)
    else:
        draw.rectangle((48, 48, 144, 144), fill=fill)
    img.save(path)


def _load_records(path: Path) -> list[EvalRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Records not found: {path}")
    records: list[EvalRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EvalRecord(**json.loads(line)))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


if __name__ == "__main__":
    raise SystemExit(main())
