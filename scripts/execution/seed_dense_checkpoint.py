from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.schema import EvalRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse a verified dense reference in a restart-safe grid checkpoint."
    )
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--target-output-dir", required=True)
    parser.add_argument(
        "--expected-dense-records",
        type=int,
        default=4000,
        help="Exact number of unique dense intervention records required.",
    )
    args = parser.parse_args()
    source = Path(args.source_records)
    target = Path(args.target_output_dir) / "records_checkpoint.jsonl"
    if target.exists() and target.stat().st_size:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
    allowed = {field.name for field in fields(EvalRecord)}
    dense = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("method") != "dense":
            continue
        normalized = {key: value for key, value in row.items() if key in allowed}
        EvalRecord(**normalized)
        dense.append(normalized)
    keys = {
        (row["sample_id"], row["method"], str(row["budget"]), row["intervention"])
        for row in dense
    }
    if len(dense) != args.expected_dense_records or len(keys) != args.expected_dense_records:
        raise ValueError(
            f"Expected {args.expected_dense_records:,} unique dense intervention records; "
            f"found {len(dense)} / {len(keys)}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in dense:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "seeded",
                "dense_records": len(dense),
                "expected_dense_records": args.expected_dense_records,
                "target": str(target),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
