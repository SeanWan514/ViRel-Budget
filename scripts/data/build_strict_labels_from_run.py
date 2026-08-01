from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit.framework_repair_audit import (  # noqa: E402
    _build_actions,
    _label_and_monotonicity_rows,
    _read_jsonl,
)
from virel_budget.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build repaired strict-safety labels from one completed run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    sample_path = resolve_path(config, config["dataset"]["path"])
    assert sample_path is not None
    samples = _read_jsonl(sample_path)
    sample_by_id = {row["sample_id"]: row for row in samples}
    records = _read_jsonl(Path(args.records))
    actions = _build_actions(records, [], sample_by_id)
    configured_methods = list(config["pruning"]["methods"])
    if len(configured_methods) != 1:
        raise ValueError("Strict-label builder requires exactly one configured pruning method")
    method = str(configured_methods[0])
    labels, monotonicity = _label_and_monotonicity_rows({method: actions})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "safe_budget_labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    (output / "budget_monotonicity_audit.json").write_text(
        json.dumps(monotonicity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "method": method,
        "sample_count": len(actions),
        "label_rows": len(labels),
        "strict_target": "original-answer fidelity AND dense intervention-answer trajectory fidelity",
        "source_records": str(Path(args.records)),
        "source_config": str(Path(args.config)),
    }
    (output / "strict_label_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
