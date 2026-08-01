from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = ("eligible_intervention_yes_no_uncertain", "artifact_present_yes_no_uncertain")


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["blind_id"]: row for row in csv.DictReader(handle)}


def kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    labels = sorted(set(left) | set(right))
    lc, rc = Counter(left), Counter(right)
    expected = sum((lc[label] / len(left)) * (rc[label] / len(right)) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/qualitative_annotation_48")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    a = read(root / "annotation_form_A.csv")
    b = read(root / "annotation_form_B.csv")
    if set(a) != set(b):
        raise ValueError("Annotator forms contain different blind IDs")
    output: dict[str, Any] = {"n": len(a), "fields": {}}
    complete = True
    for field in FIELDS:
        pairs = [(a[key][field].strip().lower(), b[key][field].strip().lower()) for key in sorted(a)]
        complete &= all(left and right for left, right in pairs)
        answered = [(left, right) for left, right in pairs if left and right]
        left = [item[0] for item in answered]
        right = [item[1] for item in answered]
        output["fields"][field] = {
            "answered_pairs": len(answered),
            "raw_agreement": sum(x == y for x, y in answered) / len(answered) if answered else None,
            "cohen_kappa": kappa(left, right),
            "annotator_a_distribution": dict(Counter(left)),
            "annotator_b_distribution": dict(Counter(right)),
        }
    output["complete"] = complete
    (root / "annotation_agreement.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.require_complete and not complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
