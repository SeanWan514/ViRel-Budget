from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge restart-attempt NVIDIA telemetry.")
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sources = sorted(Path(args.attempt_root).glob("attempt_*/nvidia_smi_samples.csv"))
    if not sources:
        raise FileNotFoundError("No attempt telemetry CSV files found")
    rows = []
    fieldnames = None
    for source in sources:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = fieldnames or reader.fieldnames
            if reader.fieldnames != fieldnames:
                raise ValueError(f"Telemetry schema mismatch: {source}")
            for row in reader:
                try:
                    float(row["epoch"])
                    float(row["power_w"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append(row)
    rows.sort(key=lambda row: float(row["epoch"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "sources": [str(path) for path in sources],
        "valid_samples": len(rows),
        "first_epoch": float(rows[0]["epoch"]) if rows else None,
        "last_epoch": float(rows[-1]["epoch"]) if rows else None,
    }
    (output.parent / "merged_telemetry_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
