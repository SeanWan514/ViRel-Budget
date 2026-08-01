from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PowerSamples:
    rows: list[dict[str, float]]
    epochs: list[float]

    def __len__(self) -> int:
        return len(self.rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Attribute nvidia-smi power samples to timestamped ViRel records.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--out-records-jsonl",
        default=None,
        help="Optional records.jsonl copy with measured_energy_joule merged back into each EvalRecord.",
    )
    args = parser.parse_args()

    records = _read_jsonl(Path(args.records))
    samples = _read_power_samples(Path(args.samples))
    record_rows = [_attribute_record(record, samples) for record in records]
    call_rows = _unique_call_rows(records, samples)
    by_method_budget = _summarize(record_rows, call_rows)
    out = {
        "records": len(records),
        "unique_model_calls": len(call_rows),
        "monitor_samples": len(samples),
        "attribution_method": (
            "For each model call, average nvidia-smi power samples whose timestamps fall inside "
            "the backend start/end epoch window, then multiply by call duration. Per-record energy "
            "is original-call plus intervention-call energy. If a short call falls between two 200ms "
            "monitor samples, use the nearest measured power sample to the call midpoint. Method/budget "
            "totals deduplicate cached original-image calls so they reflect actual measured calls made "
            "during the run. This is measured GPU power attribution, not CPU/platform energy."
        ),
        "records_with_measured_energy": sum(row["measured_energy_joule"] is not None for row in record_rows),
        "unique_calls_with_measured_energy": sum(row["measured_energy_joule"] is not None for row in call_rows),
        "total_attributed_unique_call_energy_joule": _sum_optional(row["measured_energy_joule"] for row in call_rows),
        "by_method_budget": by_method_budget,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(Path(args.out_csv), by_method_budget)
    if args.out_records_jsonl:
        _write_records_with_measured_energy(Path(args.out_records_jsonl), records, record_rows)
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_power_samples(path: Path) -> PowerSamples:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "epoch": _row_epoch(row),
                        "power_w": float(str(row["power_w"]).strip()),
                    }
                )
            except Exception:
                continue
    rows.sort(key=lambda item: item["epoch"])
    return PowerSamples(rows=rows, epochs=[row["epoch"] for row in rows])


def _row_epoch(row: dict[str, str]) -> float:
    epoch = (row.get("epoch") or "").strip()
    if epoch:
        return float(epoch)
    return _timestamp_to_epoch(str(row["timestamp"]).strip())


def _timestamp_to_epoch(value: str) -> float:
    dt = datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f")
    return time.mktime(dt.timetuple()) + dt.microsecond / 1_000_000


def _attribute_record(record: dict[str, Any], samples: PowerSamples) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    call_rows = []
    for role in ["original_backend", "intervened_backend"]:
        call = metadata.get(role) or {}
        energy = _attribute_call(call, samples)
        call_rows.append(energy)
    measured_energy = _sum_optional(row["measured_energy_joule"] for row in call_rows)
    return {
        "sample_id": record.get("sample_id"),
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "method": record.get("method"),
        "budget": record.get("budget"),
        "intervention": record.get("intervention"),
        "latency_ms": _float_or_none(record.get("latency_ms")),
        "proxy_energy_joule": _float_or_none(record.get("proxy_energy_joule")),
        "measured_energy_joule": measured_energy,
    }


def _unique_call_rows(records: list[dict[str, Any]], samples: PowerSamples) -> list[dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        for role in ["original_backend", "intervened_backend"]:
            call = metadata.get(role) or {}
            start = _float_or_none(call.get("start_epoch"))
            end = _float_or_none(call.get("end_epoch"))
            if start is None or end is None:
                continue
            key = (
                record.get("sample_id"),
                record.get("method"),
                str(record.get("budget")),
                role,
                round(start, 6),
                round(end, 6),
            )
            if key in rows:
                continue
            energy = _attribute_call(call, samples)
            rows[key] = {
                "sample_id": record.get("sample_id"),
                "method": record.get("method"),
                "budget": record.get("budget"),
                "role": role,
                "duration_ms": max((end - start) * 1000.0, 0.0),
                "measured_energy_joule": energy["measured_energy_joule"],
            }
    return list(rows.values())


def _attribute_call(call: dict[str, Any], samples: PowerSamples) -> dict[str, float | None]:
    start = _float_or_none(call.get("start_epoch"))
    end = _float_or_none(call.get("end_epoch"))
    if start is None or end is None or end <= start:
        return {"measured_energy_joule": None, "mean_power_w": None}
    left = bisect_left(samples.epochs, start)
    right = bisect_right(samples.epochs, end)
    powers = [sample["power_w"] for sample in samples.rows[left:right]]
    if not powers:
        nearest = _nearest_sample(samples, (start + end) / 2.0)
        if nearest is None:
            return {"measured_energy_joule": None, "mean_power_w": None}
        powers = [nearest["power_w"]]
    mean_power = statistics.fmean(powers)
    return {"measured_energy_joule": mean_power * (end - start), "mean_power_w": mean_power}


def _nearest_sample(samples: PowerSamples, epoch: float) -> dict[str, float] | None:
    if not samples:
        return None
    index = bisect_left(samples.epochs, epoch)
    candidates = []
    if index < len(samples.rows):
        candidates.append(samples.rows[index])
    if index:
        candidates.append(samples.rows[index - 1])
    return min(candidates, key=lambda sample: abs(sample["epoch"] - epoch))


def _summarize(records: list[dict[str, Any]], calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((str(record["method"]), str(record["budget"])), []).append(record)
    call_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for call in calls:
        call_groups.setdefault((str(call["method"]), str(call["budget"])), []).append(call)
    rows = []
    for (method, budget), group in sorted(groups.items(), key=lambda item: (item[0][0], _budget_sort(item[0][1]))):
        call_group = call_groups.get((method, budget), [])
        measured_calls = [row["measured_energy_joule"] for row in call_group if row["measured_energy_joule"] is not None]
        proxy = [row["proxy_energy_joule"] for row in group if row["proxy_energy_joule"] is not None]
        latency = [row["latency_ms"] for row in group if row["latency_ms"] is not None]
        total_measured = sum(measured_calls) if measured_calls else None
        rows.append(
            {
                "method": method,
                "budget": budget,
                "n": len(group),
                "n_unique_calls": len(call_group),
                "n_measured_calls": len(measured_calls),
                "total_measured_energy_joule": total_measured,
                "mean_measured_energy_per_record_joule": (total_measured / len(group)) if total_measured is not None and group else None,
                "mean_measured_energy_per_call_joule": statistics.fmean(measured_calls) if measured_calls else None,
                "total_proxy_energy_joule": sum(proxy) if proxy else None,
                "mean_proxy_energy_joule": statistics.fmean(proxy) if proxy else None,
                "mean_latency_ms": statistics.fmean(latency) if latency else None,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_records_with_measured_energy(
    path: Path,
    records: list[dict[str, Any]],
    record_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record, measured in zip(records, record_rows, strict=True):
            updated = dict(record)
            measured_energy = measured.get("measured_energy_joule")
            updated["measured_energy_joule"] = measured_energy
            if measured_energy is not None:
                updated["energy_joule"] = measured_energy
            updated["proxy_energy_joule"] = measured.get("proxy_energy_joule")
            metadata = dict(updated.get("metadata") or {})
            metadata["measured_energy_attribution"] = {
                "source": "nvidia-smi 200ms polling",
                "value_scope": "per EvalRecord original-call plus intervention-call energy",
                "measured_energy_joule": measured_energy,
            }
            updated["metadata"] = metadata
            f.write(json.dumps(updated, sort_keys=True) + "\n")


def _sum_optional(values: Any) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return float(sum(kept)) if kept else None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _budget_sort(value: str) -> int:
    if value == "full":
        return 10**9
    try:
        return int(value)
    except Exception:
        return 10**8


if __name__ == "__main__":
    raise SystemExit(main())
