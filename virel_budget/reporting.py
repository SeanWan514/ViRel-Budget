from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Any

from virel_budget.datasets.jsonl import write_jsonl
from virel_budget.schema import EvalRecord, PolicyDecision


def write_records(path: str | Path, records: Iterable[Any]) -> None:
    rows = [_to_jsonable(r) for r in records]
    write_jsonl(path, rows)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify(row.get(k)) for k in fieldnames})


def write_json(path: str | Path, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_to_jsonable(obj), indent=2, sort_keys=True), encoding="utf-8")


def write_frontier_svg(path: str | Path, rows: list[dict], title: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    points = [
        r
        for r in rows
        if r.get("mean_latency_ms") is not None and r.get("mean_rr") is not None
    ]
    width, height = 900, 540
    margin = 70
    if not points:
        out.write_text(_empty_svg(width, height, title), encoding="utf-8")
        return
    xs = [float(p["mean_latency_ms"]) for p in points]
    ys = [float(p["mean_rr"]) for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_max += 1.0
    colors = {
        "random": "#2f80ed",
        "center": "#27ae60",
        "saliency": "#eb5757",
        "dense": "#111111",
    }
    circles = []
    labels = []
    for point in points:
        x = margin + (float(point["mean_latency_ms"]) - x_min) / (x_max - x_min) * (width - 2 * margin)
        y = height - margin - (float(point["mean_rr"]) - y_min) / (y_max - y_min) * (height - 2 * margin)
        method = str(point.get("method", "method"))
        budget = str(point.get("budget", "budget"))
        color = colors.get(method, "#6f42c1")
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" opacity="0.86" />')
        labels.append(f'<text x="{x + 10:.1f}" y="{y - 8:.1f}" font-size="12">{_escape(method)}:{_escape(budget)}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2:.0f}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{_escape(title)}</text>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222" stroke-width="1.5"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222" stroke-width="1.5"/>
  <text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="14">Mean latency / cost proxy (ms)</text>
  <text x="20" y="{height / 2:.0f}" transform="rotate(-90 20 {height / 2:.0f})" text-anchor="middle" font-family="Arial" font-size="14">Mean reliance retention</text>
  <text x="{margin}" y="{height - margin + 22}" font-family="Arial" font-size="11">{x_min:.2f}</text>
  <text x="{width - margin - 35}" y="{height - margin + 22}" font-family="Arial" font-size="11">{x_max:.2f}</text>
  <text x="{margin - 50}" y="{height - margin}" font-family="Arial" font-size="11">{y_min:.2f}</text>
  <text x="{margin - 50}" y="{margin + 4}" font-family="Arial" font-size="11">{y_max:.2f}</text>
  {"".join(circles)}
  {"".join(labels)}
</svg>
"""
    out.write_text(svg, encoding="utf-8")


def qualitative_examples(records: list[EvalRecord], max_examples: int = 12) -> list[dict]:
    candidates = [
        r
        for r in records
        if r.is_correct and r.dense_vem is not None and r.vem < r.dense_vem and str(r.budget) != "full"
    ]
    candidates.sort(key=lambda r: (r.vem - (r.dense_vem or 0.0), r.cost))
    return [
        {
            "sample_id": r.sample_id,
            "method": r.method,
            "budget": r.budget,
            "intervention": r.intervention,
            "answer": r.answer,
            "gold_answer": r.gold_answer,
            "vem": r.vem,
            "dense_vem": r.dense_vem,
            "delta_vem": r.delta_vem,
            "reliance_retention": r.reliance_retention,
            "note": "accuracy preserved while visual evidence margin dropped",
        }
        for r in candidates[:max_examples]
    ]


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _stringify(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_to_jsonable(value), sort_keys=True)
    return value


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _empty_svg(width: int, height: int, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" font-family="Arial" font-size="20">{_escape(title)}: no plottable points</text>
</svg>
"""
