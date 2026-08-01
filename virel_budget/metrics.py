from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable

from virel_budget.schema import EvalRecord, PolicyDecision, normalize_answer


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def visual_evidence_margin(original_logprob: float, intervened_logprob: float) -> float:
    return float(original_logprob - intervened_logprob)


def reliance_retention(vem: float, dense_vem: float | None, min_denominator: float = 1e-6) -> float | None:
    if dense_vem is None or abs(dense_vem) < min_denominator:
        return None
    return float(vem / dense_vem)


def cost_from_latency(latency_ms: float, dollar_per_1k_ms: float) -> float:
    return float(latency_ms / 1000.0 * dollar_per_1k_ms)


def summarize_records(records: Iterable[EvalRecord]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[EvalRecord]] = {}
    for record in records:
        groups.setdefault((record.method, str(record.budget), record.intervention), []).append(record)
    rows = []
    for (method, budget, intervention), items in sorted(groups.items()):
        supported = [r for r in items if r.vem >= 0]
        correct = [r for r in items if r.is_correct]
        cost_total = sum(r.cost for r in items)
        supported_count = max(len(supported), 1)
        correct_count = max(len(correct), 1)
        rows.append(
            {
                "method": method,
                "budget": budget,
                "intervention": intervention,
                "n": len(items),
                "accuracy": len(correct) / len(items) if items else 0.0,
                "mean_vem": mean([r.vem for r in items]) if items else 0.0,
                "mean_rr": _mean_optional([r.reliance_retention for r in items]),
                "mean_delta_vem": _mean_optional([r.delta_vem for r in items]),
                "shortcut_persistence_rate": mean([float(r.shortcut_persistence) for r in items]) if items else 0.0,
                "mean_latency_ms": mean([r.latency_ms for r in items]) if items else 0.0,
                "mean_token_count": mean([r.token_count for r in items]) if items else 0.0,
                "total_cost": cost_total,
                "cost_per_accurate_answer": cost_total / correct_count,
                "cost_per_visually_supported_answer": cost_total / supported_count,
            }
        )
    return rows


def summarize_decisions(decisions: Iterable[PolicyDecision]) -> dict:
    items = list(decisions)
    if not items:
        return {}
    selected = Counter(str(d.selected_budget) for d in items)
    methods = Counter(str(d.selected_method) for d in items)
    accepted = [d for d in items if d.accepted]
    correct = [d for d in items if d.is_correct]
    total_cost = sum(d.cost for d in items)
    total_measured_energy = _sum_optional([d.measured_energy_joule for d in items])
    total_proxy_energy = _sum_optional([d.proxy_energy_joule for d in items])
    return {
        "n": len(items),
        "accepted_rate": len(accepted) / len(items),
        "accuracy": len(correct) / len(items),
        "visually_supported_rate": len(accepted) / len(items),
        "mean_vem": mean([d.vem for d in items]),
        "mean_rr": _mean_optional([d.reliance_retention for d in items]),
        "mean_clipped_rr": _mean_optional([_clip_optional(d.reliance_retention, 0.0, 2.0) for d in items]),
        "mean_latency_ms": mean([d.latency_ms for d in items]),
        "mean_token_count": mean([d.token_count for d in items]),
        "mean_escalations": mean([d.escalations for d in items]),
        "mean_speedup_vs_dense": _mean_optional([d.speedup_vs_dense for d in items]),
        "mean_retained_token_ratio": _mean_optional([d.retained_token_ratio for d in items]),
        "mean_token_reduction": _mean_optional([d.token_reduction for d in items]),
        "total_cost": total_cost,
        "cost_per_accurate_answer": total_cost / max(len(correct), 1),
        "cost_per_visually_supported_answer": total_cost / max(len(accepted), 1),
        "mean_api_cost_usd": _mean_optional([d.api_cost_usd for d in items]),
        "total_api_cost_usd": _sum_optional([d.api_cost_usd for d in items]),
        "mean_measured_energy_joule": _mean_optional([d.measured_energy_joule for d in items]),
        "total_measured_energy_joule": total_measured_energy,
        "measured_energy_per_accurate_answer_joule": (
            total_measured_energy / max(len(correct), 1) if total_measured_energy is not None else None
        ),
        "measured_energy_per_visually_supported_answer_joule": (
            total_measured_energy / max(len(accepted), 1) if total_measured_energy is not None else None
        ),
        "mean_proxy_energy_joule": _mean_optional([d.proxy_energy_joule for d in items]),
        "total_proxy_energy_joule": total_proxy_energy,
        "proxy_energy_per_visually_supported_answer_joule": (
            total_proxy_energy / max(len(accepted), 1) if total_proxy_energy is not None else None
        ),
        "mean_online_cumulative_cost": _mean_optional([d.online_cumulative_cost for d in items]),
        "mean_online_cumulative_latency_ms": _mean_optional([d.online_cumulative_latency_ms for d in items]),
        "selected_budget_distribution": dict(sorted(selected.items())),
        "selected_method_distribution": dict(sorted(methods.items())),
        "dense_avoidance_rate": mean(
            [float(d.selected_method != "dense" and str(d.selected_budget) != "full") for d in items]
        ),
        "fallback_full_rate": mean([float(d.reason == "fallback_full") for d in items]),
    }


def _mean_optional(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(mean(clean))


def _sum_optional(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(sum(clean))


def _clip_optional(value: float | None, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    return min(max(float(value), lower), upper)
