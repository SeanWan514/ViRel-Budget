from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from virel_budget.schema import Budget, EvalRecord, PolicyDecision, budget_sort_key


@dataclass(frozen=True)
class Thresholds:
    tau: float
    rho: float | None
    dense_vem_min: float
    require_dense_reliance: bool = True


def calibrate_thresholds(
    records: Iterable[EvalRecord],
    tau_grid: list[float],
    rho_grid: list[float],
    dense_vem_min: float,
    prefer_rr: bool,
) -> Thresholds:
    validation = [r for r in records if r.split == "validation"]
    if not validation:
        return Thresholds(tau=tau_grid[0], rho=rho_grid[0] if prefer_rr and rho_grid else None, dense_vem_min=dense_vem_min)
    validation_sample_count = len({(r.sample_id, r.split) for r in validation if r.method == "dense"}) or len(
        {(r.sample_id, r.split) for r in validation}
    )
    best: tuple[float, float, float, float, Thresholds] | None = None
    rho_values: Sequence[float | None] = rho_grid if prefer_rr else [None]
    for tau in tau_grid:
        for rho in rho_values:
            thresholds = Thresholds(tau=tau, rho=rho, dense_vem_min=dense_vem_min)
            decisions = apply_policy(validation, thresholds=thresholds)
            if not decisions:
                continue
            coverage = len(decisions) / max(validation_sample_count, 1)
            supported_rate = sum(float(d.accepted) for d in decisions) / len(decisions)
            accuracy = sum(float(d.is_correct) for d in decisions) / len(decisions)
            cost = sum(d.cost for d in decisions) / len(decisions)
            score = (coverage, supported_rate, accuracy, -cost)
            candidate = (score[0], score[1], score[2], score[3], thresholds)
            if best is None or candidate[:4] > best[:4]:
                best = candidate
    if best is None:
        return Thresholds(tau=tau_grid[0], rho=rho_grid[0] if prefer_rr and rho_grid else None, dense_vem_min=dense_vem_min)
    return best[4]


def with_dense_reliance(thresholds: Thresholds, required: bool) -> Thresholds:
    return Thresholds(
        tau=thresholds.tau,
        rho=thresholds.rho,
        dense_vem_min=thresholds.dense_vem_min,
        require_dense_reliance=required,
    )


def apply_policy(
    records: Iterable[EvalRecord],
    thresholds: Thresholds,
    allowed_methods: Sequence[str] | None = None,
    label: str = "virel_budget",
) -> list[PolicyDecision]:
    """Select one cheapest visually supported method/budget per sample.

    The decision unit is a query, not a query-method pair. Only interventions for
    which the dense model has positive visual evidence are used as constraints.
    This matches the proposal's dense-reliant evaluation rule and avoids blaming
    pruning for samples where the dense model was not visually grounded.
    """

    allowed = set(allowed_methods) if allowed_methods is not None else None
    grouped = _group_by_sample(records)
    decisions: list[PolicyDecision] = []
    for (_sample_id, _split), items in sorted(grouped.items()):
        eligible_interventions = _eligible_interventions(items, thresholds)
        if not eligible_interventions:
            continue
        dense_records = _records_for(items, "dense", None, eligible_interventions)
        dense_aggregate = _aggregate_budget_records(dense_records) if dense_records else None
        candidates: list[tuple[Budget, float, str, list[EvalRecord], bool]] = []
        by_method_budget: dict[tuple[str, Budget], list[EvalRecord]] = {}
        for item in items:
            if item.method == "dense":
                continue
            if allowed is not None and item.method not in allowed:
                continue
            if item.intervention not in eligible_interventions:
                continue
            by_method_budget.setdefault((item.method, item.budget), []).append(item)
        for (method, budget), budget_records in by_method_budget.items():
            if not _covers_interventions(budget_records, eligible_interventions):
                continue
            aggregate = _aggregate_budget_records(budget_records)
            candidates.append((budget, float(aggregate["cost"]), method, budget_records, _budget_passes(budget_records, thresholds)))
        accepted = [candidate for candidate in candidates if candidate[4]]
        if accepted:
            budget, _cost, method, chosen_records, _passes = sorted(
                accepted,
                key=lambda c: (budget_sort_key(c[0], _dense_token_count(dense_aggregate, items)), c[1], c[2]),
            )[0]
            ordered_budgets = sorted({c[0] for c in candidates}, key=lambda b: budget_sort_key(b, _dense_token_count(dense_aggregate, items)))
            escalations = sum(1 for b in ordered_budgets if budget_sort_key(b, _dense_token_count(dense_aggregate, items)) < budget_sort_key(budget, _dense_token_count(dense_aggregate, items)))
            decisions.append(
                _make_decision(
                    label=label,
                    selected_method=method,
                    chosen_records=chosen_records,
                    dense_aggregate=dense_aggregate,
                    accepted=True,
                    escalations=escalations,
                    reason="accepted_visual_support" if escalations == 0 else "accepted_after_escalation",
                    eligible_interventions=eligible_interventions,
                )
            )
        elif dense_records:
            decisions.append(
                _make_decision(
                    label=label,
                    selected_method="dense",
                    chosen_records=dense_records,
                    dense_aggregate=dense_aggregate,
                    accepted=_budget_passes(dense_records, thresholds),
                    escalations=len({c[0] for c in candidates}),
                    reason="fallback_full",
                    eligible_interventions=eligible_interventions,
                )
            )
    return decisions


def apply_online_cascade_policy(
    records: Iterable[EvalRecord],
    thresholds: Thresholds,
    allowed_methods: Sequence[str] | None = None,
    label: str = "virel_budget_online",
) -> list[PolicyDecision]:
    """Select a budget with cumulative cost for failed earlier probes."""

    allowed = set(allowed_methods) if allowed_methods is not None else None
    grouped = _group_by_sample(records)
    decisions: list[PolicyDecision] = []
    for (_sample_id, _split), items in sorted(grouped.items()):
        eligible_interventions = _eligible_interventions(items, thresholds)
        if not eligible_interventions:
            continue
        dense_records = _records_for(items, "dense", None, eligible_interventions)
        dense_aggregate = _aggregate_budget_records(dense_records) if dense_records else None
        candidates: list[tuple[Budget, float, str, list[EvalRecord], bool]] = []
        by_method_budget: dict[tuple[str, Budget], list[EvalRecord]] = {}
        for item in items:
            if item.method == "dense":
                continue
            if allowed is not None and item.method not in allowed:
                continue
            if item.intervention not in eligible_interventions:
                continue
            by_method_budget.setdefault((item.method, item.budget), []).append(item)
        for (method, budget), budget_records in by_method_budget.items():
            if not _covers_interventions(budget_records, eligible_interventions):
                continue
            aggregate = _aggregate_budget_records(budget_records)
            candidates.append((budget, float(aggregate["cost"]), method, budget_records, _budget_passes(budget_records, thresholds)))
        ordered = sorted(
            candidates,
            key=lambda c: (budget_sort_key(c[0], _dense_token_count(dense_aggregate, items)), c[1], c[2]),
        )
        cumulative_cost = 0.0
        cumulative_latency = 0.0
        cumulative_energy = 0.0
        cumulative_measured_energy: float | None = None
        cumulative_proxy_energy: float | None = None
        for idx, (budget, _cost, method, chosen_records, passes) in enumerate(ordered):
            aggregate = _aggregate_budget_records(chosen_records)
            cumulative_cost += float(aggregate["cost"])
            cumulative_latency += float(aggregate["latency_ms"])
            cumulative_energy += float(aggregate.get("energy_joule") or 0.0)
            cumulative_measured_energy = _add_optional(
                cumulative_measured_energy, aggregate.get("measured_energy_joule")
            )
            cumulative_proxy_energy = _add_optional(cumulative_proxy_energy, aggregate.get("proxy_energy_joule"))
            if passes:
                decisions.append(
                    _make_decision(
                        label=label,
                        selected_method=method,
                        chosen_records=chosen_records,
                        dense_aggregate=dense_aggregate,
                        accepted=True,
                        escalations=idx,
                        reason="accepted_visual_support" if idx == 0 else "accepted_after_escalation",
                        eligible_interventions=eligible_interventions,
                        online_cost=cumulative_cost,
                        online_latency=cumulative_latency,
                        online_energy=cumulative_energy,
                        online_measured_energy=cumulative_measured_energy,
                        online_proxy_energy=cumulative_proxy_energy,
                    )
                )
                break
        else:
            if dense_records:
                aggregate = _aggregate_budget_records(dense_records)
                cumulative_cost += float(aggregate["cost"])
                cumulative_latency += float(aggregate["latency_ms"])
                cumulative_energy += float(aggregate.get("energy_joule") or 0.0)
                cumulative_measured_energy = _add_optional(
                    cumulative_measured_energy, aggregate.get("measured_energy_joule")
                )
                cumulative_proxy_energy = _add_optional(cumulative_proxy_energy, aggregate.get("proxy_energy_joule"))
                decisions.append(
                    _make_decision(
                        label=label,
                        selected_method="dense",
                        chosen_records=dense_records,
                        dense_aggregate=dense_aggregate,
                        accepted=_budget_passes(dense_records, thresholds),
                        escalations=len(ordered),
                        reason="fallback_full",
                        eligible_interventions=eligible_interventions,
                        online_cost=cumulative_cost,
                        online_latency=cumulative_latency,
                        online_energy=cumulative_energy,
                        online_measured_energy=cumulative_measured_energy,
                        online_proxy_energy=cumulative_proxy_energy,
                    )
                )
    return decisions


def eligible_interventions_by_sample(
    records: Iterable[EvalRecord],
    thresholds: Thresholds,
) -> dict[tuple[str, str], tuple[str, ...]]:
    grouped = _group_by_sample(records)
    return {key: _eligible_interventions(items, thresholds) for key, items in grouped.items()}


def apply_fixed_budget_policy(
    records: Iterable[EvalRecord],
    thresholds: Thresholds,
    method: str,
    budget: Budget,
    label: str,
) -> list[PolicyDecision]:
    grouped = _group_by_sample(records)
    decisions: list[PolicyDecision] = []
    for (_sample_id, _split), items in sorted(grouped.items()):
        eligible_interventions = _eligible_interventions(items, thresholds)
        if not eligible_interventions:
            continue
        chosen_records = _records_for(items, method, budget, eligible_interventions)
        if not chosen_records or not _covers_interventions(chosen_records, eligible_interventions):
            continue
        dense_records = _records_for(items, "dense", None, eligible_interventions)
        dense_aggregate = _aggregate_budget_records(dense_records) if dense_records else None
        decisions.append(
            _make_decision(
                label=label,
                selected_method=method,
                chosen_records=chosen_records,
                dense_aggregate=dense_aggregate,
                accepted=_budget_passes(chosen_records, thresholds),
                escalations=0,
                reason="fixed_budget",
                eligible_interventions=eligible_interventions,
            )
        )
    return decisions


def choose_accuracy_only_budget(
    validation_records: Iterable[EvalRecord],
    method: str,
    budget_schedule: list[Budget],
    tolerance: float = 0.0,
) -> Budget:
    accuracies: dict[Budget, float] = {}
    for budget in budget_schedule:
        records = [r for r in validation_records if r.method == method and r.budget == budget]
        if not records:
            continue
        by_sample: dict[str, list[EvalRecord]] = {}
        for record in records:
            by_sample.setdefault(record.sample_id, []).append(record)
        sample_correct = [all(item.is_correct for item in items) for items in by_sample.values()]
        accuracies[budget] = sum(float(v) for v in sample_correct) / max(len(sample_correct), 1)
    if not accuracies:
        return budget_schedule[-1]
    best_acc = max(accuracies.values())
    eligible = [b for b in budget_schedule if b in accuracies and accuracies[b] >= best_acc - tolerance]
    return sorted(eligible, key=lambda b: budget_sort_key(b, None))[0]


def _group_by_sample(records: Iterable[EvalRecord]) -> dict[tuple[str, str], list[EvalRecord]]:
    grouped: dict[tuple[str, str], list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault((record.sample_id, record.split), []).append(record)
    return grouped


def _eligible_interventions(items: list[EvalRecord], thresholds: Thresholds) -> tuple[str, ...]:
    dense_records = [r for r in items if r.method == "dense"]
    dense_required_vem = max(thresholds.dense_vem_min, thresholds.tau)
    if dense_records:
        if not thresholds.require_dense_reliance:
            return tuple(sorted({r.intervention for r in dense_records}))
        return tuple(sorted(r.intervention for r in dense_records if r.is_correct and r.vem >= dense_required_vem))
    return tuple(sorted({r.intervention for r in items if r.dense_vem is None or r.dense_vem >= dense_required_vem}))


def _records_for(
    items: list[EvalRecord],
    method: str,
    budget: Budget | None,
    interventions: tuple[str, ...],
) -> list[EvalRecord]:
    wanted = set(interventions)
    return [
        item
        for item in items
        if item.method == method
        and (budget is None or item.budget == budget)
        and item.intervention in wanted
    ]


def _covers_interventions(records: list[EvalRecord], interventions: tuple[str, ...]) -> bool:
    return {r.intervention for r in records} >= set(interventions)


def _passes(record: EvalRecord, thresholds: Thresholds) -> bool:
    if record.vem < thresholds.tau:
        return False
    if thresholds.rho is not None:
        if record.reliance_retention is None:
            return False
        if record.reliance_retention < thresholds.rho:
            return False
    return True


def _budget_passes(records: list[EvalRecord], thresholds: Thresholds) -> bool:
    if not records:
        return False
    return all(_passes(record, thresholds) for record in records)


def _aggregate_budget_records(records: list[EvalRecord]) -> dict:
    first = records[0]
    rr_values = [r.reliance_retention for r in records if r.reliance_retention is not None]
    dense_values = [r.dense_vem for r in records if r.dense_vem is not None]
    return {
        "sample_id": first.sample_id,
        "split": first.split,
        "dataset": first.dataset,
        "budget": first.budget,
        "answer": first.answer,
        "gold_answer": first.gold_answer,
        "is_correct": all(r.is_correct for r in records),
        "vem": min(r.vem for r in records),
        "dense_vem": min(dense_values) if dense_values else None,
        "reliance_retention": min(rr_values) if rr_values else None,
        "cost": sum(r.cost for r in records),
        "latency_ms": sum(r.latency_ms for r in records),
        "energy_joule": sum(r.energy_joule for r in records),
        "api_cost_usd": _sum_optional([r.api_cost_usd for r in records]),
        "measured_energy_joule": _sum_optional([r.measured_energy_joule for r in records]),
        "proxy_energy_joule": _sum_optional([r.proxy_energy_joule for r in records]),
        "support_metric": first.support_metric,
        "token_count": first.token_count,
    }


def _make_decision(
    *,
    label: str,
    selected_method: str,
    chosen_records: list[EvalRecord],
    dense_aggregate: dict | None,
    accepted: bool,
    escalations: int,
    reason: str,
    eligible_interventions: tuple[str, ...],
    online_cost: float | None = None,
    online_latency: float | None = None,
    online_energy: float | None = None,
    online_measured_energy: float | None = None,
    online_proxy_energy: float | None = None,
) -> PolicyDecision:
    chosen = _aggregate_budget_records(chosen_records)
    dense_latency = float(dense_aggregate["latency_ms"]) if dense_aggregate else None
    dense_cost = float(dense_aggregate["cost"]) if dense_aggregate else None
    dense_tokens = int(dense_aggregate["token_count"]) if dense_aggregate else None
    reported_latency = float(online_latency) if online_latency is not None else float(chosen["latency_ms"])
    reported_cost = float(online_cost) if online_cost is not None else float(chosen["cost"])
    speedup = dense_latency / reported_latency if dense_latency and reported_latency else None
    retained_ratio = float(chosen["token_count"]) / dense_tokens if dense_tokens else None
    token_reduction = 1.0 - retained_ratio if retained_ratio is not None else None
    return PolicyDecision(
        sample_id=str(chosen["sample_id"]),
        split=str(chosen["split"]),
        dataset=str(chosen["dataset"]),
        gold_answer=str(chosen["gold_answer"]),
        method=label,
        selected_method=selected_method,
        selected_budget=chosen["budget"],
        accepted=accepted,
        escalations=escalations,
        answer=str(chosen["answer"]),
        is_correct=bool(chosen["is_correct"]),
        vem=float(chosen["vem"]),
        dense_vem=chosen["dense_vem"],
        reliance_retention=chosen["reliance_retention"],
        cost=reported_cost,
        latency_ms=reported_latency,
        token_count=int(chosen["token_count"]),
        reason=reason,
        eligible_interventions=eligible_interventions,
        support_status="supported" if accepted else "unsupported",
        dense_latency_ms=dense_latency,
        dense_token_count=dense_tokens,
        dense_cost=dense_cost,
        speedup_vs_dense=speedup,
        token_reduction=token_reduction,
        retained_token_ratio=retained_ratio,
        online_cumulative_cost=online_cost,
        online_cumulative_latency_ms=online_latency,
        online_cumulative_energy_joule=online_energy,
        api_cost_usd=chosen.get("api_cost_usd"),
        measured_energy_joule=(
            online_measured_energy if online_measured_energy is not None else chosen.get("measured_energy_joule")
        ),
        proxy_energy_joule=online_proxy_energy if online_proxy_energy is not None else chosen.get("proxy_energy_joule"),
        support_metric=str(chosen.get("support_metric") or "score_vem"),
    )


def _dense_token_count(dense_aggregate: dict | None, items: list[EvalRecord]) -> int | None:
    if dense_aggregate:
        return int(dense_aggregate["token_count"])
    token_counts = [r.token_count for r in items if r.budget == "full"]
    return max(token_counts) if token_counts else None


def _sum_optional(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) if clean else None


def _add_optional(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    if current is None:
        return float(value)
    return float(current) + float(value)
