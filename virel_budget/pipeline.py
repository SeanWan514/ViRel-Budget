from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Any

import pandas as pd

from virel_budget.backends.base import VLMBackend
from virel_budget.backends.deterministic import DeterministicBackend
from virel_budget.backends.fastv_llava import FastVLlavaBackend
from virel_budget.backends.openrouter import OpenRouterBackend
from virel_budget.backends.scope_llava import ScopeLlavaBackend
from virel_budget.backends.smolvlm import SmolVLMBackend
from virel_budget.config import resolve_path
from virel_budget.datasets.jsonl import load_jsonl_samples
from virel_budget.images import materialize_interventions
from virel_budget.metrics import (
    cost_from_latency,
    exact_match,
    reliance_retention,
    summarize_decisions,
    summarize_records,
    visual_evidence_margin,
)
from virel_budget.policies import (
    apply_fixed_budget_policy,
    apply_online_cascade_policy,
    apply_policy,
    calibrate_thresholds,
    choose_accuracy_only_budget,
    eligible_interventions_by_sample,
    with_dense_reliance,
)
from virel_budget.reporting import (
    qualitative_examples,
    write_csv,
    write_frontier_svg,
    write_json,
    write_records,
)
from virel_budget.schema import Budget, EvalRecord, Sample


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_root_dir"])
    out_dir = resolve_path(config, config["outputs"]["dir"])
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    _log(config, f"[virel] Loading samples from {config['dataset']['path']}")
    samples = _load_samples(config)
    _log(config, f"[virel] Loaded {len(samples)} samples. Initializing backend '{config['backend']['name']}'...")
    backend = _make_backend(config)
    try:
        records = _evaluate_all(config, samples, backend, out_dir, seed)
    finally:
        backend.close()
    _log(config, f"[virel] Evaluation complete: {len(records)} records. Calibrating policy...")
    analysis = analyze_records(config, records, out_dir)
    return {
        "run_name": config["run_name"],
        "output_dir": str(out_dir),
        "n_samples": len(samples),
        "n_records": len(records),
        "root": str(root),
        **analysis,
    }


def analyze_records(config: dict[str, Any], records: list[EvalRecord], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_config = config["policy"]
    thresholds = calibrate_thresholds(
        records,
        tau_grid=[float(x) for x in policy_config["tau_grid"]],
        rho_grid=[float(x) for x in policy_config.get("rho_grid", [])],
        dense_vem_min=float(policy_config.get("dense_vem_min", 0.0)),
        prefer_rr=bool(policy_config.get("prefer_rr", True)),
    )
    test_records = [r for r in records if r.split == config["dataset"].get("test_split", "test")]
    has_pruning_methods = bool(config["pruning"].get("methods"))
    if has_pruning_methods:
        decisions = apply_policy(test_records, thresholds)
    else:
        decisions = apply_fixed_budget_policy(
            test_records,
            thresholds,
            "dense",
            config["pruning"].get("dense_budget", "full"),
            "dense_only",
        )
    dense_reliance_audit = _dense_reliance_audit(test_records, thresholds)
    baseline_decisions = _policy_decision_sets(config, records, test_records, decisions, thresholds)
    baseline_comparison = {name: summarize_decisions(items) for name, items in baseline_decisions.items() if items}
    budget_curve = _budget_curve(config, test_records, thresholds)
    macro_rows = _policy_macro_rows(baseline_decisions)
    ci_rows = _bootstrap_ci_rows(baseline_decisions, int(config.get("analysis", {}).get("bootstrap_iterations", 500)), int(config.get("seed", 13)))
    paired_ci_rows = _paired_bootstrap_ci_rows(baseline_decisions, int(config.get("analysis", {}).get("bootstrap_iterations", 500)), int(config.get("seed", 13)))
    ablation_rows = _ablation_rows(config, test_records, thresholds)
    summary_rows = summarize_records(records)
    decision_summary = summarize_decisions(decisions)
    artifacts = _write_outputs(
        out_dir,
        records,
        decisions,
        summary_rows,
        decision_summary,
        baseline_comparison,
        budget_curve,
        dense_reliance_audit,
        macro_rows,
        ci_rows,
        paired_ci_rows,
        ablation_rows,
        thresholds,
        config,
    )
    return {
        "n_decisions": len(decisions),
        "thresholds": {
            "tau": thresholds.tau,
            "rho": thresholds.rho,
            "dense_vem_min": thresholds.dense_vem_min,
        },
        "decision_summary": decision_summary,
        "dense_reliance_audit": dense_reliance_audit,
        "baseline_comparison": baseline_comparison,
        "macro_rows": macro_rows,
        "ci_rows": ci_rows,
        "paired_ci_rows": paired_ci_rows,
        "ablation_rows": ablation_rows,
        "artifacts": artifacts,
    }


def _load_samples(config: dict[str, Any]) -> list[Sample]:
    dataset = config["dataset"]
    path = resolve_path(config, dataset["path"])
    assert path is not None
    samples = load_jsonl_samples(path, dataset_name=dataset["name"], limit=dataset.get("limit"))
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples


def _make_backend(config: dict[str, Any]) -> VLMBackend:
    spec = config["backend"]
    name = spec["name"]
    if name == "deterministic":
        dense_count = int(config.get("cost", {}).get("dense_token_count") or 64)
        return DeterministicBackend(dense_token_count=dense_count, profile=str(spec.get("profile", "offline-sanity")))
    if name == "openrouter":
        return OpenRouterBackend(
            model_id=str(spec["model_id"]),
            api_key_env=str(spec.get("api_key_env", "OPENROUTER_API_KEY")),
            referer=str(spec.get("referer", "https://localhost")),
            title=str(spec.get("title", "ViRel-Budget")),
            timeout_s=float(spec.get("timeout_s", 120.0)),
            max_tokens=int(spec.get("max_tokens", 16)),
            temperature=float(spec.get("temperature", 0.0)),
        )
    if name == "fastv_llava":
        return FastVLlavaBackend(
            model_path=str(spec.get("model_path", "liuhaotian/llava-v1.5-7b")),
            fastv_repo=str(spec.get("fastv_repo", "/workspace/virel_external/FastV")),
            device=str(spec.get("device", "cuda")),
            max_new_tokens=int(spec.get("max_new_tokens", 24)),
            conv_mode=spec.get("conv_mode"),
            image_aspect_ratio=str(spec.get("image_aspect_ratio", "pad")),
            fastv_agg_layer=int(spec.get("fastv_agg_layer", 2)),
            load_8bit=bool(spec.get("load_8bit", False)),
            load_4bit=bool(spec.get("load_4bit", False)),
            instrument_features=bool(spec.get("instrument_features", False)),
        )
    if name == "scope_llava":
        return ScopeLlavaBackend(
            model_path=str(spec.get("model_path", "liuhaotian/llava-v1.5-7b")),
            scope_repo=str(spec.get("scope_repo", "/workspace/virel_external/SCOPE")),
            device=str(spec.get("device", "cuda")),
            max_new_tokens=int(spec.get("max_new_tokens", 24)),
            conv_mode=spec.get("conv_mode"),
            image_aspect_ratio=str(spec.get("image_aspect_ratio", "pad")),
            load_8bit=bool(spec.get("load_8bit", False)),
            load_4bit=bool(spec.get("load_4bit", False)),
            instrument_features=bool(spec.get("instrument_features", False)),
        )
    if name == "smolvlm":
        return SmolVLMBackend(
            model_id=str(spec.get("model_id", "HuggingFaceTB/SmolVLM-500M-Instruct")),
            device=str(spec.get("device", "cuda")),
            max_new_tokens=int(spec.get("max_new_tokens", 24)),
            torch_dtype=str(spec.get("torch_dtype", "float16")),
            trust_remote_code=bool(spec.get("trust_remote_code", True)),
        )
    raise ValueError(f"Unknown backend: {name}")


def _evaluate_all(
    config: dict[str, Any],
    samples: list[Sample],
    backend: VLMBackend,
    out_dir: Path,
    seed: int,
) -> list[EvalRecord]:
    intervention_specs = _resolve_intervention_specs(config)
    _log(config, f"[virel] Materializing {len(intervention_specs)} interventions per sample...")
    interventions = materialize_interventions(samples, intervention_specs, out_dir / "interventions", seed)
    sample_interventions = {
        sample.sample_id: _available_interventions_for_sample(sample, interventions, intervention_specs)
        for sample in samples
    }
    methods = list(config["pruning"]["methods"])
    budgets = list(config["pruning"]["budget_schedule"])
    dense_budget = config["pruning"].get("dense_budget", "full")
    dense_total = sum(len(items) for items in sample_interventions.values())
    pruned_total = len(methods) * len(budgets) * dense_total
    _log(config, f"[virel] Dense reference evaluations: {dense_total}")
    original_cache: dict[tuple[str, str, str], Any] = {}
    dense_records = _dense_references(config, samples, backend, sample_interventions, dense_budget, seed, original_cache)
    dense_vem_by_sample = {(r.sample_id, r.intervention): r.vem for r in dense_records}
    all_records = list(dense_records)
    _log(config, f"[virel] Pruned evaluations: {pruned_total}")
    completed = 0
    for method in methods:
        for budget in budgets:
            _log(config, f"[virel] Running method={method}, budget={budget}...")
            for sample in samples:
                for intervention_name, intervention in sample_interventions.get(sample.sample_id, []):
                    record = _evaluate_one(
                        config,
                        sample,
                        backend,
                        method,
                        budget,
                        intervention_name,
                        intervention.path,
                        dense_vem_by_sample.get((sample.sample_id, intervention_name)),
                        seed,
                        original_cache,
                    )
                    all_records.append(record)
                    completed += 1
                if _should_log_progress(max(len(sample_interventions.get(sample.sample_id, [])), 1), completed, pruned_total):
                    _log(config, f"[virel] Progress {completed}/{pruned_total} pruned evaluations")
    return all_records


def _dense_references(
    config: dict[str, Any],
    samples: list[Sample],
    backend: VLMBackend,
    sample_interventions: dict[str, list[tuple[str, Any]]],
    dense_budget: Budget,
    seed: int,
    original_cache: dict[tuple[str, str, str], Any],
) -> list[EvalRecord]:
    records = []
    total = sum(len(items) for items in sample_interventions.values())
    completed = 0
    for sample in samples:
        sample_items = sample_interventions.get(sample.sample_id, [])
        for intervention_name, intervention in sample_items:
            records.append(
                _evaluate_one(
                    config,
                    sample,
                    backend,
                    "dense",
                    dense_budget,
                    intervention_name,
                    intervention.path,
                    dense_vem=None,
                    seed=seed,
                    original_cache=original_cache,
                )
            )
            completed += 1
        if _should_log_progress(max(len(sample_items), 1), completed, total):
            _log(config, f"[virel] Progress {completed}/{total} dense evaluations")
    return [
        EvalRecord(
            **{
                **r.__dict__,
                "dense_vem": r.vem,
                "reliance_retention": 1.0 if abs(r.vem) > 1e-9 else None,
                "delta_vem": 0.0,
            }
        )
        for r in records
    ]


def _evaluate_one(
    config: dict[str, Any],
    sample: Sample,
    backend: VLMBackend,
    method: str,
    budget: Budget,
    intervention_name: str,
    intervention_path: Path | None,
    dense_vem: float | None,
    seed: int,
    original_cache: dict[tuple[str, str, str], Any] | None = None,
) -> EvalRecord:
    if intervention_path is None:
        raise ValueError(f"Intervention {intervention_name} for {sample.sample_id} has no path")
    original_cache = original_cache if original_cache is not None else {}
    cache_key = (sample.sample_id, method, str(budget))
    if cache_key not in original_cache:
        original_cache[cache_key] = backend.score_options(sample, sample.image_path, method, budget, seed)
    original = original_cache[cache_key]
    intervened = backend.score_options(sample, intervention_path, method, budget, seed)
    score_type = str(original.metadata.get("score_type", "score_vem")) if isinstance(original.metadata, dict) else "score_vem"
    if score_type == "answer_only":
        answer_changed = not exact_match(original.answer, intervened.answer)
        vem = 1.0 if answer_changed else 0.0
        support_metric = "answer_flip_support"
        answer_flip_support = answer_changed
    else:
        vem = visual_evidence_margin(original.logprob, intervened.logprob)
        support_metric = "score_vem"
        answer_flip_support = None
    rr = reliance_retention(vem, dense_vem)
    delta = None if dense_vem is None else dense_vem - vem
    latency_ms = original.latency_ms + intervened.latency_ms
    cost_spec = config.get("cost", {})
    if cost_spec.get("base_ms") is not None and config["backend"]["name"] == "deterministic":
        dense_count = int(cost_spec.get("dense_token_count") or backend.dense_token_count)
        token_count = dense_count if budget == "full" else int(budget)
        latency_ms = float(cost_spec.get("base_ms", 0.0)) + float(cost_spec.get("per_token_ms", 1.0)) * token_count
    token_count = int(original.token_count)
    api_cost = _sum_optional(original.api_cost_usd, intervened.api_cost_usd)
    cost = api_cost if api_cost is not None else cost_from_latency(latency_ms, float(cost_spec.get("dollar_per_1k_ms", 0.0)))
    proxy_energy = latency_ms * float(cost_spec.get("energy_joule_per_ms", 0.0))
    measured_energy = _sum_optional(original.measured_energy_joule, intervened.measured_energy_joule)
    energy = measured_energy if measured_energy is not None else proxy_energy
    return EvalRecord(
        sample_id=sample.sample_id,
        split=sample.split,
        dataset=sample.dataset,
        method=method,
        budget=budget,
        intervention=intervention_name,
        answer=original.answer,
        gold_answer=sample.answer,
        is_correct=exact_match(original.answer, sample.answer),
        logprob_original=original.logprob,
        logprob_intervened=intervened.logprob,
        confidence=original.confidence,
        vem=vem,
        dense_vem=dense_vem,
        reliance_retention=rr,
        delta_vem=delta,
        shortcut_persistence=exact_match(original.answer, intervened.answer),
        latency_ms=latency_ms,
        token_count=token_count,
        cost=cost,
        energy_joule=energy,
        support_metric=support_metric,
        answer_flip_support=answer_flip_support,
        measured_latency_ms=latency_ms,
        api_prompt_tokens=_sum_int_optional(original.api_prompt_tokens, intervened.api_prompt_tokens),
        api_completion_tokens=_sum_int_optional(original.api_completion_tokens, intervened.api_completion_tokens),
        api_total_tokens=_sum_int_optional(original.api_total_tokens, intervened.api_total_tokens),
        api_cost_usd=api_cost,
        measured_energy_joule=measured_energy,
        proxy_energy_joule=proxy_energy,
        metadata={
            "original_backend": original.metadata,
            "intervened_backend": intervened.metadata,
        },
    )


def _resolve_intervention_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for spec in config["interventions"]:
        resolved = dict(spec)
        if spec.get("pool"):
            pool_path = resolve_path(config, spec["pool"])
            resolved["pool"] = str(pool_path)
        specs.append(resolved)
    return specs


def _available_interventions_for_sample(
    sample: Sample,
    interventions: dict[tuple[str, str], Any],
    intervention_specs: list[dict[str, Any]],
) -> list[tuple[str, Any]]:
    items = []
    for spec in intervention_specs:
        name = str(spec["name"])
        intervention = interventions.get((sample.sample_id, name))
        if intervention is not None:
            items.append((name, intervention))
    return items


def _sum_optional(left: float | None, right: float | None) -> float | None:
    values = [v for v in [left, right] if v is not None]
    return float(sum(values)) if values else None


def _sum_int_optional(left: int | None, right: int | None) -> int | None:
    values = [v for v in [left, right] if v is not None]
    return int(sum(values)) if values else None


def _write_outputs(
    out_dir: Path,
    records: list[EvalRecord],
    decisions: list,
    summary_rows: list[dict],
    decision_summary: dict,
    baseline_comparison: dict,
    budget_curve: list[dict[str, Any]],
    dense_reliance_audit: list[dict[str, Any]],
    macro_rows: list[dict[str, Any]],
    ci_rows: list[dict[str, Any]],
    paired_ci_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    thresholds,
    config: dict[str, Any],
) -> dict[str, str]:
    artifacts = {
        "records_jsonl": str(out_dir / "records.jsonl"),
        "summary_csv": str(out_dir / "summary_by_method_budget.csv"),
        "decisions_jsonl": str(out_dir / "policy_decisions.jsonl"),
        "decision_summary_json": str(out_dir / "policy_summary.json"),
        "decision_by_dataset_csv": str(out_dir / "policy_by_dataset.csv"),
        "dense_reliance_audit_csv": str(out_dir / "dense_reliance_audit.csv"),
        "budget_curve_csv": str(out_dir / "budget_curve_dense_reliant.csv"),
        "policy_macro_csv": str(out_dir / "policy_macro_summary.csv"),
        "bootstrap_ci_csv": str(out_dir / "bootstrap_ci.csv"),
        "paired_bootstrap_ci_csv": str(out_dir / "paired_bootstrap_ci.csv"),
        "ablation_summary_csv": str(out_dir / "ablation_summary.csv"),
        "greenmm_summary_json": str(out_dir / "greenmm_summary.json"),
        "greenmm_table_csv": str(out_dir / "greenmm_table.csv"),
        "proposal_claims_json": str(out_dir / "proposal_claims.json"),
        "proposal_main_table_csv": str(out_dir / "proposal_main_table.csv"),
        "proposal_report_md": str(out_dir / "proposal_report.md"),
        "qualitative_json": str(out_dir / "qualitative_examples.json"),
        "frontier_svg": str(out_dir / "frontier_latency_rr.svg"),
        "policy_comparison_csv": str(out_dir / "policy_comparison.csv"),
    }
    write_records(artifacts["records_jsonl"], records)
    write_csv(artifacts["summary_csv"], summary_rows)
    write_records(artifacts["decisions_jsonl"], decisions)
    write_json(
        artifacts["decision_summary_json"],
        {
            "run_name": config["run_name"],
            "thresholds": {
                "tau": thresholds.tau,
                "rho": thresholds.rho,
                "dense_vem_min": thresholds.dense_vem_min,
            },
            "policy": decision_summary,
            "baselines": baseline_comparison,
        },
    )
    write_csv(
        artifacts["policy_comparison_csv"],
        [{"policy": name, **values} for name, values in baseline_comparison.items()],
    )
    write_csv(
        artifacts["decision_by_dataset_csv"],
        [{"dataset": dataset, **values} for dataset, values in _summarize_decisions_by_dataset(decisions).items()],
    )
    write_csv(artifacts["dense_reliance_audit_csv"], dense_reliance_audit)
    write_csv(artifacts["budget_curve_csv"], budget_curve)
    write_csv(artifacts["policy_macro_csv"], macro_rows)
    write_csv(artifacts["bootstrap_ci_csv"], ci_rows)
    write_csv(artifacts["paired_bootstrap_ci_csv"], paired_ci_rows)
    write_csv(artifacts["ablation_summary_csv"], ablation_rows)
    greenmm_summary, greenmm_table = _greenmm_outputs(baseline_comparison, config)
    write_json(artifacts["greenmm_summary_json"], greenmm_summary)
    write_csv(artifacts["greenmm_table_csv"], greenmm_table)
    proposal_claims, proposal_table, proposal_report = _proposal_outputs(
        baseline_comparison,
        dense_reliance_audit,
        thresholds,
        config,
    )
    write_json(artifacts["proposal_claims_json"], proposal_claims)
    write_csv(artifacts["proposal_main_table_csv"], proposal_table)
    Path(artifacts["proposal_report_md"]).write_text(proposal_report, encoding="utf-8")
    write_json(artifacts["qualitative_json"], qualitative_examples(records))
    test_rows = [r for r in summary_rows if r["intervention"] == config["interventions"][0]["name"]]
    write_frontier_svg(artifacts["frontier_svg"], test_rows, f"{config['run_name']} latency-reliance frontier")
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_method_budget_pandas.csv", index=False)
    return artifacts


def _policy_decision_sets(
    config: dict[str, Any],
    all_records: list[EvalRecord],
    test_records: list[EvalRecord],
    virel_decisions: list,
    thresholds,
) -> dict[str, list]:
    budget_schedule = list(config["pruning"]["budget_schedule"])
    if not budget_schedule:
        budget_schedule = [config["pruning"].get("dense_budget", "full")]
    aggressive_budget = budget_schedule[0]
    medium_budget = budget_schedule[len(budget_schedule) // 2]
    validation_records = [r for r in all_records if r.split == config["dataset"].get("validation_split", "validation")]
    comparison: dict[str, list] = {}
    if config["pruning"].get("methods"):
        comparison["virel_budget"] = virel_decisions
        comparison["virel_budget_online"] = apply_online_cascade_policy(test_records, thresholds)
    comparison["dense_only"] = apply_fixed_budget_policy(
        test_records, thresholds, "dense", config["pruning"].get("dense_budget", "full"), "dense_only"
    )
    for method in config["pruning"]["methods"]:
        comparison[f"{method}_virel_budget"] = apply_policy(test_records, thresholds, allowed_methods=[method], label=f"{method}_virel_budget")
        comparison[f"{method}_virel_budget_online"] = apply_online_cascade_policy(
            test_records, thresholds, allowed_methods=[method], label=f"{method}_virel_budget_online"
        )
        comparison[f"{method}_fixed_aggressive"] = apply_fixed_budget_policy(
            test_records, thresholds, method, aggressive_budget, f"{method}_fixed_aggressive"
        )
        comparison[f"{method}_fixed_medium"] = apply_fixed_budget_policy(test_records, thresholds, method, medium_budget, f"{method}_fixed_medium")
        comparison[f"{method}_cost_only"] = comparison[f"{method}_fixed_aggressive"]
        acc_budget = choose_accuracy_only_budget(validation_records, method, budget_schedule)
        comparison[f"{method}_accuracy_only"] = apply_fixed_budget_policy(test_records, thresholds, method, acc_budget, f"{method}_accuracy_only")
    return comparison


def _policy_macro_rows(decision_sets: dict[str, list]) -> list[dict[str, Any]]:
    rows = []
    for policy, decisions in sorted(decision_sets.items()):
        if not decisions:
            continue
        by_dataset: dict[str, list] = {}
        for decision in decisions:
            by_dataset.setdefault(str(decision.dataset), []).append(decision)
        dataset_summaries = {dataset: summarize_decisions(items) for dataset, items in by_dataset.items() if items}
        for metric in [
            "accuracy",
            "visually_supported_rate",
            "mean_token_count",
            "mean_token_reduction",
            "dense_avoidance_rate",
            "cost_per_visually_supported_answer",
            "measured_energy_per_visually_supported_answer_joule",
            "mean_measured_energy_joule",
            "mean_latency_ms",
        ]:
            values = [summary.get(metric) for summary in dataset_summaries.values() if summary.get(metric) is not None]
            if values:
                rows.append(
                    {
                        "policy": policy,
                        "metric": metric,
                        "macro_mean": sum(float(v) for v in values) / len(values),
                        "n_datasets": len(values),
                        "datasets": sorted(dataset_summaries),
                    }
                )
    return rows


def _bootstrap_ci_rows(decision_sets: dict[str, list], iterations: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = [
        "accuracy",
        "visually_supported_rate",
        "mean_token_count",
        "mean_token_reduction",
        "dense_avoidance_rate",
        "cost_per_visually_supported_answer",
        "measured_energy_per_visually_supported_answer_joule",
        "mean_measured_energy_joule",
        "mean_latency_ms",
    ]
    rng = random.Random(seed)
    for policy, decisions in sorted(decision_sets.items()):
        items = list(decisions)
        if not items:
            continue
        point = summarize_decisions(items)
        boot_values: dict[str, list[float]] = {metric: [] for metric in metrics}
        for _ in range(max(iterations, 1)):
            sample = [items[rng.randrange(len(items))] for _ in items]
            summary = summarize_decisions(sample)
            for metric in metrics:
                value = summary.get(metric)
                if value is not None:
                    boot_values[metric].append(float(value))
        for metric in metrics:
            values = sorted(boot_values[metric])
            if not values or point.get(metric) is None:
                continue
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "point": point.get(metric),
                    "ci_low": _quantile(values, 0.025),
                    "ci_high": _quantile(values, 0.975),
                    "bootstrap_iterations": iterations,
                    "n": len(items),
                }
            )
    return rows


def _paired_bootstrap_ci_rows(decision_sets: dict[str, list], iterations: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    references = [name for name in ["dense_only", "random_fixed_aggressive", "virel_budget"] if name in decision_sets]
    metrics = [
        "accuracy",
        "visually_supported_rate",
        "cost_per_visually_supported_answer",
        "measured_energy_per_visually_supported_answer_joule",
        "mean_measured_energy_joule",
        "mean_token_count",
        "mean_latency_ms",
    ]
    rng = random.Random(seed + 997)
    maps = {
        policy: {decision.sample_id: decision for decision in decisions}
        for policy, decisions in decision_sets.items()
        if decisions
    }
    for policy, policy_map in sorted(maps.items()):
        for reference in references:
            if policy == reference or reference not in maps:
                continue
            common_ids = sorted(set(policy_map) & set(maps[reference]))
            if not common_ids:
                continue
            point = _paired_metric_deltas(policy_map, maps[reference], common_ids, metrics)
            boot_values: dict[str, list[float]] = {metric: [] for metric in metrics}
            for _ in range(max(iterations, 1)):
                sample_ids = [common_ids[rng.randrange(len(common_ids))] for _ in common_ids]
                deltas = _paired_metric_deltas(policy_map, maps[reference], sample_ids, metrics)
                for metric, value in deltas.items():
                    if value is not None:
                        boot_values[metric].append(float(value))
            for metric in metrics:
                values = sorted(boot_values[metric])
                if not values or point.get(metric) is None:
                    continue
                rows.append(
                    {
                        "policy": policy,
                        "reference": reference,
                        "metric": metric,
                        "delta_point": point[metric],
                        "ci_low": _quantile(values, 0.025),
                        "ci_high": _quantile(values, 0.975),
                        "bootstrap_iterations": iterations,
                        "n_common": len(common_ids),
                    }
                )
    return rows


def _paired_metric_deltas(left: dict[str, Any], right: dict[str, Any], sample_ids: list[str], metrics: list[str]) -> dict[str, float | None]:
    left_summary = summarize_decisions([left[sample_id] for sample_id in sample_ids])
    right_summary = summarize_decisions([right[sample_id] for sample_id in sample_ids])
    return {
        metric: (
            float(left_summary[metric]) - float(right_summary[metric])
            if left_summary.get(metric) is not None and right_summary.get(metric) is not None
            else None
        )
        for metric in metrics
    }


def _ablation_rows(config: dict[str, Any], test_records: list[EvalRecord], thresholds) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tau_values = [0.10, 0.25, 0.50, 0.75, 1.00]
    rho_values = [0.50, 0.70, 0.90] if thresholds.rho is not None else [None]
    for tau in tau_values:
        for rho in rho_values:
            candidate = type(thresholds)(tau=tau, rho=rho, dense_vem_min=thresholds.dense_vem_min)
            summary = summarize_decisions(apply_policy(test_records, candidate))
            if summary:
                rows.append({"ablation": "threshold", "tau": tau, "rho": rho, "setting": f"tau={tau},rho={rho}", **summary})
    available_interventions = sorted({record.intervention for record in test_records})
    intervention_sets: dict[str, set[str]] = {name: {name} for name in available_interventions}
    non_counterfactual = {name for name in available_interventions if name != "counterfactual"}
    if non_counterfactual:
        intervention_sets["all_non_counterfactual"] = non_counterfactual
    if available_interventions:
        intervention_sets["all_available"] = set(available_interventions)
    for label, wanted in sorted(intervention_sets.items()):
        filtered = [record for record in test_records if record.intervention in wanted]
        summary = summarize_decisions(apply_policy(filtered, thresholds))
        if summary:
            rows.append({"ablation": "intervention", "setting": label, "interventions": sorted(wanted), **summary})
    no_dense_filter = with_dense_reliance(thresholds, False)
    summary = summarize_decisions(apply_policy(test_records, no_dense_filter))
    if summary:
        rows.append({"ablation": "dense_filter", "setting": "without_dense_reliant_filter", **summary})
    return rows


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _budget_curve(config: dict[str, Any], test_records: list[EvalRecord], thresholds) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dense_budget = config["pruning"].get("dense_budget", "full")
    dense_summary = summarize_decisions(
        apply_fixed_budget_policy(test_records, thresholds, "dense", dense_budget, "dense_only")
    )
    rows.append({"policy": "dense_only", "method": "dense", "budget": dense_budget, **dense_summary})
    for method in config["pruning"]["methods"]:
        for budget in config["pruning"]["budget_schedule"]:
            label = f"{method}_fixed_{budget}"
            summary = summarize_decisions(apply_fixed_budget_policy(test_records, thresholds, method, budget, label))
            rows.append({"policy": label, "method": method, "budget": budget, **summary})
    return rows


def _dense_reliance_audit(test_records: list[EvalRecord], thresholds) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[EvalRecord]] = {}
    for record in test_records:
        groups.setdefault((record.sample_id, record.split), []).append(record)
    eligible_by_sample = eligible_interventions_by_sample(test_records, thresholds)
    dense_required_vem = max(float(thresholds.dense_vem_min), float(thresholds.tau))

    accum: dict[str, dict[str, Any]] = {}

    def bucket(dataset: str) -> dict[str, Any]:
        return accum.setdefault(
            dataset,
            {
                "dataset": dataset,
                "n_test_queries": 0,
                "n_dense_reliant_queries": 0,
                "total_dense_interventions": 0,
                "total_dense_correct_interventions": 0,
                "total_dense_supported_interventions": 0,
                "total_eligible_interventions": 0,
            },
        )

    for key, items in groups.items():
        dense_records = [r for r in items if r.method == "dense"]
        if not dense_records:
            continue
        dataset = str(dense_records[0].dataset)
        eligible = eligible_by_sample.get(key, ())
        for row in (bucket("_all"), bucket(dataset)):
            row["n_test_queries"] += 1
            row["n_dense_reliant_queries"] += int(bool(eligible))
            row["total_dense_interventions"] += len(dense_records)
            row["total_dense_correct_interventions"] += sum(int(r.is_correct) for r in dense_records)
            row["total_dense_supported_interventions"] += sum(
                int(r.is_correct and r.vem >= dense_required_vem) for r in dense_records
            )
            row["total_eligible_interventions"] += len(eligible)

    rows = []
    for dataset, row in sorted(accum.items(), key=lambda item: (item[0] != "_all", item[0])):
        n_queries = max(int(row["n_test_queries"]), 1)
        n_interventions = max(int(row["total_dense_interventions"]), 1)
        rows.append(
            {
                **row,
                "dense_vem_threshold": dense_required_vem,
                "dense_reliant_query_rate": row["n_dense_reliant_queries"] / n_queries,
                "mean_eligible_interventions_per_query": row["total_eligible_interventions"] / n_queries,
                "eligible_intervention_rate": row["total_eligible_interventions"] / n_interventions,
                "dense_correct_intervention_rate": row["total_dense_correct_interventions"] / n_interventions,
                "dense_supported_intervention_rate": row["total_dense_supported_interventions"] / n_interventions,
            }
        )
    return rows


def _summarize_decisions_by_dataset(decisions: list) -> dict[str, dict]:
    by_dataset: dict[str, list] = {}
    for decision in decisions:
        by_dataset.setdefault(str(decision.dataset), []).append(decision)
    return {dataset: summarize_decisions(items) for dataset, items in sorted(by_dataset.items())}


def _paper_policy_labels(config: dict[str, Any], baseline_comparison: dict[str, dict]) -> list[tuple[str, str]]:
    methods = [str(method) for method in config.get("pruning", {}).get("methods", [])]
    labels: list[tuple[str, str]] = [
        ("virel_budget", "ViRel-Budget (ours)"),
        ("virel_budget_online", "Online ViRel cascade (ours)"),
        ("dense_only", "Dense full-token reference"),
    ]
    for method in methods:
        display = _display_method(method)
        labels.extend(
            [
                (f"{method}_virel_budget", f"{display}-only ViRel-Budget"),
                (f"{method}_virel_budget_online", f"{display}-only online ViRel"),
                (f"{method}_fixed_aggressive", f"{display} fixed aggressive"),
                (f"{method}_fixed_medium", f"{display} fixed medium"),
                (f"{method}_accuracy_only", f"{display} accuracy-only budget"),
                (f"{method}_cost_only", f"{display} cost-only aggressive"),
            ]
        )
    for key in sorted(baseline_comparison):
        if key not in {label[0] for label in labels}:
            labels.append((key, _policy_display_label(key, config)))
    return [(key, label) for key, label in labels if key in baseline_comparison]


def _primary_aggressive_reference(config: dict[str, Any], baseline_comparison: dict[str, dict]) -> tuple[str | None, dict[str, Any]]:
    methods = [str(method) for method in config.get("pruning", {}).get("methods", [])]
    for method in methods:
        key = f"{method}_fixed_aggressive"
        if key in baseline_comparison:
            return key, baseline_comparison[key]
    for key in sorted(baseline_comparison):
        if key.endswith("_fixed_aggressive"):
            return key, baseline_comparison[key]
    return None, {}


def _policy_display_label(policy: str | None, config: dict[str, Any]) -> str:
    if not policy:
        return ""
    if policy == "virel_budget":
        return "ViRel-Budget"
    if policy == "virel_budget_online":
        return "Online ViRel cascade"
    if policy == "dense_only":
        return "dense full-token reference"
    for method in [str(method) for method in config.get("pruning", {}).get("methods", [])]:
        prefix = f"{method}_"
        if policy.startswith(prefix):
            suffix = policy[len(prefix) :]
            display = _display_method(method)
            if suffix == "fixed_aggressive":
                budget = config.get("pruning", {}).get("budget_schedule", ["low"])[0]
                return f"{display} fixed-{budget}"
            if suffix == "fixed_medium":
                schedule = list(config.get("pruning", {}).get("budget_schedule", ["medium"]))
                budget = schedule[len(schedule) // 2]
                return f"{display} fixed-{budget}"
            return f"{display} {suffix.replace('_', '-')}"
    return policy.replace("_", " ")


def _display_method(method: str) -> str:
    names = {"fastv": "FastV", "scope": "SCOPE", "random": "Random", "center": "Center", "saliency": "Saliency"}
    return names.get(method, method)


def _greenmm_outputs(baseline_comparison: dict[str, dict], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    columns = [
        "n",
        "accuracy",
        "visually_supported_rate",
        "mean_token_count",
        "mean_token_reduction",
        "mean_retained_token_ratio",
        "mean_speedup_vs_dense",
        "cost_per_visually_supported_answer",
        "measured_energy_per_visually_supported_answer_joule",
        "mean_measured_energy_joule",
        "total_measured_energy_joule",
        "dense_avoidance_rate",
        "mean_escalations",
        "selected_budget_distribution",
    ]
    table = []
    for name, _label in _paper_policy_labels(config, baseline_comparison):
        values = baseline_comparison.get(name)
        if not values:
            continue
        table.append({"policy": name, **{key: values.get(key) for key in columns}})
    virel = baseline_comparison.get("virel_budget", {})
    dense = baseline_comparison.get("dense_only", {})
    aggressive_key, aggressive = _primary_aggressive_reference(config, baseline_comparison)
    summary = {
        "headline": "Dense-reliant GreenMM summary: cost is evaluated per visually supported answer.",
        "virel_budget": virel,
        "dense_only": dense,
        "fixed_aggressive_reference_key": aggressive_key,
        "fixed_aggressive_reference": aggressive,
        "claim_deltas": {
            "token_reduction_vs_dense": virel.get("mean_token_reduction"),
            "dense_avoidance_rate": virel.get("dense_avoidance_rate"),
            "visual_support_gain_vs_fixed_aggressive": _delta(virel, aggressive, "visually_supported_rate"),
            "accuracy_delta_vs_fixed_aggressive": _delta(virel, aggressive, "accuracy"),
            "cost_per_supported_delta_vs_fixed_aggressive": _delta(virel, aggressive, "cost_per_visually_supported_answer"),
            "cost_per_supported_delta_vs_dense": _delta(virel, dense, "cost_per_visually_supported_answer"),
            "measured_energy_per_supported_delta_vs_dense_joule": _delta(
                virel, dense, "measured_energy_per_visually_supported_answer_joule"
            ),
        },
        "notes": [
            "The dense-reliant subset keeps only interventions where dense full-token inference is correct and visually supported.",
            "FastV/SCOPE are external visual-token pruning methods; ViRel-Budget is the training-free reliability-aware budget controller.",
            "Measured energy is GPU power attribution from nvidia-smi samples for the evaluated hardware, not full-system carbon accounting.",
        ],
    }
    return summary, table


def _proposal_outputs(
    baseline_comparison: dict[str, dict],
    dense_reliance_audit: list[dict[str, Any]],
    thresholds,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    policy_labels = _paper_policy_labels(config, baseline_comparison)
    table = []
    for key, label in policy_labels:
        values = baseline_comparison.get(key)
        if not values:
            continue
        table.append(
            {
                "policy": key,
                "paper_label": label,
                "n": values.get("n"),
                "accuracy_pct": _pct(values.get("accuracy")),
                "visual_support_pct": _pct(values.get("visually_supported_rate")),
                "mean_vem": values.get("mean_vem"),
                "mean_rr": values.get("mean_rr"),
                "mean_tokens": values.get("mean_token_count"),
                "token_reduction_pct": _pct(values.get("mean_token_reduction")),
                "dense_avoidance_pct": _pct(values.get("dense_avoidance_rate")),
                "fallback_full_pct": _pct(values.get("fallback_full_rate")),
                "mean_escalations": values.get("mean_escalations"),
                "cost_per_supported_answer": values.get("cost_per_visually_supported_answer"),
                "measured_energy_per_supported_answer_joule": values.get(
                    "measured_energy_per_visually_supported_answer_joule"
                ),
                "mean_measured_energy_joule": values.get("mean_measured_energy_joule"),
                "selected_budgets": values.get("selected_budget_distribution"),
                "selected_methods": values.get("selected_method_distribution"),
            }
        )

    virel = baseline_comparison.get("virel_budget", {})
    dense = baseline_comparison.get("dense_only", {})
    audit_all = next((row for row in dense_reliance_audit if row.get("dataset") == "_all"), {})
    if not virel and dense:
        claim_deltas = {
            "token_reduction_vs_dense_pct": None,
            "dense_avoidance_pct": 0.0,
            "visual_support_gain_vs_fixed_aggressive_pp": None,
            "accuracy_gain_vs_fixed_aggressive_pp": None,
            "cost_per_supported_delta_vs_fixed_aggressive": None,
            "visual_support_delta_vs_dense_pp": 0.0,
            "accuracy_delta_vs_dense_pp": 0.0,
        }
        claims = {
            "run_name": config["run_name"],
            "dataset_path": config["dataset"]["path"],
            "test_split": config["dataset"].get("test_split", "test"),
            "thresholds": {
                "tau": thresholds.tau,
                "rho": thresholds.rho,
                "dense_vem_min": thresholds.dense_vem_min,
            },
            "dense_reliance_audit": audit_all,
            "main_policy": {},
            "dense_reference": dense,
            "fixed_aggressive_reference": {},
            "fixed_aggressive_reference_key": None,
            "claim_deltas": claim_deltas,
            "claim_sentences": _dense_only_claim_sentences(dense, audit_all),
            "honesty_boundary": (
                "This run is a dense modern-VLM validation track. It measures answer-level intervention sensitivity, "
                "latency, token usage, and energy/cost accounting, but it does not evaluate visual-token pruning or "
                "budget selection."
            ),
        }
        report = _proposal_report(config, claims, table, dense_reliance_audit)
        return claims, table, report
    budget_schedule = list(config.get("pruning", {}).get("budget_schedule", []))
    aggressive_budget = budget_schedule[0] if budget_schedule else config.get("pruning", {}).get("dense_budget", "full")
    aggressive_key, aggressive = _primary_aggressive_reference(config, baseline_comparison)
    aggressive_label = _policy_display_label(aggressive_key, config) if aggressive_key else f"fixed-{aggressive_budget}"
    claim_deltas = {
        "token_reduction_vs_dense_pct": _pct(virel.get("mean_token_reduction")),
        "dense_avoidance_pct": _pct(virel.get("dense_avoidance_rate")),
        "visual_support_gain_vs_fixed_aggressive_pp": _pp(
            _delta(virel, aggressive, "visually_supported_rate")
        ),
        "accuracy_gain_vs_fixed_aggressive_pp": _pp(_delta(virel, aggressive, "accuracy")),
        "cost_per_supported_delta_vs_fixed_aggressive": _delta(
            virel, aggressive, "cost_per_visually_supported_answer"
        ),
        "measured_energy_per_supported_delta_vs_dense_joule": _delta(
            virel, dense, "measured_energy_per_visually_supported_answer_joule"
        ),
        "visual_support_delta_vs_dense_pp": _pp(_delta(virel, dense, "visually_supported_rate")),
        "accuracy_delta_vs_dense_pp": _pp(_delta(virel, dense, "accuracy")),
    }
    claims = {
        "run_name": config["run_name"],
        "dataset_path": config["dataset"]["path"],
        "test_split": config["dataset"].get("test_split", "test"),
        "thresholds": {
            "tau": thresholds.tau,
            "rho": thresholds.rho,
            "dense_vem_min": thresholds.dense_vem_min,
        },
        "dense_reliance_audit": audit_all,
        "main_policy": virel,
        "dense_reference": dense,
        "fixed_aggressive_reference": aggressive,
        "fixed_aggressive_reference_key": aggressive_key,
        "claim_deltas": claim_deltas,
        "claim_sentences": _claim_sentences(virel, dense, aggressive, audit_all, aggressive_label),
        "honesty_boundary": (
            "This paper-track run evaluates visual-reliance-aware budget selection over external LLaVA-compatible "
            "visual-token pruning methods. Energy claims should be stated as measured GPU-energy attribution on the "
            "reported hardware, not as universal carbon reduction."
        ),
    }
    report = _proposal_report(config, claims, table, dense_reliance_audit)
    return claims, table, report


def _dense_only_claim_sentences(dense: dict[str, Any], audit_all: dict[str, Any]) -> list[str]:
    return [
        (
            "Dense-reliant evaluation keeps "
            f"{int(audit_all.get('n_dense_reliant_queries', 0))}/"
            f"{int(audit_all.get('n_test_queries', 0))} test queries where dense inference is itself visually supported."
        ),
        (
            "This dense validation track reaches "
            f"{_fmt_pct(dense.get('visually_supported_rate'))} answer-level intervention support on that subset."
        ),
        (
            "Mean latency is "
            f"{_fmt_float(dense.get('mean_latency_ms'))} ms and measured GPU energy per supported answer is "
            f"{_fmt_float(dense.get('measured_energy_per_visually_supported_answer_joule'))} J."
        ),
        "No visual-token pruning or dense-avoidance claim should be made from this dense-only track.",
    ]


def _claim_sentences(
    virel: dict[str, Any],
    dense: dict[str, Any],
    aggressive: dict[str, Any],
    audit_all: dict[str, Any],
    aggressive_label: str,
) -> list[str]:
    return [
        (
            "Dense-reliant evaluation keeps "
            f"{int(audit_all.get('n_dense_reliant_queries', 0))}/"
            f"{int(audit_all.get('n_test_queries', 0))} test queries where dense inference is itself visually supported."
        ),
        (
            "ViRel-Budget selects compressed budgets for "
            f"{_fmt_pct(virel.get('dense_avoidance_rate'))} of dense-reliant queries while retaining "
            f"{_fmt_pct(virel.get('visually_supported_rate'))} visual support."
        ),
        (
            "Compared with dense full-token inference, ViRel-Budget reduces visual tokens by "
            f"{_fmt_pct(virel.get('mean_token_reduction'))} on average."
        ),
        (
            f"Compared with {aggressive_label}, ViRel-Budget changes visual support by "
            f"{_fmt_pp(_delta(virel, aggressive, 'visually_supported_rate'))} and accuracy by "
            f"{_fmt_pp(_delta(virel, aggressive, 'accuracy'))}."
        ),
        (
            "Compared with dense full-token inference, accuracy changes by "
            f"{_fmt_pp(_delta(virel, dense, 'accuracy'))}; this is the safety-efficiency tradeoff to report."
        ),
    ]


def _proposal_report(
    config: dict[str, Any],
    claims: dict[str, Any],
    table: list[dict[str, Any]],
    dense_reliance_audit: list[dict[str, Any]],
) -> str:
    lines = [
        "# ViRel-Budget Proposal-Aligned Report",
        "",
        f"Run: `{config['run_name']}`",
        f"Dataset: `{config['dataset']['path']}`",
        "",
        "## Claim-Ready Summary",
        "",
    ]
    lines.extend(f"- {sentence}" for sentence in claims["claim_sentences"])
    lines.extend(
        [
            "",
            "## Main Table",
            "",
            "| Policy | n | Acc. | Visual support | Mean tokens | Token reduction | Dense avoidance | Cost/support |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in table:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("paper_label", row.get("policy"))),
                    str(row.get("n", "")),
                    _fmt_pct_from_percent(row.get("accuracy_pct")),
                    _fmt_pct_from_percent(row.get("visual_support_pct")),
                    _fmt_float(row.get("mean_tokens")),
                    _fmt_pct_from_percent(row.get("token_reduction_pct")),
                    _fmt_pct_from_percent(row.get("dense_avoidance_pct")),
                    _fmt_sci(row.get("cost_per_supported_answer")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Dense-Reliant Audit",
            "",
            "| Dataset | test queries | dense-reliant queries | dense-reliant rate | eligible interventions/query |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dense_reliance_audit:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset")),
                    str(row.get("n_test_queries", "")),
                    str(row.get("n_dense_reliant_queries", "")),
                    _fmt_pct(row.get("dense_reliant_query_rate")),
                    _fmt_float(row.get("mean_eligible_interventions_per_query")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            claims["honesty_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def _delta(left: dict, right: dict, key: str) -> float | None:
    if left.get(key) is None or right.get(key) is None:
        return None
    return float(left[key]) - float(right[key])


def _pct(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def _pp(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def _fmt_pp(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:+.1f} pp"


def _fmt_pct_from_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_sci(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3e}"


def _log(config: dict[str, Any], message: str) -> None:
    if bool(config.get("progress", True)):
        print(message, file=sys.stderr, flush=True)


def _should_log_progress(step_size: int, completed: int, total: int) -> bool:
    if completed >= total:
        return True
    return completed % max(step_size * 5, 1) == 0
