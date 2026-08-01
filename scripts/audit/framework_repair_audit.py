from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from virel_budget.schema import canonicalize_answer  # noqa: E402


BUDGETS = (64, 128, 256, 432)
RISK_LIMITS = (0.01, 0.05, 0.10)
INTERVENTION_ORDER = ("gray", "blur", "irrelevant", "counterfactual")


@dataclass
class Action:
    sample_id: str
    split: str
    dataset: str
    method: str
    budget: int | str
    answer: str
    gold_answer: str
    is_correct: bool
    dense_answer: str
    dense_correct: bool
    eligible_interventions: tuple[str, ...]
    intervention_pass: bool
    intervention_answer_fidelity: bool
    answer_fidelity: bool
    legacy_flip_safe: bool
    reference_safe: bool
    combined_gold_safe: bool | None
    original_latency_ms: float | None
    original_energy_joule: float | None
    token_count: int
    pattern: str
    per_intervention: dict[str, dict[str, Any]]


def main() -> int:
    parser = argparse.ArgumentParser(description="No-GPU ViRel-Budget framework repair audit.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="results/framework_repair_no_gpu")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    samples = _read_jsonl(root / "data/paper_1200/samples.jsonl")
    sample_by_id = {row["sample_id"]: row for row in samples}
    methods = {}
    for method in ("fastv", "scope"):
        run_dir = root / f"results/paper1200_llava15_{method}"
        records = _read_jsonl(run_dir / "records_with_measured_energy_optionfix.jsonl")
        power = _read_power_samples(run_dir / "hardware_monitor/nvidia_smi_samples.csv")
        methods[method] = _build_actions(records, power, sample_by_id)

    split_audit = _split_leakage_audit(root, samples)
    dense_audit = _dense_consistency_audit(methods)
    labels, monotonicity = _label_and_monotonicity_rows(methods)
    populations = _population_rows(methods, samples)
    controller_rows, controller_details = _controller_replay(methods, sample_by_id)
    crossfit_rows, crossfit_details = _group_crossfit_replay(root, methods, sample_by_id)
    controller_rows.extend(crossfit_rows)
    controller_details["group_crossfit"] = crossfit_details
    qualitative = _qualitative_candidates(methods, sample_by_id, controller_details)
    cost_audit = _cost_boundary_audit(methods)

    _write_json(out / "split_leakage_audit.json", split_audit)
    _write_json(out / "dense_consistency_audit.json", dense_audit)
    _write_csv(out / "safe_budget_labels.csv", labels)
    _write_json(out / "budget_monotonicity_audit.json", monotonicity)
    _write_csv(out / "population_metrics.csv", populations)
    _write_csv(out / "controller_replay.csv", controller_rows)
    _write_json(out / "controller_replay_details.json", controller_details)
    _write_json(out / "cost_boundary_audit.json", cost_audit)
    _write_json(out / "qualitative_audit_candidates.json", qualitative)
    _write_framework_spec(out / "FRAMEWORK_SPEC.md")
    _write_report(
        out / "FRAMEWORK_REPAIR_REPORT.md",
        split_audit,
        dense_audit,
        monotonicity,
        populations,
        controller_rows,
        cost_audit,
        qualitative,
    )
    print(json.dumps({"output_dir": str(out), "artifacts": sorted(p.name for p in out.iterdir())}, indent=2))
    return 0


def _build_actions(
    records: list[dict[str, Any]],
    power: list[dict[str, float]],
    sample_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Action]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[str(record["sample_id"])].append(record)
    result: dict[str, dict[str, Action]] = {}
    for sample_id, items in by_sample.items():
        dense_records = [row for row in items if row["method"] == "dense"]
        if not dense_records:
            continue
        dense_answer = str(dense_records[0]["answer"])
        dense_correct = bool(dense_records[0]["is_correct"])
        sample = sample_by_id[sample_id]
        dense_by_intervention = {str(row["intervention"]): row for row in dense_records}
        eligible = tuple(
            name
            for name in INTERVENTION_ORDER
            if any(row["intervention"] == name and float(row["vem"]) >= 1.0 for row in dense_records)
        )
        actions: dict[str, Action] = {}
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            grouped[(str(row["method"]), str(row["budget"]))].append(row)
        for (method, budget_text), group in grouped.items():
            budget: int | str = "full" if budget_text == "full" else int(budget_text)
            first = group[0]
            by_intervention = {str(row["intervention"]): row for row in group}
            covers = all(name in by_intervention for name in eligible)
            intervention_pass = covers and all(float(by_intervention[name]["vem"]) >= 1.0 for name in eligible)
            answer_fidelity = _norm(first["answer"]) == _norm(dense_answer)
            intervention_answer_fidelity = covers and all(
                _norm(_intervened_answer(by_intervention[name], sample))
                == _norm(_intervened_answer(dense_by_intervention[name], sample))
                for name in eligible
            )
            legacy_flip_safe = answer_fidelity and intervention_pass
            reference_safe = answer_fidelity and intervention_answer_fidelity
            combined_gold_safe = (
                bool(first["is_correct"]) and intervention_answer_fidelity
                if dense_correct and eligible
                else None
            )
            original_meta = (first.get("metadata") or {}).get("original_backend") or {}
            start = _float_or_none(original_meta.get("start_epoch"))
            end = _float_or_none(original_meta.get("end_epoch"))
            original_latency = (end - start) * 1000.0 if start is not None and end is not None and end > start else None
            original_energy = _call_energy(original_meta, power)
            key = "dense" if method == "dense" else str(budget)
            actions[key] = Action(
                sample_id=sample_id,
                split=str(first["split"]),
                dataset=str(first["dataset"]),
                method=method,
                budget=budget,
                answer=str(first["answer"]),
                gold_answer=str(first["gold_answer"]),
                is_correct=bool(first["is_correct"]),
                dense_answer=dense_answer,
                dense_correct=dense_correct,
                eligible_interventions=eligible,
                intervention_pass=intervention_pass,
                intervention_answer_fidelity=intervention_answer_fidelity,
                answer_fidelity=answer_fidelity,
                legacy_flip_safe=legacy_flip_safe,
                reference_safe=reference_safe,
                combined_gold_safe=combined_gold_safe,
                original_latency_ms=original_latency,
                original_energy_joule=original_energy,
                token_count=int(first["token_count"]),
                pattern="",
                per_intervention={
                    name: {
                        "vem": float(row["vem"]),
                        "answer": str(row["answer"]),
                        "is_correct": bool(row["is_correct"]),
                        "shortcut_persistence": bool(row["shortcut_persistence"]),
                        "intervened_raw_answer": str(
                            ((row.get("metadata") or {}).get("intervened_backend") or {}).get("raw_answer", "")
                        ),
                    }
                    for name, row in by_intervention.items()
                },
            )
        pattern = "".join("1" if actions.get(str(b)) and actions[str(b)].reference_safe else "0" for b in BUDGETS)
        for action in actions.values():
            action.pattern = pattern
        result[sample_id] = actions
    return result


def _split_leakage_audit(root: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[str, set[str]] = defaultdict(set)
    by_hash: dict[str, set[str]] = defaultdict(set)
    hash_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_question: dict[str, set[str]] = defaultdict(set)
    question_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_source_group: dict[str, set[str]] = defaultdict(set)
    missing = []
    for row in samples:
        split = str(row["split"])
        path = root / "data/paper_1200" / str(row["image_path"])
        by_path[str(path.resolve())].add(split)
        if path.exists():
            content_hash = _sha256(path)
            by_hash[content_hash].add(split)
            hash_members[content_hash].append(
                {
                    "sample_id": str(row["sample_id"]),
                    "dataset": str(row["dataset"]),
                    "split": split,
                    "image_path": str(row["image_path"]),
                }
            )
        else:
            missing.append(str(path))
        by_question[_norm(row["question"])].add(split)
        question_members[_norm(row["question"])].append(
            {"sample_id": str(row["sample_id"]), "dataset": str(row["dataset"]), "split": split}
        )
        metadata = row.get("metadata") or {}
        source_group = (
            metadata.get("source_meta_info", {}).get("image_path")
            if isinstance(metadata.get("source_meta_info"), dict)
            else None
        )
        source_group = source_group or metadata.get("coco_image_id") or metadata.get("image_id")
        if source_group is not None:
            by_source_group[f"{row['dataset']}::{source_group}"].add(split)
    cross_path = sorted(k for k, v in by_path.items() if len(v) > 1)
    cross_hash = sorted(k for k, v in by_hash.items() if len(v) > 1)
    cross_question = sorted(k for k, v in by_question.items() if len(v) > 1)
    cross_source = sorted(k for k, v in by_source_group.items() if len(v) > 1)
    return {
        "n_samples": len(samples),
        "split_counts": dict(Counter(str(row["split"]) for row in samples)),
        "missing_original_images": missing,
        "cross_split_exact_path_count": len(cross_path),
        "cross_split_content_hash_count": len(cross_hash),
        "cross_split_exact_question_count": len(cross_question),
        "cross_split_source_group_count": len(cross_source),
        "cross_split_exact_paths": cross_path[:100],
        "cross_split_content_hashes": cross_hash[:100],
        "cross_split_content_hash_groups": {
            key: hash_members[key] for key in cross_hash[:100]
        },
        "cross_split_exact_questions": cross_question[:100],
        "cross_split_exact_question_groups": {
            key: question_members[key] for key in cross_question[:100]
        },
        "cross_split_source_groups": cross_source[:100],
        "controller_consequence": (
            "The current validation/test split is not group-clean and must not be used as the final prospective "
            "controller evaluation split. Treat current controller replay as developmental feasibility only. "
            "Use group-aware cross-validation for the 1,200-case development pool and freeze the controller before "
            "evaluation on newly added 2,100-pool cases."
        ),
    }


def _dense_consistency_audit(methods: dict[str, dict[str, dict[str, Action]]]) -> dict[str, Any]:
    shared = sorted(set(methods["fastv"]) & set(methods["scope"]))
    rows = []
    for sample_id in shared:
        left = methods["fastv"][sample_id]["dense"]
        right = methods["scope"][sample_id]["dense"]
        rows.append(
            {
                "sample_id": sample_id,
                "split": left.split,
                "dataset": left.dataset,
                "answer_agree": _norm(left.answer) == _norm(right.answer),
                "correctness_agree": left.is_correct == right.is_correct,
                "eligible_interventions_agree": left.eligible_interventions == right.eligible_interventions,
                "fastv_dense_answer": left.answer,
                "scope_dense_answer": right.answer,
                "fastv_eligible": list(left.eligible_interventions),
                "scope_eligible": list(right.eligible_interventions),
            }
        )
    disagreements = [row for row in rows if not all(row[k] for k in ("answer_agree", "correctness_agree", "eligible_interventions_agree"))]
    common_reliant = [
        row for row in rows
        if row["split"] == "test"
        and methods["fastv"][row["sample_id"]]["dense"].dense_correct
        and methods["scope"][row["sample_id"]]["dense"].dense_correct
        and methods["fastv"][row["sample_id"]]["dense"].eligible_interventions
        and methods["scope"][row["sample_id"]]["dense"].eligible_interventions
    ]
    return {
        "n_shared_samples": len(shared),
        "answer_agreement_rate": _mean([float(row["answer_agree"]) for row in rows]),
        "correctness_agreement_rate": _mean([float(row["correctness_agree"]) for row in rows]),
        "eligible_intervention_agreement_rate": _mean([float(row["eligible_interventions_agree"]) for row in rows]),
        "n_any_disagreement": len(disagreements),
        "n_common_dense_correct_reliant_test": len(common_reliant),
        "disagreements": disagreements[:200],
    }


def _label_and_monotonicity_rows(
    methods: dict[str, dict[str, dict[str, Action]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_rows = []
    summary: dict[str, Any] = {}
    for method, samples in methods.items():
        patterns = Counter()
        reversals = []
        for sample_id, actions in sorted(samples.items()):
            dense = actions["dense"]
            pattern = "".join("1" if actions[str(b)].reference_safe else "0" for b in BUDGETS)
            patterns[pattern] += 1
            seen_safe = False
            monotonic = True
            for bit in pattern:
                if bit == "1":
                    seen_safe = True
                elif seen_safe:
                    monotonic = False
            if not monotonic:
                reversals.append(sample_id)
            safe_budgets = [b for b in BUDGETS if actions[str(b)].reference_safe]
            min_safe = min(safe_budgets) if safe_budgets else "full"
            for budget in BUDGETS:
                action = actions[str(budget)]
                label_rows.append(
                    {
                        "method": method,
                        "sample_id": sample_id,
                        "split": action.split,
                        "dataset": action.dataset,
                        "budget": budget,
                        "dense_correct": dense.dense_correct,
                        "dense_reliant": bool(dense.eligible_interventions),
                        "eligible_interventions": "|".join(dense.eligible_interventions),
                        "answer_fidelity": action.answer_fidelity,
                        "intervention_pass": action.intervention_pass,
                        "intervention_answer_fidelity": action.intervention_answer_fidelity,
                        "legacy_flip_safe": action.legacy_flip_safe,
                        "reference_safe": action.reference_safe,
                        "combined_gold_safe": action.combined_gold_safe,
                        "safety_pattern_64_128_256_432": pattern,
                        "minimum_observed_safe_budget": min_safe,
                        "monotonic": monotonic,
                    }
                )
        total = sum(patterns.values())
        monotonic_count = total - len(reversals)
        summary[method] = {
            "n": total,
            "monotonic_count": monotonic_count,
            "monotonic_rate": monotonic_count / total if total else None,
            "nonmonotonic_count": len(reversals),
            "pattern_counts": dict(sorted(patterns.items())),
            "nonmonotonic_sample_ids": reversals[:300],
        }
    return label_rows, summary


def _population_rows(
    methods: dict[str, dict[str, dict[str, Action]]], samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for method, states in methods.items():
        for split in ("validation", "test"):
            subset = [actions for actions in states.values() if actions["dense"].split == split]
            populations = {
                "full": subset,
                "dense_correct": [x for x in subset if x["dense"].dense_correct],
                "dense_correct_reliant": [
                    x for x in subset if x["dense"].dense_correct and x["dense"].eligible_interventions
                ],
            }
            for population, groups in populations.items():
                rows.append(
                    {
                        "method": method,
                        "split": split,
                        "population": population,
                        "n": len(groups),
                        "coverage_of_split": len(groups) / len(subset) if subset else None,
                        "dense_accuracy": _mean([float(x["dense"].is_correct) for x in groups]),
                        "dense_reliance_rate": _mean([float(bool(x["dense"].eligible_interventions)) for x in groups]),
                    }
                )
    return rows


def _controller_replay(
    methods: dict[str, dict[str, dict[str, Action]]], sample_by_id: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    details: dict[str, Any] = {}
    for method, states in methods.items():
        validation_ids = sorted(k for k, v in states.items() if v["dense"].split == "validation")
        test_ids = sorted(k for k, v in states.items() if v["dense"].split == "test")
        feature_names, train_x = _feature_matrix(validation_ids, sample_by_id)
        _, test_x = _feature_matrix(test_ids, sample_by_id, feature_names=feature_names)

        models = {}
        for budget in BUDGETS:
            labels = [int(states[sid][str(budget)].reference_safe) for sid in validation_ids]
            models[budget] = _fit_logistic(train_x, labels)
        test_probs = {
            sid: {budget: _predict_logistic(models[budget], test_x[idx]) for budget in BUDGETS}
            for idx, sid in enumerate(test_ids)
        }
        validation_probs = {
            sid: {budget: _predict_logistic(models[budget], train_x[idx]) for budget in BUDGETS}
            for idx, sid in enumerate(validation_ids)
        }

        policies: dict[str, dict[str, int | str]] = {
            "dense": {sid: "dense" for sid in test_ids},
            **{f"fixed_{budget}": {sid: budget for sid in test_ids} for budget in BUDGETS},
            "offline_oracle": {sid: _oracle_budget(states[sid]) for sid in test_ids},
        }
        for risk_limit in RISK_LIMITS:
            fixed_budget = _choose_fixed_budget(states, validation_ids, risk_limit)
            policies[f"validation_fixed_risk_{risk_limit:.2f}"] = {sid: fixed_budget for sid in test_ids}
            threshold = _choose_probability_threshold(
                states, validation_ids, validation_probs, risk_limit
            )
            policies[f"metadata_logistic_risk_{risk_limit:.2f}"] = {
                sid: _predicted_budget(test_probs[sid], threshold) for sid in test_ids
            }

        method_details = {
            "feature_names": feature_names,
            "validation_n": len(validation_ids),
            "test_n": len(test_ids),
            "probability_model": "four independent pure-Python logistic regressions trained on validation only",
            "test_predictions": {},
        }
        for policy, decisions in policies.items():
            summary = _summarize_policy(policy, decisions, states)
            summary["method"] = method
            rows.append(summary)
            if policy.startswith("metadata_logistic"):
                method_details["test_predictions"][policy] = [
                    {
                        "sample_id": sid,
                        "selected_budget": decisions[sid],
                        "probabilities": {str(k): v for k, v in test_probs[sid].items()},
                        "actually_safe": states[sid][
                            "dense" if decisions[sid] == "dense" else str(decisions[sid])
                        ].reference_safe,
                    }
                    for sid in test_ids
                ]
        details[method] = method_details
    return rows, details


def _group_crossfit_replay(
    root: Path,
    methods: dict[str, dict[str, dict[str, Action]]],
    sample_by_id: dict[str, dict[str, Any]],
    folds: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Development-only group-aware cross-fit over the full 1,200-case pool.

    Connected components join samples sharing either identical image content or
    an exact normalized question. This prevents both forms of overlap found by
    the split audit from crossing a fold boundary.
    """

    group_map = _connected_group_map(root, sample_by_id)
    group_sizes = Counter(group_map.values())
    fold_loads = [0] * folds
    fold_by_group: dict[str, int] = {}
    for group, size in sorted(
        group_sizes.items(),
        key=lambda item: (-item[1], hashlib.sha256(item[0].encode("utf-8")).hexdigest()),
    ):
        fold = min(range(folds), key=lambda idx: (fold_loads[idx], idx))
        fold_by_group[group] = fold
        fold_loads[fold] += size
    fold_by_sample = {sample_id: fold_by_group[group] for sample_id, group in group_map.items()}
    rows = []
    details: dict[str, Any] = {
        "folds": folds,
        "group_definition": "connected components over identical image hash OR exact normalized question",
        "n_groups": len(fold_by_group),
        "fold_sample_counts": dict(Counter(str(v) for v in fold_by_sample.values())),
        "methods": {},
    }
    for method, states in methods.items():
        all_ids = sorted(states)
        decisions_by_risk: dict[float, dict[str, int | str]] = {risk: {} for risk in RISK_LIMITS}
        thresholds_by_risk: dict[float, list[float]] = {risk: [] for risk in RISK_LIMITS}
        for outer_fold in range(folds):
            train_ids = [sid for sid in all_ids if fold_by_sample[sid] != outer_fold]
            held_ids = [sid for sid in all_ids if fold_by_sample[sid] == outer_fold]
            names, train_x = _feature_matrix(train_ids, sample_by_id)
            _, held_x = _feature_matrix(held_ids, sample_by_id, feature_names=names)
            models = {}
            for budget in BUDGETS:
                labels = [int(states[sid][str(budget)].reference_safe) for sid in train_ids]
                models[budget] = _fit_logistic(train_x, labels, steps=500)
            train_probs = {
                sid: {budget: _predict_logistic(models[budget], train_x[idx]) for budget in BUDGETS}
                for idx, sid in enumerate(train_ids)
            }
            held_probs = {
                sid: {budget: _predict_logistic(models[budget], held_x[idx]) for budget in BUDGETS}
                for idx, sid in enumerate(held_ids)
            }
            for risk in RISK_LIMITS:
                threshold = _choose_probability_threshold(states, train_ids, train_probs, risk)
                thresholds_by_risk[risk].append(threshold)
                decisions_by_risk[risk].update(
                    {sid: _predicted_budget(held_probs[sid], threshold) for sid in held_ids}
                )
        for risk, decisions in decisions_by_risk.items():
            summary = _summarize_policy(
                f"group_crossfit_metadata_logistic_risk_{risk:.2f}", decisions, states
            )
            summary["method"] = method
            summary["accounting_boundary"] += "; group-aware development cross-fit"
            rows.append(summary)
        details["methods"][method] = {
            "thresholds_by_risk": {
                f"{risk:.2f}": values for risk, values in thresholds_by_risk.items()
            }
        }
    return rows, details


def _connected_group_map(root: Path, sample_by_id: dict[str, dict[str, Any]]) -> dict[str, str]:
    parents = {sample_id: sample_id for sample_id in sample_by_id}

    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    hash_first: dict[str, str] = {}
    question_first: dict[str, str] = {}
    for sample_id, row in sample_by_id.items():
        path = root / "data/paper_1200" / str(row["image_path"])
        content_hash = _sha256(path)
        question = _norm(row["question"])
        if content_hash in hash_first:
            union(sample_id, hash_first[content_hash])
        else:
            hash_first[content_hash] = sample_id
        if question in question_first:
            union(sample_id, question_first[question])
        else:
            question_first[question] = sample_id
    return {sample_id: find(sample_id) for sample_id in sample_by_id}


def _summarize_policy(
    policy: str, decisions: dict[str, int | str], states: dict[str, dict[str, Action]]
) -> dict[str, Any]:
    chosen = [
        states[sid]["dense" if budget == "dense" else str(budget)]
        for sid, budget in decisions.items()
    ]
    full = chosen
    dense_correct = [a for a in chosen if states[a.sample_id]["dense"].dense_correct]
    reliant = [
        a for a in chosen
        if states[a.sample_id]["dense"].dense_correct
        and states[a.sample_id]["dense"].eligible_interventions
    ]
    energies = [a.original_energy_joule for a in chosen if a.original_energy_joule is not None]
    latencies = [a.original_latency_ms for a in chosen if a.original_latency_ms is not None]
    supported_correct = [
        a for a in reliant if a.is_correct and a.intervention_pass
    ]
    return {
        "policy": policy,
        "n_full": len(full),
        "unconditional_accuracy": _mean([float(a.is_correct) for a in full]),
        "dense_behavior_safe_rate": _mean([float(a.reference_safe) for a in full]),
        "unsafe_acceptance_rate": _mean(
            [float(a.budget != "full" and not a.reference_safe) for a in full]
        ),
        "dense_avoidance_rate": _mean([float(a.budget != "full") for a in full]),
        "mean_token_count": _mean([float(a.token_count) for a in full]),
        "mean_original_call_latency_ms": _mean(latencies),
        "mean_original_call_energy_joule": _mean(energies),
        "n_dense_correct": len(dense_correct),
        "dense_correct_answer_preservation": _mean([float(a.answer_fidelity) for a in dense_correct]),
        "n_dense_correct_reliant": len(reliant),
        "reliant_intervention_preservation": _mean([float(a.intervention_pass) for a in reliant]),
        "reliant_intervention_trajectory_fidelity": _mean(
            [float(a.intervention_answer_fidelity) for a in reliant]
        ),
        "reliant_combined_safe_rate": _mean([float(bool(a.combined_gold_safe)) for a in reliant]),
        "supported_correct_answers_per_kwh_full_denominator": (
            len(supported_correct) / (sum(energies) / 3_600_000.0)
            if energies and sum(energies) > 0 else None
        ),
        "selected_budget_distribution": json.dumps(
            dict(sorted(Counter(str(v) for v in decisions.values()).items()))
        ),
        "accounting_boundary": "original model call only; evaluator-only interventions excluded",
    }


def _choose_fixed_budget(
    states: dict[str, dict[str, Action]], validation_ids: list[str], risk_limit: float
) -> int | str:
    for budget in BUDGETS:
        unsafe = _mean([float(not states[sid][str(budget)].reference_safe) for sid in validation_ids])
        if unsafe is not None and unsafe <= risk_limit:
            return budget
    return "dense"


def _choose_probability_threshold(
    states: dict[str, dict[str, Action]],
    validation_ids: list[str],
    probabilities: dict[str, dict[int, float]],
    risk_limit: float,
) -> float:
    candidates = [i / 100.0 for i in range(5, 100, 5)]
    feasible = []
    for threshold in candidates:
        decisions = {sid: _predicted_budget(probabilities[sid], threshold) for sid in validation_ids}
        chosen = [
            states[sid]["dense" if budget == "dense" else str(budget)]
            for sid, budget in decisions.items()
        ]
        unsafe = _mean([float(a.budget != "full" and not a.reference_safe) for a in chosen]) or 0.0
        energy = _mean([a.original_energy_joule for a in chosen if a.original_energy_joule is not None])
        if unsafe <= risk_limit:
            feasible.append((float("inf") if energy is None else energy, -threshold, threshold))
    return min(feasible)[2] if feasible else 1.0


def _predicted_budget(probabilities: dict[int, float], threshold: float) -> int | str:
    for budget in BUDGETS:
        if probabilities[budget] >= threshold:
            return budget
    return "dense"


def _oracle_budget(actions: dict[str, Action]) -> int | str:
    for budget in BUDGETS:
        if actions[str(budget)].reference_safe:
            return budget
    return "dense"


def _feature_matrix(
    sample_ids: list[str],
    sample_by_id: dict[str, dict[str, Any]],
    feature_names: list[str] | None = None,
) -> tuple[list[str], list[list[float]]]:
    datasets = [
        "mmstar",
        "pope_adversarial",
        "pope_popular",
        "pope_random",
        "visual_counterfact_color",
        "visual_counterfact_size",
    ]
    keywords = {
        "color": ("color", "colour"),
        "size": ("size", "larger", "smaller", "big", "small"),
        "count": ("how many", "number of", "count"),
        "existence": ("is there", "are there", "does the image", "present"),
        "spatial": ("left", "right", "above", "below", "behind", "front"),
        "text": ("word", "text", "written", "read", "letter"),
        "emotion": ("feeling", "mood", "emotion"),
    }
    names = ["word_count", "char_count", "option_count", "is_multiple_choice"]
    names += [f"dataset={name}" for name in datasets]
    names += [f"keyword={name}" for name in keywords]
    if feature_names is not None and feature_names != names:
        raise ValueError("Feature schema mismatch")
    matrix = []
    for sample_id in sample_ids:
        row = sample_by_id[sample_id]
        q = str(row["question"]).lower()
        options = row.get("options") or []
        values = [
            min(len(q.split()) / 50.0, 2.0),
            min(len(q) / 300.0, 2.0),
            len(options) / 4.0,
            float(bool(options)),
        ]
        values += [float(row["dataset"] == name) for name in datasets]
        values += [float(any(token in q for token in tokens)) for tokens in keywords.values()]
        matrix.append(values)
    return names, matrix


def _fit_logistic(x: list[list[float]], y: list[int], steps: int = 1200, rate: float = 0.08) -> list[float]:
    if not x:
        return [0.0]
    weights = [0.0] * (len(x[0]) + 1)
    positive = sum(y)
    if positive in (0, len(y)):
        prior = (positive + 0.5) / (len(y) + 1.0)
        weights[0] = math.log(prior / (1.0 - prior))
        return weights
    for _ in range(steps):
        grad = [0.0] * len(weights)
        for row, label in zip(x, y, strict=True):
            pred = _sigmoid(weights[0] + sum(w * value for w, value in zip(weights[1:], row, strict=True)))
            error = pred - label
            grad[0] += error
            for idx, value in enumerate(row, start=1):
                grad[idx] += error * value
        for idx in range(len(weights)):
            penalty = 0.0 if idx == 0 else 0.02 * weights[idx]
            weights[idx] -= rate * (grad[idx] / len(x) + penalty)
    return weights


def _predict_logistic(weights: list[float], row: list[float]) -> float:
    return _sigmoid(weights[0] + sum(w * value for w, value in zip(weights[1:], row, strict=True)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _qualitative_candidates(
    methods: dict[str, dict[str, dict[str, Action]]],
    sample_by_id: dict[str, dict[str, Any]],
    controller_details: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "status": "candidate set for blinded human semantic review; automatic fields are not human judgments",
        "selection_rule": "deterministic stratification by failure/mechanism category, method, and dataset",
        "cases": [],
    }
    for method, states in methods.items():
        categories: dict[str, list[str]] = defaultdict(list)
        for sample_id, actions in states.items():
            if actions["dense"].split != "test":
                continue
            pattern = actions["64"].pattern
            if "10" in pattern:
                categories["nonmonotonic"].append(sample_id)
            if actions["64"].reference_safe:
                categories["safe_64"].append(sample_id)
            if not any(actions[str(b)].reference_safe for b in BUDGETS):
                categories["requires_dense"].append(sample_id)
            if len(actions["dense"].eligible_interventions) > 1:
                passes = {name: actions["64"].per_intervention.get(name, {}).get("vem") for name in actions["dense"].eligible_interventions}
                if len(set(passes.values())) > 1:
                    categories["intervention_disagreement"].append(sample_id)
        for category, sample_ids in sorted(categories.items()):
            selected = _stratified_take(sample_ids, states, 4)
            for sample_id in selected:
                sample = sample_by_id[sample_id]
                dense = states[sample_id]["dense"]
                output["cases"].append(
                    {
                        "method": method,
                        "category": category,
                        "sample_id": sample_id,
                        "dataset": dense.dataset,
                        "question": sample["question"],
                        "gold_answer": sample["answer"],
                        "dense_answer": dense.answer,
                        "dense_eligible_interventions": list(dense.eligible_interventions),
                        "budget_pattern_64_128_256_432": dense.pattern,
                        "image_path": str(sample["image_path"]),
                        "counterfactual_image_path": sample.get("counterfactual_image_path"),
                        "human_semantic_validity": "PENDING",
                        "human_notes": "",
                    }
                )
    return output


def _stratified_take(sample_ids: list[str], states: dict[str, dict[str, Action]], count: int) -> list[str]:
    selected = []
    seen_datasets = set()
    for sample_id in sorted(sample_ids):
        dataset = states[sample_id]["dense"].dataset
        if dataset not in seen_datasets:
            selected.append(sample_id)
            seen_datasets.add(dataset)
        if len(selected) == count:
            return selected
    for sample_id in sorted(sample_ids):
        if sample_id not in selected:
            selected.append(sample_id)
        if len(selected) == count:
            break
    return selected


def _cost_boundary_audit(methods: dict[str, dict[str, dict[str, Action]]]) -> dict[str, Any]:
    result = {
        "finding": (
            "EvalRecord latency/energy includes original plus intervention calls, and the cached original call is repeated "
            "in every intervention record. Summing records therefore duplicates original-call cost. This audit uses unique "
            "original backend timestamps and raw nvidia-smi samples for deployment replay."
        ),
        "intervention_terminology_finding": (
            "The existing config uses type='gray', which materializes a uniform RGB(128,128,128) null image. "
            "It is not a grayscale conversion. Frozen results must therefore be labeled constant-gray/null-image, "
            "not grayscale. A distinct type='grayscale' implementation has been added for future experiments."
        ),
        "methods": {},
    }
    for method, states in methods.items():
        actions = [action for group in states.values() for action in group.values()]
        available = [a for a in actions if a.original_energy_joule is not None]
        result["methods"][method] = {
            "n_actions": len(actions),
            "n_actions_with_original_call_energy": len(available),
            "coverage": len(available) / len(actions) if actions else None,
            "mean_original_call_energy_by_budget": {
                str(budget): _mean(
                    [
                        a.original_energy_joule
                        for a in actions
                        if str(a.budget) == str(budget) and a.original_energy_joule is not None
                    ]
                )
                for budget in (*BUDGETS, "full")
            },
            "mean_original_call_latency_by_budget_ms": {
                str(budget): _mean(
                    [
                        a.original_latency_ms
                        for a in actions
                        if str(a.budget) == str(budget) and a.original_latency_ms is not None
                    ]
                )
                for budget in (*BUDGETS, "full")
            },
        }
    return result


def _write_framework_spec(path: Path) -> None:
    path.write_text(
        """# ViRel-Budget Framework Contract (No-GPU Repair)

## Primary operational claim

The deployable controller predicts a pruning action using only features available before or during its counted execution path. Dense answers, gold answers, dense-reliance status, intervention labels, and unexecuted candidate outputs are forbidden test-time inputs.

## Separate outcomes

1. **Answer fidelity:** the pruned original-image answer matches the dense reference answer.
2. **Task correctness:** the pruned original-image answer matches the gold answer.
3. **Legacy flip preservation:** every intervention that changes the dense answer also changes the pruned answer. This is necessary but not sufficient.
4. **Intervention trajectory fidelity:** for every dense-eligible intervention, the pruned intervened answer matches the dense intervened answer.
5. **Reference safety:** original-answer fidelity AND intervention trajectory fidelity.
6. **Combined gold safety:** on dense-correct, dense-reliant cases, task correctness AND intervention trajectory fidelity.

Reference safety is the controller-training target because it measures compression-induced behavioral drift without pretending that a controller can repair an incorrect dense model. The earlier flip-only result is retained as a legacy diagnostic, not the repaired safety target. Gold correctness and combined gold safety remain evaluation outcomes.

## Intervention interpretation

VEM is an operational measure of intervention-defined visual sensitivity. It is not a semantic-grounding certificate. Targeted counterfactual, irrelevant replacement, blur, and grayscale results must be reported separately before aggregation.

## Populations

- Full held-out population: unconditional accuracy, dense-behavior safety, energy/query, and operational dense avoidance.
- Dense-correct population: preservation of dense-correct answers.
- Dense-correct and dense-reliant population: intervention-response and combined-safety preservation.

## Cost boundaries

- **Deployment boundary:** every model call actually used by the controller plus controller overhead.
- **Evaluator boundary:** dense and intervention calls used only to create hidden labels; reported separately and never charged as deployment calls.
- **Oracle boundary:** selected-output result after observing the full grid; upper bound only.

## Final claim rule

The method may be called a deployable green controller only if a frozen held-out policy satisfies a predeclared unsafe-acceptance limit and reduces repeated, end-to-end measured energy relative to dense inference. Otherwise it must be framed as a reliability audit, risk-efficiency frontier, or oracle attainable-efficiency bound.
""",
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    split: dict[str, Any],
    dense: dict[str, Any],
    monotonicity: dict[str, Any],
    populations: list[dict[str, Any]],
    controllers: list[dict[str, Any]],
    cost: dict[str, Any],
    qualitative: dict[str, Any],
) -> None:
    lines = [
        "# ViRel-Budget No-GPU Framework Repair Report",
        "",
        "This report is generated exclusively from the frozen 1,200-case artifacts. No GPU inference was run.",
        "",
        "## Executive findings",
        "",
        f"- Split audit: {split['cross_split_content_hash_count']} content-hash overlaps, "
        f"{split['cross_split_exact_question_count']} exact-question overlaps, and "
        f"{split['cross_split_source_group_count']} source-group overlaps across validation/test.",
        f"- Dense implementations agree on answers for {dense['answer_agreement_rate']:.2%} of shared cases and "
        f"on eligible intervention sets for {dense['eligible_intervention_agreement_rate']:.2%}.",
        f"- Common dense-correct, dense-reliant test intersection: {dense['n_common_dense_correct_reliant_test']} cases.",
    ]
    for method, values in monotonicity.items():
        lines.append(
            f"- {method}: reference-safety is monotonic across 64/128/256/432 for "
            f"{values['monotonic_rate']:.2%} of cases; {values['nonmonotonic_count']} cases contain reversals."
        )
    lines += [
        "",
        "## Critical accounting correction",
        "",
        cost["finding"],
        "",
        "Consequently, earlier policy summaries must not be treated as deployment energy measurements. "
        "The controller replay in this report uses only each action's unique original-image call. "
        "Evaluator-only intervention calls are excluded from deployment cost and must never influence held-out decisions.",
        "",
        "## Intervention implementation correction",
        "",
        cost["intervention_terminology_finding"],
        "",
        "## Population audit",
        "",
        _markdown_table(
            populations,
            ["method", "split", "population", "n", "coverage_of_split", "dense_accuracy", "dense_reliance_rate"],
        ),
        "",
        "## No-GPU controller replay",
        "",
        "The metadata-logistic controller is deliberately weak and interpretable. It uses validation-only question structure, "
        "task keywords, and dataset indicators; it does not use dense/test labels, confidence, attention, or unexecuted budgets.",
        "",
        "**Important:** because the split audit found cross-split image and question duplication, this is a developmental "
        "feasibility replay, not a publishable prospective test result. It may guide feature instrumentation, but the final "
        "controller must be frozen before evaluation on group-isolated new cases.",
        "",
        _markdown_table(
            controllers,
            [
                "method",
                "policy",
                "unconditional_accuracy",
                "dense_behavior_safe_rate",
                "unsafe_acceptance_rate",
                "dense_avoidance_rate",
                "mean_original_call_energy_joule",
                "reliant_combined_safe_rate",
            ],
        ),
        "",
        "## Qualitative audit status",
        "",
        f"{len(qualitative['cases'])} blinded candidates were selected. Automatic categories are not semantic judgments; "
        "the PENDING fields must be completed by human inspection before intervention-validity claims are revised.",
        "",
        "## GPU decision rule",
        "",
        "Proceed to a 120-case feature-instrumentation pilot only if the no-GPU results show that trivial policies do not already "
        "dominate and if confidence/attention features have a specific opportunity to reduce unsafe acceptance. The pilot must "
        "be frozen before any paper-scale controller execution.",
        "",
        "## Immediate framework decision",
        "",
        "The existing 1,200 cases should now be treated as a development pool. Use group-aware cross-validation for model "
        "development. The future 900-case expansion should be sampled and group-isolated before inference, then reserved as "
        "the prospective controller test set. This repairs the discovered leakage without discarding the completed experiments.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    shown = rows
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in shown:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_power_samples(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({"epoch": float(row["epoch"]), "power_w": float(row["power_w"])})
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows, key=lambda row: row["epoch"])


def _call_energy(call: dict[str, Any], power: list[dict[str, float]]) -> float | None:
    start = _float_or_none(call.get("start_epoch"))
    end = _float_or_none(call.get("end_epoch"))
    if start is None or end is None or end <= start:
        return None
    values = [row["power_w"] for row in power if start <= row["epoch"] <= end]
    if not values and power:
        midpoint = (start + end) / 2.0
        values = [min(power, key=lambda row: abs(row["epoch"] - midpoint))["power_w"]]
    return statistics.fmean(values) * (end - start) if values else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _intervened_answer(record: dict[str, Any], sample: dict[str, Any]) -> str:
    raw = str(
        (((record.get("metadata") or {}).get("intervened_backend") or {}).get("raw_answer", ""))
    )
    metadata = sample.get("metadata") or {}
    option_map = metadata.get("option_map") if isinstance(metadata.get("option_map"), dict) else None
    return canonicalize_answer(raw, option_map=option_map, options=sample.get("options") or [])


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
