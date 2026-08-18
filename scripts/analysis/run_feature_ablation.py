from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "feature_analysis"
sys.path.insert(0, str(ROOT))

from virel_budget.frozen_controller import (
    BUDGETS,
    FEATURE_NAMES,
    fit_logistic,
    fit_standardizer,
    sigmoid,
    standardize,
)


CELLS = (
    "7b_fastv",
    "7b_scope",
    "7b_random",
    "13b_fastv",
    "13b_scope",
    "13b_random",
)
GROUPS = {
    "question_structure": tuple(range(0, 4)),
    "task_source": tuple(range(4, 10)),
    "semantic_keywords": tuple(range(10, 17)),
}
CONFIGURATIONS = {
    "A0_full": ("question_structure", "task_source", "semantic_keywords"),
    "A1_structure_only": ("question_structure",),
    "A2_task_only": ("task_source",),
    "A3_keywords_only": ("semantic_keywords",),
    "A4_full_minus_task": ("question_structure", "semantic_keywords"),
    "A5_full_minus_keywords": ("question_structure", "task_source"),
    "A6_intercept_only": (),
}


def feature_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def labels(path: Path, method: str) -> dict[str, dict[int, bool]]:
    output: dict[str, dict[int, bool]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            budget = int(row["budget"])
            if row["method"] == method and budget in BUDGETS:
                output[row["sample_id"]][budget] = row["reference_safe"] == "True"
    return output


def indices(groups: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(index for group in groups for index in GROUPS[group])


def feature_row(record: dict[str, Any], selected: tuple[int, ...]) -> list[float]:
    full = [float(record[name]) for name in FEATURE_NAMES]
    return [full[index] for index in selected]


def fit_models(
    records: list[dict[str, Any]],
    label_set: dict[str, dict[int, bool]],
    selected: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    matrix = [feature_row(row, selected) for row in records]
    if selected:
        means, scales = fit_standardizer(matrix)
        transformed = [standardize(row, means, scales) for row in matrix]
    else:
        means, scales, transformed = [], [], matrix
    models: dict[int, dict[str, Any]] = {}
    for budget in BUDGETS:
        weights = fit_logistic(
            transformed,
            [int(label_set[row["sample_id"]][budget]) for row in records],
        )
        models[budget] = {"weights": weights, "means": means, "scales": scales}
    return models


def probability(model: dict[str, Any], raw: list[float]) -> float:
    row = standardize(raw, model["means"], model["scales"]) if raw else []
    return sigmoid(model["weights"][0] + sum(w * x for w, x in zip(model["weights"][1:], row, strict=True)))


def wilson_upper(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return 1.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (center + radius) / denominator


def select(probabilities: dict[int, float], threshold: float) -> int | str:
    return next((budget for budget in BUDGETS if probabilities[budget] >= threshold), "dense")


def auc(labels_: list[int], scores: list[float]) -> float | None:
    positives = sum(labels_)
    negatives = len(labels_) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels_, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def diagnostics(labels_: list[int], scores: list[float]) -> dict[str, float | None]:
    epsilon = 1e-15
    clipped = [min(max(score, epsilon), 1.0 - epsilon) for score in scores]
    return {
        "auroc": auc(labels_, scores),
        "brier": sum((score - label) ** 2 for score, label in zip(scores, labels_, strict=True)) / len(labels_),
        "log_loss": -sum(
            label * math.log(score) + (1 - label) * math.log(1 - score)
            for score, label in zip(clipped, labels_, strict=True)
        ) / len(labels_),
    }


def evaluate_configuration(
    records: list[dict[str, Any]],
    label_set: dict[str, dict[int, bool]],
    folds: dict[str, int],
    selected: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    oof: dict[str, dict[int, float]] = {}
    for fold in range(5):
        train = [row for row in records if folds[row["sample_id"]] != fold]
        test = [row for row in records if folds[row["sample_id"]] == fold]
        models = fit_models(train, label_set, selected)
        for row in test:
            raw = feature_row(row, selected)
            oof[row["sample_id"]] = {
                budget: probability(models[budget], raw) for budget in BUDGETS
            }

    candidates = sorted({score for values in oof.values() for score in values.values()}, reverse=True)
    candidates = [1.000001, *candidates, 0.0]
    feasible: list[tuple[float, float, float, dict[str, Any], dict[str, int | str]]] = []
    for threshold in candidates:
        decisions = {
            row["sample_id"]: select(oof[row["sample_id"]], threshold) for row in records
        }
        unsafe = sum(
            action != "dense" and not label_set[sample_id][int(action)]
            for sample_id, action in decisions.items()
        )
        upper = wilson_upper(unsafe, len(records))
        if upper > 0.05:
            continue
        avoidance = sum(action != "dense" for action in decisions.values()) / len(records)
        mean_tokens = sum(576 if action == "dense" else int(action) for action in decisions.values()) / len(records)
        report = {
            "threshold": threshold,
            "unsafe_acceptance_count": unsafe,
            "unsafe_acceptance_rate": unsafe / len(records),
            "wilson_upper": upper,
            "risk_limit_met": upper <= 0.05,
            "dense_avoidance_rate": avoidance,
            "dense_fallback_rate": 1.0 - avoidance,
            "mean_selected_visual_tokens": mean_tokens,
            "median_selected_visual_tokens": None,
            "selected_action_distribution": dict(Counter(str(value) for value in decisions.values())),
        }
        feasible.append((avoidance, -mean_tokens, threshold, report, decisions))
    _, _, _, report, decisions = max(feasible)
    token_values = sorted(576 if action == "dense" else int(action) for action in decisions.values())
    report["median_selected_visual_tokens"] = (token_values[599] + token_values[600]) / 2.0

    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        sample_ids = [row["sample_id"] for row in records if folds[row["sample_id"]] == fold]
        unsafe = sum(
            decisions[sample_id] != "dense"
            and not label_set[sample_id][int(decisions[sample_id])]
            for sample_id in sample_ids
        )
        avoidance = sum(decisions[sample_id] != "dense" for sample_id in sample_ids) / len(sample_ids)
        fold_rows.append(
            {
                "fold": fold,
                "n": len(sample_ids),
                "unsafe_acceptance_count": unsafe,
                "unsafe_acceptance_rate": unsafe / len(sample_ids),
                "dense_avoidance_rate": avoidance,
            }
        )

    diagnostic_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        labels_ = [int(label_set[row["sample_id"]][budget]) for row in records]
        scores = [oof[row["sample_id"]][budget] for row in records]
        diagnostic_rows.append({"budget": budget, **diagnostics(labels_, scores)})
    return report, fold_rows, diagnostic_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    protocol = OUTPUT / "feature_ablation_protocol.json"
    sample_path = OUTPUT / "development_features_and_folds.csv"
    records = feature_records(sample_path)
    folds = {row["sample_id"]: int(row["fold"]) for row in records}

    main_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for cell in CELLS:
        model, method = cell.split("_", 1)
        label_path = (
            ROOT
            / "results"
            / "controllers"
            / "labels"
            / "development"
            / cell
            / "safe_budget_labels.csv"
        )
        label_set = labels(label_path, method)
        if len(label_set) != len(records) or any(set(value) != set(BUDGETS) for value in label_set.values()):
            raise ValueError(f"Incomplete labels for {cell}")
        for configuration, feature_groups in CONFIGURATIONS.items():
            selected = indices(feature_groups)
            report, folds_report, diagnostics_report = evaluate_configuration(
                records, label_set, folds, selected
            )
            common = {
                "cell": cell,
                "model": model,
                "method": method,
                "configuration": configuration,
                "feature_groups": "+".join(feature_groups) if feature_groups else "intercept_only",
                "feature_count": len(selected),
            }
            main_rows.append(
                {
                    **common,
                    "n": len(records),
                    "threshold": report["threshold"],
                    "unsafe_acceptance_count": report["unsafe_acceptance_count"],
                    "unsafe_acceptance_rate": report["unsafe_acceptance_rate"],
                    "wilson_upper": report["wilson_upper"],
                    "risk_limit_met": report["risk_limit_met"],
                    "dense_avoidance_rate": report["dense_avoidance_rate"],
                    "dense_fallback_rate": report["dense_fallback_rate"],
                    "mean_selected_visual_tokens": report["mean_selected_visual_tokens"],
                    "median_selected_visual_tokens": report["median_selected_visual_tokens"],
                }
            )
            for action in ("64", "128", "256", "432", "dense"):
                distribution_rows.append(
                    {
                        **common,
                        "action": action,
                        "count": report["selected_action_distribution"].get(action, 0),
                    }
                )
            for row in folds_report:
                fold_rows.append({**common, **row})
            for row in diagnostics_report:
                diagnostic_rows.append({**common, **row})
            print(json.dumps({**common, **report}, sort_keys=True))

    write_csv(OUTPUT / "feature_ablation_by_controller.csv", main_rows)
    write_csv(OUTPUT / "feature_ablation_budget_distributions.csv", distribution_rows)
    write_csv(OUTPUT / "feature_ablation_fold_stability.csv", fold_rows)
    write_csv(OUTPUT / "feature_ablation_predictor_diagnostics.csv", diagnostic_rows)
    run_manifest = {
        "protocol_sha256": __import__("hashlib").sha256(protocol.read_bytes()).hexdigest(),
        "development_features_sha256": __import__("hashlib").sha256(sample_path.read_bytes()).hexdigest(),
        "original_development_samples_sha256": "4c9d8e92b594c9ac8e2bc549f5759188a22a58581d910e9793f37f267bb66de0",
        "development_query_count": len(records),
        "group_count": 575,
        "fold_sizes": dict(Counter(str(value) for value in folds.values())),
        "prospective_labels_used": False,
        "configuration_count": len(CONFIGURATIONS),
        "controller_count": len(CELLS),
        "completed_cells": len(main_rows),
    }
    (OUTPUT / "feature_ablation_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
