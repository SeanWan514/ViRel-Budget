from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BUDGETS = (64, 128, 256, 432)
DATASETS = (
    "mmstar",
    "pope_adversarial",
    "pope_popular",
    "pope_random",
    "visual_counterfact_color",
    "visual_counterfact_size",
)
KEYWORDS = {
    "color": ("color", "colour"),
    "size": ("size", "larger", "smaller", "big", "small"),
    "count": ("how many", "number of", "count"),
    "existence": ("is there", "are there", "does the image", "present"),
    "spatial": ("left", "right", "above", "below", "behind", "front"),
    "text": ("word", "text", "written", "read", "letter"),
    "emotion": ("feeling", "mood", "emotion"),
}
FEATURE_NAMES = (
    "question_word_count",
    "question_character_count",
    "option_count",
    "multiple_choice",
    *(f"dataset={name}" for name in DATASETS),
    *(f"keyword={name}" for name in KEYWORDS),
)
FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "answer",
        "gold_answer",
        "dense_answer",
        "dense_correct",
        "dense_vem",
        "dense_reliant",
        "eligible_interventions",
        "intervention_pass",
        "intervention_answer_fidelity",
        "reference_safe",
        "combined_gold_safe",
        "minimum_observed_safe_budget",
    }
)


def controller_features(record: dict[str, Any], *, reject_forbidden: bool = True) -> list[float]:
    if reject_forbidden:
        forbidden = FORBIDDEN_INPUT_KEYS & set(record)
        if forbidden:
            raise ValueError(f"Forbidden controller input fields present: {sorted(forbidden)}")
    question = str(record["question"])
    lowered = question.lower()
    options = list(record.get("options") or [])
    dataset = str(record["dataset"])
    values = [
        min(len(question.split()) / 50.0, 2.0),
        min(len(question) / 300.0, 2.0),
        min(len(options) / 4.0, 2.0),
        float(bool(options)),
        *(float(dataset == name) for name in DATASETS),
        *(float(any(token in lowered for token in tokens)) for tokens in KEYWORDS.values()),
    ]
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("Controller feature schema mismatch")
    return values


def fit_logistic(
    rows: list[list[float]],
    labels: list[int],
    *,
    steps: int = 1600,
    learning_rate: float = 0.08,
    l2: float = 0.03,
) -> list[float]:
    if not rows:
        raise ValueError("Cannot fit controller without rows")
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        matrix = np.asarray(rows, dtype=float)
        targets = np.asarray(labels, dtype=float)
        weights = np.zeros(matrix.shape[1] + 1, dtype=float)
        for _ in range(steps):
            logits = weights[0] + matrix @ weights[1:]
            probabilities = np.where(
                logits >= 0,
                1.0 / (1.0 + np.exp(-logits)),
                np.exp(logits) / (1.0 + np.exp(logits)),
            )
            errors = probabilities - targets
            weights[0] -= learning_rate * float(errors.mean())
            gradient = matrix.T @ errors / len(matrix) + l2 * weights[1:]
            weights[1:] -= learning_rate * gradient
        return [float(value) for value in weights]
    weights = [0.0] * (len(rows[0]) + 1)
    for _ in range(steps):
        gradient = [0.0] * len(weights)
        for row, label in zip(rows, labels, strict=True):
            probability = sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], row, strict=True)))
            error = probability - label
            gradient[0] += error
            for index, value in enumerate(row, 1):
                gradient[index] += error * value
        for index in range(len(weights)):
            penalty = 0.0 if index == 0 else l2 * weights[index]
            weights[index] -= learning_rate * (gradient[index] / len(rows) + penalty)
    return weights


def fit_standardizer(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    means = [sum(row[j] for row in rows) / len(rows) for j in range(len(rows[0]))]
    scales: list[float] = []
    for j, mean in enumerate(means):
        variance = sum((row[j] - mean) ** 2 for row in rows) / max(1, len(rows) - 1)
        scales.append(max(math.sqrt(variance), 1e-8))
    return means, scales


def standardize(row: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - means[j]) / scales[j] for j, value in enumerate(row)]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class ControllerDecision:
    selected_budget: int | str
    probabilities: dict[str, float]
    controller_latency_ms: float


class FrozenBudgetController:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        if artifact["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Frozen controller feature schema does not match runtime")
        if artifact["selection_rule"] != "smallest_budget_above_global_threshold_else_dense":
            raise ValueError("Unsupported frozen controller selection rule")

    @classmethod
    def load(cls, path: str | Path, checksum_path: str | Path | None = None) -> "FrozenBudgetController":
        model_path = Path(path)
        payload = model_path.read_bytes()
        if checksum_path is not None:
            expected = Path(checksum_path).read_text(encoding="utf-8").split()[0]
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                raise ValueError(f"Controller checksum mismatch: expected {expected}, got {actual}")
        return cls(json.loads(payload))

    def decide(self, deployment_record: dict[str, Any]) -> ControllerDecision:
        started = time.perf_counter()
        raw = controller_features(deployment_record, reject_forbidden=True)
        probabilities: dict[str, float] = {}
        threshold = float(self.artifact["global_threshold"])
        selected: int | str = "dense"
        for budget in BUDGETS:
            model = self.artifact["models"][str(budget)]
            row = standardize(raw, model["means"], model["scales"])
            probability = sigmoid(
                model["weights"][0]
                + sum(w * x for w, x in zip(model["weights"][1:], row, strict=True))
            )
            probabilities[str(budget)] = probability
            if selected == "dense" and probability >= threshold:
                selected = budget
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ControllerDecision(selected, probabilities, elapsed_ms)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
