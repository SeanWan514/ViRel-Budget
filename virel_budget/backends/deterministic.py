from __future__ import annotations

import math
import time
from pathlib import Path

from virel_budget.backends.base import VLMBackend
from virel_budget.images import deterministic_seed, token_saliency
from virel_budget.pruning import select_tokens
from virel_budget.schema import Budget, Sample, ScoreResult


class DeterministicBackend(VLMBackend):
    """Offline sanity backend.

    This backend is intentionally deterministic and dependency-light. It exists to test
    pipeline mechanics and should never be reported as paper evidence.
    """

    name = "deterministic"

    def __init__(self, dense_token_count: int = 64, profile: str = "offline-sanity") -> None:
        self.dense_token_count = dense_token_count
        self.profile = profile

    def score_options(
        self,
        sample: Sample,
        image_path: Path,
        method: str,
        budget: Budget,
        seed: int,
    ) -> ScoreResult:
        started = time.perf_counter()
        options = sample.options or ["yes", "no"]
        selection = select_tokens(method, budget, self.dense_token_count, image_path=str(image_path), sample_id=sample.sample_id, seed=seed)
        saliency = token_saliency(image_path, grid_size=int(math.sqrt(self.dense_token_count)))
        retained_signal = float(saliency[list(selection.selected_indices)].mean()) if selection.selected_indices else 0.0
        gold_idx = _find_gold(options, sample.answer)
        image_penalty = 1.0 - min(retained_signal * 4.0, 1.0)
        budget_penalty = 1.0 - (selection.token_count / max(self.dense_token_count, 1))
        variant_penalty = _variant_penalty(image_path)
        scores = []
        for idx, option in enumerate(options):
            base = 1.5 if idx == gold_idx else 0.2
            noise = _noise(seed, sample.sample_id, option, method, budget)
            score = base + retained_signal * 2.5 - image_penalty * 0.8 - budget_penalty * 0.3 - variant_penalty + noise
            if idx != gold_idx:
                score += variant_penalty * 0.9 + budget_penalty * 0.15
            scores.append(score)
        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        logprob = _log_softmax(scores)[pred_idx]
        latency_ms = 2.0 + 0.08 * selection.token_count + (time.perf_counter() - started) * 1000.0
        confidence = math.exp(logprob)
        return ScoreResult(
            answer=options[pred_idx],
            logprob=float(logprob),
            confidence=float(confidence),
            latency_ms=float(latency_ms),
            token_count=selection.token_count,
            method=method,
            budget=budget,
            image_variant="computed",
            metadata={"profile": self.profile, "retained_signal": retained_signal},
        )


def _find_gold(options: list[str], answer: str) -> int:
    answer_norm = answer.strip().lower()
    for idx, option in enumerate(options):
        if option.strip().lower() == answer_norm:
            return idx
    return 0


def _noise(seed: int, *parts: object) -> float:
    value = deterministic_seed(seed, *parts) % 1000
    return (value / 1000.0 - 0.5) * 0.08


def _variant_penalty(image_path: Path) -> float:
    name = image_path.stem.lower()
    if "gray" in name or "black" in name:
        return 1.25
    if "blur" in name:
        return 0.55
    if "irrelevant" in name:
        return 1.05
    return 0.0


def _log_softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    denom = sum(exps)
    return [s - m - math.log(denom) for s in scores]
