from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


FORBIDDEN_DEPLOYMENT_KEYS = frozenset(
    {
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


@dataclass(frozen=True)
class DeploymentFeatures:
    sample_id: str
    method: str
    executed_budget: int
    dataset: str
    question_word_count: int
    option_count: int
    answer: str
    generated_token_count: int
    mean_token_logprob: float | None
    min_token_logprob: float | None
    final_token_logprob: float | None
    first_token_margin: float | None
    mean_predictive_entropy: float | None
    first_token_entropy: float | None
    feature_overhead_ms: float
    model_latency_ms: float
    token_count: int

    @classmethod
    def from_pilot_record(cls, record: dict[str, Any]) -> "DeploymentFeatures":
        forbidden = FORBIDDEN_DEPLOYMENT_KEYS & set(record)
        if forbidden:
            raise ValueError(f"Forbidden deployment fields present: {sorted(forbidden)}")
        allowed = {field.name for field in fields(cls)}
        unexpected = set(record) - allowed
        if unexpected:
            raise ValueError(f"Unexpected deployment fields present: {sorted(unexpected)}")
        return cls(**record)
