from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


Budget = int | str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    dataset: str
    image_path: Path
    question: str
    answer: str
    options: list[str] = field(default_factory=list)
    counterfactual_image_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_answer(self) -> str:
        return normalize_answer(self.answer)


@dataclass(frozen=True)
class Intervention:
    name: str
    type: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreResult:
    answer: str
    logprob: float
    confidence: float | None
    latency_ms: float
    token_count: int
    method: str
    budget: Budget
    image_variant: str
    metadata: dict[str, Any] = field(default_factory=dict)
    measured_latency_ms: float | None = None
    api_prompt_tokens: int | None = None
    api_completion_tokens: int | None = None
    api_total_tokens: int | None = None
    api_cost_usd: float | None = None
    measured_energy_joule: float | None = None
    proxy_energy_joule: float | None = None


@dataclass(frozen=True)
class EvalRecord:
    sample_id: str
    split: str
    dataset: str
    method: str
    budget: Budget
    intervention: str
    answer: str
    gold_answer: str
    is_correct: bool
    logprob_original: float
    logprob_intervened: float
    confidence: float | None
    vem: float
    dense_vem: float | None
    reliance_retention: float | None
    delta_vem: float | None
    shortcut_persistence: bool
    latency_ms: float
    token_count: int
    cost: float
    energy_joule: float
    support_metric: str = "score_vem"
    answer_flip_support: bool | None = None
    measured_latency_ms: float | None = None
    api_prompt_tokens: int | None = None
    api_completion_tokens: int | None = None
    api_total_tokens: int | None = None
    api_cost_usd: float | None = None
    measured_energy_joule: float | None = None
    proxy_energy_joule: float | None = None
    peak_memory_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    sample_id: str
    split: str
    dataset: str
    gold_answer: str
    method: str
    selected_method: str
    selected_budget: Budget
    accepted: bool
    escalations: int
    answer: str
    is_correct: bool
    vem: float
    dense_vem: float | None
    reliance_retention: float | None
    cost: float
    latency_ms: float
    token_count: int
    reason: str
    eligible_interventions: tuple[str, ...] = field(default_factory=tuple)
    support_status: str = ""
    dense_latency_ms: float | None = None
    dense_token_count: int | None = None
    dense_cost: float | None = None
    speedup_vs_dense: float | None = None
    token_reduction: float | None = None
    retained_token_ratio: float | None = None
    online_cumulative_cost: float | None = None
    online_cumulative_latency_ms: float | None = None
    online_cumulative_energy_joule: float | None = None
    api_cost_usd: float | None = None
    measured_energy_joule: float | None = None
    proxy_energy_joule: float | None = None
    support_metric: str = "score_vem"


def normalize_answer(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace(".", "").replace(",", "").replace(";", "")
    value = " ".join(value.split())
    aliases = {
        "yeah": "yes",
        "y": "yes",
        "true": "yes",
        "nope": "no",
        "n": "no",
        "false": "no",
    }
    return aliases.get(value, value)


def canonicalize_answer(value: str, option_map: dict[str, str] | None = None, options: list[str] | None = None) -> str:
    """Map common multiple-choice outputs back to the canonical option text."""

    raw = str(value).strip()
    normalized = normalize_answer(raw)
    option_map = option_map or {}
    for letter, text in option_map.items():
        letter_norm = normalize_answer(letter)
        if normalized == letter_norm or normalized.startswith(f"{letter_norm}:") or normalized.startswith(f"{letter_norm} "):
            return str(text)
        letter_pattern = re.escape(letter_norm)
        if re.search(rf"\b(?:answer|option|choice)\s*(?:is\s*)?[:\-]?\s*{letter_pattern}\b", normalized):
            return str(text)
    for option in options or []:
        option_norm = normalize_answer(option)
        if normalized == option_norm or normalized.startswith(option_norm):
            return str(option)
    if options:
        contained = [option for option in options if normalize_answer(option) in normalized]
        if len(contained) == 1:
            return str(contained[0])
    lowered = f" {normalized} "
    if " yes " in lowered and " no " not in lowered:
        return "yes"
    if " no " in lowered and " yes " not in lowered:
        return "no"
    return raw


def budget_sort_key(value: Budget, dense_token_count: int | None = None) -> int:
    if value == "full":
        return dense_token_count if dense_token_count is not None else 10**9
    return int(value)
