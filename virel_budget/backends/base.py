from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from virel_budget.schema import Budget, Sample, ScoreResult


class VLMBackend(ABC):
    name: str
    dense_token_count: int

    @abstractmethod
    def score_options(
        self,
        sample: Sample,
        image_path: Path,
        method: str,
        budget: Budget,
        seed: int,
    ) -> ScoreResult:
        """Return the selected answer and its log-probability for one image variant."""

    def close(self) -> None:
        return None
