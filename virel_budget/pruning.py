from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from virel_budget.images import deterministic_seed, token_saliency
from virel_budget.schema import Budget


@dataclass(frozen=True)
class PruneSelection:
    method: str
    budget: Budget
    dense_token_count: int
    selected_indices: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.selected_indices)


def select_tokens(
    method: str,
    budget: Budget,
    dense_token_count: int,
    *,
    image_path: str | None = None,
    sample_id: str = "",
    seed: int = 0,
) -> PruneSelection:
    count = dense_token_count if budget == "full" else min(int(budget), dense_token_count)
    if count <= 0:
        raise ValueError("Budget must retain at least one token")
    if count == dense_token_count or method in {"dense", "full"}:
        indices = tuple(range(dense_token_count))
    elif method == "random":
        rng = random.Random(deterministic_seed(seed, sample_id, method, budget))
        indices = tuple(sorted(rng.sample(range(dense_token_count), count)))
    elif method == "center":
        indices = _center_indices(dense_token_count, count)
    elif method == "saliency":
        if image_path is None:
            indices = _center_indices(dense_token_count, count)
        else:
            scores = _saliency_for_count(image_path, dense_token_count)
            indices = tuple(sorted(np.argsort(scores)[-count:].astype(int).tolist()))
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    return PruneSelection(method=method, budget=budget, dense_token_count=dense_token_count, selected_indices=indices)


def apply_indices(sequence: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    return sequence[np.asarray(list(indices), dtype=np.int64)]


def _center_indices(dense_token_count: int, count: int) -> tuple[int, ...]:
    grid = int(round(math.sqrt(dense_token_count)))
    if grid * grid != dense_token_count:
        start = max((dense_token_count - count) // 2, 0)
        return tuple(range(start, start + count))
    cells = []
    center = (grid - 1) / 2.0
    for idx in range(dense_token_count):
        y, x = divmod(idx, grid)
        dist = (y - center) ** 2 + (x - center) ** 2
        cells.append((dist, idx))
    cells.sort()
    return tuple(sorted(idx for _, idx in cells[:count]))


def _saliency_for_count(image_path: str, dense_token_count: int) -> np.ndarray:
    grid = int(round(math.sqrt(dense_token_count)))
    if grid * grid == dense_token_count:
        return token_saliency(image_path, grid_size=grid)
    scores = token_saliency(image_path, grid_size=8)
    x_old = np.linspace(0.0, 1.0, len(scores))
    x_new = np.linspace(0.0, 1.0, dense_token_count)
    return np.interp(x_new, x_old, scores)
