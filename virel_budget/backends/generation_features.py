from __future__ import annotations

import time
from typing import Any


def summarize_generation_scores(
    output: Any,
    generated_ids: Any,
    torch: Any,
) -> tuple[dict[str, float | int | None], float]:
    """Return compact deployment-safe summaries without retaining full logits."""

    started = time.perf_counter()
    scores = list(output.get("scores") or [])
    token_ids = generated_ids.reshape(-1).tolist()
    usable = min(len(scores), len(token_ids))
    if usable == 0:
        return {
            "generated_token_count_scored": 0,
            "mean_token_logprob": None,
            "min_token_logprob": None,
            "final_token_logprob": None,
            "first_token_margin": None,
            "mean_predictive_entropy": None,
            "first_token_entropy": None,
        }, (time.perf_counter() - started) * 1000.0

    logprobs: list[float] = []
    entropies: list[float] = []
    first_margin: float | None = None
    for index in range(usable):
        logits = scores[index][0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        token_id = int(token_ids[index])
        logprobs.append(float(log_probs[token_id].item()))
        entropies.append(float((-(probs * log_probs).sum()).item()))
        if index == 0:
            top_two = torch.topk(logits, k=2).values
            first_margin = float((top_two[0] - top_two[1]).item())

    summary: dict[str, float | int | None] = {
        "generated_token_count_scored": usable,
        "mean_token_logprob": sum(logprobs) / usable,
        "min_token_logprob": min(logprobs),
        "final_token_logprob": logprobs[-1],
        "first_token_margin": first_margin,
        "mean_predictive_entropy": sum(entropies) / usable,
        "first_token_entropy": entropies[0],
    }
    return summary, (time.perf_counter() - started) * 1000.0
