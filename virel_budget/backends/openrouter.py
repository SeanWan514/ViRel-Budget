from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from virel_budget.backends.base import VLMBackend
from virel_budget.schema import Budget, Sample, ScoreResult, canonicalize_answer


class OpenRouterBackend(VLMBackend):
    """Black-box OpenRouter vision backend.

    This backend is intentionally answer-level only. It records API cost and
    latency but does not expose internal visual tokens or token pruning.
    """

    name = "openrouter"
    dense_token_count = 0

    def __init__(
        self,
        model_id: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        referer: str = "https://localhost",
        title: str = "ViRel-Budget",
        timeout_s: float = 120.0,
        max_tokens: int = 16,
        temperature: float = 0.0,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"OpenRouter backend requires {api_key_env} in the environment.")
        self.model_id = model_id
        self.api_key = api_key
        self.referer = referer
        self.title = title
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def score_options(
        self,
        sample: Sample,
        image_path: Path,
        method: str,
        budget: Budget,
        seed: int,
    ) -> ScoreResult:
        started = time.perf_counter()
        prompt = _build_prompt(sample)
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.referer,
                "X-Title": self.title,
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                response_obj = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}: {detail}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if "error" in response_obj:
            raise RuntimeError(f"OpenRouter request failed: {response_obj['error']}")
        choices = response_obj.get("choices") or []
        raw_answer = str(((choices[0] if choices else {}).get("message") or {}).get("content") or "").strip()
        answer = canonicalize_answer(
            raw_answer,
            option_map=sample.metadata.get("option_map") if isinstance(sample.metadata.get("option_map"), dict) else None,
            options=sample.options,
        )
        usage = response_obj.get("usage") or {}
        prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
        completion_tokens = _int_or_none(usage.get("completion_tokens"))
        total_tokens = _int_or_none(usage.get("total_tokens"))
        api_cost = _float_or_none(usage.get("cost") or usage.get("total_cost"))
        token_count = total_tokens or prompt_tokens or 0
        return ScoreResult(
            answer=answer,
            logprob=0.0,
            confidence=None,
            latency_ms=latency_ms,
            token_count=token_count,
            method=method,
            budget=budget,
            image_variant="computed",
            metadata={
                "model": "openrouter",
                "model_id": self.model_id,
                "raw_answer": raw_answer,
                "score_type": "answer_only",
                "usage": usage,
                "response_model": response_obj.get("model"),
            },
            measured_latency_ms=latency_ms,
            api_prompt_tokens=prompt_tokens,
            api_completion_tokens=completion_tokens,
            api_total_tokens=total_tokens,
            api_cost_usd=api_cost,
        )


def _build_prompt(sample: Sample) -> str:
    if sample.options:
        option_lines = "\n".join(f"- {option}" for option in sample.options)
        return (
            f"{sample.question}\n\n"
            "Choose the best answer from the options below. "
            "Return exactly one option text and nothing else.\n"
            f"{option_lines}"
        )
    return f"{sample.question}\n\nAnswer with a short phrase and no explanation."


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None
