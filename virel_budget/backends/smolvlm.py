from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from virel_budget.backends.base import VLMBackend
from virel_budget.schema import Budget, Sample, ScoreResult, canonicalize_answer


@dataclass(frozen=True)
class _SmolVLMRuntime:
    processor: Any
    model: Any
    torch: Any


class SmolVLMBackend(VLMBackend):
    """Dense small-VLM verification backend.

    This backend is intentionally answer-only and dense-only. It is used for
    GreenMM Pareto analysis: comparing a small dense VLM against a larger
    dense/pruned/adaptive LLaVA system under the same data and interventions.
    """

    name = "smolvlm"
    dense_token_count = 0

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 24,
        torch_dtype: str = "float16",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.torch_dtype = torch_dtype
        self.trust_remote_code = bool(trust_remote_code)
        self.runtime = self._load_runtime()

    def score_options(
        self,
        sample: Sample,
        image_path: Path,
        method: str,
        budget: Budget,
        seed: int,
    ) -> ScoreResult:
        if method != "dense" or budget != "full":
            raise ValueError("SmolVLMBackend is dense-only. Use method='dense' and budget='full'.")

        torch = self.runtime.torch
        start_epoch = time.time()
        started = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        prompt = _build_prompt(sample)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.runtime.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.runtime.processor(text=text, images=[image], return_tensors="pt")
        inputs = {key: _to_device(value, self.runtime.model.device) for key, value in inputs.items()}
        input_token_count = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

        with torch.inference_mode():
            generated = self.runtime.model.generate(
                **inputs,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
            )
        continuation = generated[:, input_token_count:] if input_token_count else generated
        raw_answer = self.runtime.processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
        latency_ms = (time.perf_counter() - started) * 1000.0
        end_epoch = time.time()
        answer = canonicalize_answer(
            raw_answer,
            option_map=sample.metadata.get("option_map") if isinstance(sample.metadata.get("option_map"), dict) else None,
            options=sample.options,
        )
        return ScoreResult(
            answer=answer,
            logprob=0.0,
            confidence=None,
            latency_ms=float(latency_ms),
            token_count=input_token_count,
            method=method,
            budget=budget,
            image_variant="computed",
            metadata={
                "model": "smolvlm",
                "model_id": self.model_id,
                "raw_answer": raw_answer,
                "score_type": "answer_only",
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "device": str(self.runtime.model.device),
                "verification_role": "dense_small_vlm_pareto_point",
            },
            measured_latency_ms=float(latency_ms),
        )

    def _load_runtime(self) -> _SmolVLMRuntime:
        try:
            import torch
            from transformers import AutoProcessor
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "SmolVLMBackend requires optional model dependencies. Install with: "
                "python -m pip install -e '.[research]'"
            ) from exc

        model_cls = None
        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq

                model_cls = AutoModelForVision2Seq
            except Exception as exc:
                raise RuntimeError(
                    "The installed transformers version does not expose an image-text generation auto-model. "
                    "Upgrade transformers before running the SmolVLM backend."
                ) from exc

        dtype = getattr(torch, self.torch_dtype, torch.float16)
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        model = model_cls.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = self.device
        model = model.to(device)
        model.eval()
        return _SmolVLMRuntime(processor=processor, model=model, torch=torch)


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


def _to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value
