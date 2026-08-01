from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from virel_budget.backends.base import VLMBackend
from virel_budget.backends.generation_features import summarize_generation_scores
from virel_budget.images import deterministic_seed
from virel_budget.schema import Budget, Sample, ScoreResult, canonicalize_answer


@dataclass(frozen=True)
class _FastVRuntime:
    tokenizer: Any
    model: Any
    image_processor: Any
    conv_templates: Any
    separator_style: Any
    constants: dict[str, Any]
    process_images: Any
    tokenizer_image_token: Any
    stopping_criteria_cls: Any
    torch: Any


class FastVLlavaBackend(VLMBackend):
    """Official FastV/LLaVA backend loaded from the FastV repository.

    The `fastv` method uses FastV's modified Llama attention-mask logic:
    after the configured aggregation layer K, visual tokens outside the
    top-attention rank are masked in deeper layers. Dense uses the same
    LLaVA model with FastV disabled.
    """

    name = "fastv_llava"

    def __init__(
        self,
        model_path: str = "liuhaotian/llava-v1.5-7b",
        fastv_repo: str = "/workspace/virel_external/FastV",
        device: str = "cuda",
        max_new_tokens: int = 24,
        conv_mode: str | None = None,
        image_aspect_ratio: str = "pad",
        fastv_agg_layer: int = 2,
        load_8bit: bool = False,
        load_4bit: bool = False,
        instrument_features: bool = False,
    ) -> None:
        self.model_path = model_path
        self.fastv_repo = Path(fastv_repo)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.conv_mode_override = conv_mode
        self.image_aspect_ratio = image_aspect_ratio
        self.fastv_agg_layer = int(fastv_agg_layer)
        self.load_8bit = bool(load_8bit)
        self.load_4bit = bool(load_4bit)
        self.instrument_features = bool(instrument_features)
        self.runtime = self._load_runtime()
        self.dense_token_count = int(getattr(self.runtime.model.get_vision_tower(), "num_patches", 576))
        self.conv_mode = self._infer_conv_mode()
        self.fastv_commit = _git_commit(self.fastv_repo)

    def score_options(
        self,
        sample: Sample,
        image_path: Path,
        method: str,
        budget: Budget,
        seed: int,
    ) -> ScoreResult:
        torch = self.runtime.torch
        start_epoch = time.time()
        started = time.perf_counter()
        prompt = self._build_question(sample)
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.runtime.process_images([image], self.runtime.image_processor, self)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.runtime.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.runtime.model.device, dtype=torch.float16)

        full_prompt = self._conversation_prompt(prompt)
        input_ids = self.runtime.tokenizer_image_token(
            full_prompt,
            self.runtime.tokenizer,
            self.runtime.constants["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        ).unsqueeze(0).to(self.runtime.model.device)
        image_token_positions = (input_ids[0] == self.runtime.constants["IMAGE_TOKEN_INDEX"]).nonzero(as_tuple=False)
        if len(image_token_positions) != 1:
            raise RuntimeError(f"Expected exactly one image token, found {len(image_token_positions)}")
        image_token_start = int(image_token_positions[0].item())

        random_seed = deterministic_seed(seed, sample.sample_id, str(image_path), method, budget)
        self._configure_fastv(method, budget, image_token_start, random_seed)
        stop_str = self._stop_string()
        stopping_criteria = self.runtime.stopping_criteria_cls([stop_str], self.runtime.tokenizer, input_ids)

        with torch.inference_mode():
            output = self.runtime.model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=None,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_attentions=(method == "fastv"),
                output_scores=self.instrument_features,
                return_dict_in_generate=True,
            )
        generated_ids = output["sequences"][0, input_ids.shape[1] :]
        generation_features, feature_overhead_ms = (
            summarize_generation_scores(output, generated_ids, torch)
            if self.instrument_features
            else ({}, 0.0)
        )
        raw_answer = self.runtime.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip().replace("</s>", "").strip()
        end_epoch = time.time()
        answer = canonicalize_answer(
            raw_answer,
            option_map=sample.metadata.get("option_map") if isinstance(sample.metadata.get("option_map"), dict) else None,
            options=sample.options,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        token_count = self._token_count(method, budget)
        return ScoreResult(
            answer=answer,
            logprob=0.0,
            confidence=None,
            latency_ms=float(latency_ms),
            token_count=token_count,
            method=method,
            budget=budget,
            image_variant="computed",
            metadata={
                "model": "fastv_llava",
                "score_type": "answer_only",
                "model_path": self.model_path,
                "fastv_repo": str(self.fastv_repo),
                "fastv_commit": self.fastv_commit,
                "raw_answer": raw_answer,
                "conv_mode": self.conv_mode,
                "image_token_start": image_token_start,
                "image_token_length": self.dense_token_count,
                "fastv_agg_layer": self.fastv_agg_layer if method == "fastv" else None,
                "fastv_attention_rank": token_count if method == "fastv" else None,
                "official_fastv_mechanism": method == "fastv",
                "random_pruning_control": method == "random",
                "random_pruning_seed": random_seed if method == "random" else None,
                "instrument_features": self.instrument_features,
                "generation_features": generation_features,
                "feature_overhead_ms": feature_overhead_ms,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
            },
            measured_latency_ms=float(latency_ms),
        )

    def _load_runtime(self) -> _FastVRuntime:
        fastv_transformers = self.fastv_repo / "src" / "transformers" / "src"
        fastv_llava = self.fastv_repo / "src" / "LLaVA"
        if not fastv_transformers.exists() or not fastv_llava.exists():
            raise RuntimeError(
                "FastV repository is missing expected source folders. "
                f"Expected {fastv_transformers} and {fastv_llava}."
            )
        for path in [str(fastv_transformers), str(fastv_llava)]:
            if path not in sys.path:
                sys.path.insert(0, path)

        import torch
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, _context_len = load_pretrained_model(
            self.model_path,
            None,
            model_name,
            self.load_8bit,
            self.load_4bit,
            device=self.device,
        )
        model.eval()
        return _FastVRuntime(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            conv_templates=conv_templates,
            separator_style=SeparatorStyle,
            constants={
                "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
                "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
                "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
                "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            },
            process_images=process_images,
            tokenizer_image_token=tokenizer_image_token,
            stopping_criteria_cls=KeywordsStoppingCriteria,
            torch=torch,
        )

    def _infer_conv_mode(self) -> str:
        if self.conv_mode_override:
            return self.conv_mode_override
        model_name = self.model_path.lower()
        if "llama-2" in model_name:
            return "llava_llama_2"
        if "v1" in model_name:
            return "llava_v1"
        if "mpt" in model_name:
            return "mpt"
        return "llava_v0"

    def _build_question(self, sample: Sample) -> str:
        if sample.options:
            options = "\n".join(f"- {option}" for option in sample.options)
            return (
                f"{sample.question}\n\n"
                "Choose the best answer from the options below. "
                "Return exactly one option text and nothing else.\n"
                f"{options}"
            )
        return f"{sample.question}\n\nAnswer with a short phrase and no explanation."

    def _conversation_prompt(self, question: str) -> str:
        model = self.runtime.model
        conv = self.runtime.conv_templates[self.conv_mode].copy()
        if model.config.mm_use_im_start_end:
            question = (
                self.runtime.constants["DEFAULT_IM_START_TOKEN"]
                + self.runtime.constants["DEFAULT_IMAGE_TOKEN"]
                + self.runtime.constants["DEFAULT_IM_END_TOKEN"]
                + "\n"
                + question
            )
        else:
            question = self.runtime.constants["DEFAULT_IMAGE_TOKEN"] + "\n" + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _stop_string(self) -> str:
        conv = self.runtime.conv_templates[self.conv_mode]
        return conv.sep if conv.sep_style != self.runtime.separator_style.TWO else conv.sep2

    def _configure_fastv(
        self,
        method: str,
        budget: Budget,
        image_token_start: int,
        random_seed: int = 0,
    ) -> None:
        model = self.runtime.model
        if method in {"fastv", "random"}:
            if method == "random" and not hasattr(model.model, "fast_v_selection_mode"):
                raise RuntimeError(
                    "Random pruning requires patches/fastv_random_pruning.patch "
                    "applied to the pinned FastV repository."
                )
            attention_rank = self._token_count(method, budget)
            model.config.use_fast_v = True
            model.config.fast_v_sys_length = image_token_start
            model.config.fast_v_image_token_length = self.dense_token_count
            model.config.fast_v_attention_rank = attention_rank
            model.config.fast_v_agg_layer = self.fastv_agg_layer
            model.config.fast_v_selection_mode = "random" if method == "random" else "attention"
            model.config.fast_v_random_seed = int(random_seed)
        elif method in {"dense", "full"}:
            model.config.use_fast_v = False
            model.config.fast_v_sys_length = image_token_start
            model.config.fast_v_image_token_length = self.dense_token_count
            model.config.fast_v_attention_rank = self.dense_token_count
            model.config.fast_v_agg_layer = self.fastv_agg_layer
            model.config.fast_v_selection_mode = "attention"
            model.config.fast_v_random_seed = 0
        else:
            raise ValueError(f"FastVLlavaBackend supports methods dense, fastv, and random, got {method}")
        model.model.reset_fastv()

    def _token_count(self, method: str, budget: Budget) -> int:
        if method not in {"fastv", "random"} or budget == "full":
            return int(self.dense_token_count)
        return max(1, min(int(budget), int(self.dense_token_count)))

    def close(self) -> None:
        try:
            self.runtime.torch.cuda.empty_cache()
        except Exception:
            pass


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
