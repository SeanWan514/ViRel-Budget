from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from virel_budget.backends.base import VLMBackend
from virel_budget.backends.generation_features import summarize_generation_scores
from virel_budget.schema import Budget, Sample, ScoreResult, canonicalize_answer


@dataclass
class _ScopeRuntime:
    tokenizer: Any
    model: Any
    image_processor: Any
    conv_templates: Any
    separator_style: Any
    constants: dict[str, Any]
    process_images: Any
    tokenizer_image_token: Any
    stopping_criteria_cls: Any
    scope_fn: Any
    torch: Any


class ScopeLlavaBackend(VLMBackend):
    """Official SCOPE/LLaVA backend loaded from the SCOPE repository.

    Dense calls run before SCOPE is enabled. The `scope` method then applies
    SCOPE's published saliency-coverage token pruning wrapper to LLaVA-1.5 and
    records the actual retained visual-token budget.
    """

    name = "scope_llava"

    def __init__(
        self,
        model_path: str = "liuhaotian/llava-v1.5-7b",
        scope_repo: str = "/workspace/virel_external/SCOPE",
        device: str = "cuda",
        max_new_tokens: int = 24,
        conv_mode: str | None = None,
        image_aspect_ratio: str = "pad",
        load_8bit: bool = False,
        load_4bit: bool = False,
        instrument_features: bool = False,
    ) -> None:
        self.model_path = model_path
        self.scope_repo = Path(scope_repo)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.conv_mode_override = conv_mode
        self.image_aspect_ratio = image_aspect_ratio
        self.load_8bit = bool(load_8bit)
        self.load_4bit = bool(load_4bit)
        self.instrument_features = bool(instrument_features)
        self.scope_enabled = False
        self.current_scope_budget: int | None = None
        self.runtime = self._load_runtime()
        self.dense_token_count = int(getattr(self.runtime.model.get_vision_tower(), "num_patches", 576))
        self.conv_mode = self._infer_conv_mode()
        self.scope_commit = _git_commit(self.scope_repo)

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

        token_count = self._configure_scope(method, budget)
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
                output_attentions=False,
                output_scores=self.instrument_features,
                return_dict_in_generate=True,
            )
        sequence = output["sequences"][0]
        generated_ids, decode_mode = _generated_ids(sequence, input_ids[0], torch)
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
                "model": "scope_llava",
                "score_type": "answer_only",
                "model_path": self.model_path,
                "scope_repo": str(self.scope_repo),
                "scope_commit": self.scope_commit,
                "raw_answer": raw_answer,
                "conv_mode": self.conv_mode,
                "image_token_start": image_token_start,
                "image_token_length": self.dense_token_count,
                "scope_token_num": token_count if method == "scope" else None,
                "scope_alpha": os.environ.get("ALPHA", "1.0") if method == "scope" else None,
                "scope_combined": os.environ.get("COMBINED", "multi") if method == "scope" else None,
                "official_scope_mechanism": method == "scope",
                "input_token_length": int(input_ids.shape[1]),
                "sequence_token_length": int(sequence.shape[0]),
                "generated_token_length": int(generated_ids.shape[0]),
                "decode_mode": decode_mode,
                "instrument_features": self.instrument_features,
                "generation_features": generation_features,
                "feature_overhead_ms": feature_overhead_ms,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
            },
            measured_latency_ms=float(latency_ms),
        )

    def _load_runtime(self) -> _ScopeRuntime:
        scope_llava = self.scope_repo / "LLaVA"
        if not self.scope_repo.exists() or not scope_llava.exists():
            raise RuntimeError(
                "SCOPE repository is missing expected source folders. "
                f"Expected {self.scope_repo} and {scope_llava}."
            )
        for path in [str(self.scope_repo), str(scope_llava)]:
            if path not in sys.path:
                sys.path.insert(0, path)

        _install_llava_namespace(scope_llava)
        _install_scope_import_stubs()
        os.environ["BASELINE"] = "DENSE"

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
        from scope import SCOPE

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, _context_len = load_pretrained_model(
            self.model_path,
            None,
            model_name,
            load_8bit=self.load_8bit,
            load_4bit=self.load_4bit,
            device=self.device,
        )
        if self.device == "cuda":
            model.get_vision_tower().to(device=getattr(model, "device", self.device), dtype=torch.float16)
        model.eval()
        return _ScopeRuntime(
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
            scope_fn=SCOPE,
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

    def _configure_scope(self, method: str, budget: Budget) -> int:
        if method in {"dense", "full"}:
            if self.scope_enabled:
                raise RuntimeError(
                    "Dense LLaVA evaluation after SCOPE has been enabled is not supported in one process. "
                    "Run dense references before scope pruning, as the ViRel pipeline does."
                )
            return int(self.dense_token_count)
        if method != "scope":
            raise ValueError(f"ScopeLlavaBackend supports methods dense and scope, got {method}")

        token_count = self._token_count(method, budget)
        if not self.scope_enabled:
            self.runtime.model = self.runtime.scope_fn(self.runtime.model, num_token=token_count)
            self.scope_enabled = True
        elif self.current_scope_budget != token_count:
            self._set_scope_budget(token_count)
        self.current_scope_budget = token_count
        return token_count

    def _set_scope_budget(self, token_count: int) -> None:
        vision_model = self.runtime.model.model.vision_tower.vision_tower
        if not hasattr(vision_model, "_info"):
            self.runtime.model = self.runtime.scope_fn(self.runtime.model, num_token=token_count)
            return
        vision_model._info["dominant"] = int(token_count)
        vision_model._info["contextual"] = 0
        for module in vision_model.modules():
            if hasattr(module, "_info"):
                module._info = vision_model._info

    def _token_count(self, method: str, budget: Budget) -> int:
        if method != "scope" or budget == "full":
            return int(self.dense_token_count)
        return max(1, min(int(budget), int(self.dense_token_count)))

    def close(self) -> None:
        try:
            self.runtime.torch.cuda.empty_cache()
        except Exception:
            pass


def _install_scope_import_stubs() -> None:
    """SCOPE imports two unused author-local helper modules at import time."""

    for name in ["submodular_function", "submodular_optimizer"]:
        sys.modules.setdefault(name, types.ModuleType(name))


def _install_llava_namespace(scope_llava: Path) -> None:
    """Bypass SCOPE's broken top-level llava/__init__.py import."""

    if "llava" in sys.modules:
        return
    package = types.ModuleType("llava")
    package.__path__ = [str(scope_llava / "llava")]
    package.__file__ = str(scope_llava / "llava" / "__init__.py")
    sys.modules["llava"] = package


def _generated_ids(sequence: Any, input_ids: Any, torch: Any) -> tuple[Any, str]:
    if sequence.shape[0] >= input_ids.shape[0] and torch.equal(sequence[: input_ids.shape[0]], input_ids):
        return sequence[input_ids.shape[0] :], "prompt_prefix_sliced"
    return sequence, "sequence_only"


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
