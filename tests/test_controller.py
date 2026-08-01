from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from virel_budget.controller import DeploymentFeatures
from virel_budget.datasets.deployment_jsonl import load_deployment_samples
from virel_budget.frozen_controller import FrozenBudgetController, canonical_json_bytes


def _valid_record() -> dict:
    return {
        "sample_id": "sample",
        "method": "scope",
        "executed_budget": 64,
        "dataset": "mmstar",
        "question_word_count": 8,
        "option_count": 4,
        "answer": "red",
        "generated_token_count": 1,
        "mean_token_logprob": -0.2,
        "min_token_logprob": -0.2,
        "final_token_logprob": -0.2,
        "first_token_margin": 2.0,
        "mean_predictive_entropy": 0.8,
        "first_token_entropy": 0.8,
        "feature_overhead_ms": 0.1,
        "model_latency_ms": 100.0,
        "token_count": 64,
    }


class DeploymentFeaturesTest(unittest.TestCase):
    def test_accepts_only_deployment_fields(self) -> None:
        features = DeploymentFeatures.from_pilot_record(_valid_record())
        self.assertEqual(features.executed_budget, 64)

    def test_rejects_gold_and_dense_labels(self) -> None:
        for forbidden in ("gold_answer", "dense_answer", "reference_safe"):
            record = _valid_record()
            record[forbidden] = "forbidden"
            with self.assertRaises(ValueError):
                DeploymentFeatures.from_pilot_record(record)

    def test_frozen_controller_rejects_gold_input_and_selects_once(self) -> None:
        artifact = {
            "feature_names": [
                "question_word_count",
                "question_character_count",
                "option_count",
                "multiple_choice",
                "dataset=mmstar",
                "dataset=pope_adversarial",
                "dataset=pope_popular",
                "dataset=pope_random",
                "dataset=visual_counterfact_color",
                "dataset=visual_counterfact_size",
                "keyword=color",
                "keyword=size",
                "keyword=count",
                "keyword=existence",
                "keyword=spatial",
                "keyword=text",
                "keyword=emotion",
            ],
            "selection_rule": "smallest_budget_above_global_threshold_else_dense",
            "global_threshold": 0.5,
            "models": {
                str(budget): {
                    "weights": [10.0] + [0.0] * 17,
                    "means": [0.0] * 17,
                    "scales": [1.0] * 17,
                }
                for budget in (64, 128, 256, 432)
            },
        }
        controller = FrozenBudgetController(artifact)
        record = {"sample_id": "x", "dataset": "mmstar", "question": "What?", "options": []}
        self.assertEqual(controller.decide(record).selected_budget, 64)
        with self.assertRaises(ValueError):
            controller.decide({**record, "gold_answer": "yes"})

    def test_deployment_loader_rejects_answer_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_bytes(
                canonical_json_bytes(
                    {
                        "sample_id": "x",
                        "dataset": "mmstar",
                        "image_path": "/tmp/x.jpg",
                        "question": "What?",
                        "options": [],
                        "answer": "forbidden",
                    }
                )
            )
            with self.assertRaises(ValueError):
                load_deployment_samples(path)
