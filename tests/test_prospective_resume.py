from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.execution.run_frozen_controller import (
    _append_durable,
    _read_jsonl,
    _validate_executed_prefix,
    _validate_recomputed_plan,
)


class ProspectiveResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planned = [
            {
                "sample_id": "a",
                "dataset": "mmstar",
                "selected_budget": 64,
                "method": "fastv",
            },
            {
                "sample_id": "b",
                "dataset": "pope_random",
                "selected_budget": "dense",
                "method": "fastv",
            },
        ]

    def test_durable_append_round_trip_and_valid_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "executed.jsonl"
            row = {
                **self.planned[0],
                "answer": "x",
                "backend_invocation_count": 1,
            }
            _append_durable(path, row)
            loaded = _read_jsonl(path)
            self.assertEqual(loaded, [row])
            _validate_executed_prefix(loaded, self.planned)

    def test_rejects_divergent_or_duplicate_checkpoint(self) -> None:
        divergent = [
            {
                **self.planned[0],
                "selected_budget": 128,
                "backend_invocation_count": 1,
            }
        ]
        with self.assertRaises(ValueError):
            _validate_executed_prefix(divergent, self.planned)
        duplicate = [
            {**self.planned[0], "backend_invocation_count": 1},
            {
                **self.planned[1],
                "sample_id": "a",
                "backend_invocation_count": 1,
            },
        ]
        with self.assertRaises(ValueError):
            _validate_executed_prefix(duplicate, self.planned)

    def test_accepts_valid_out_of_order_checkpoint(self) -> None:
        executed = [
            {**self.planned[1], "backend_invocation_count": 1},
            {**self.planned[0], "backend_invocation_count": 1},
        ]
        _validate_executed_prefix(executed, self.planned)

    def test_rejects_non_single_invocation_checkpoint(self) -> None:
        row = {**self.planned[0], "backend_invocation_count": 2}
        with self.assertRaises(ValueError):
            _validate_executed_prefix([row], self.planned)

    def test_recomputed_plan_ignores_only_runtime_latency(self) -> None:
        existing = [
            {
                **self.planned[0],
                "probabilities": {"64": 0.9},
                "controller_latency_ms": 0.01,
            }
        ]
        recomputed = [
            {
                **self.planned[0],
                "probabilities": {"64": 0.9},
                "controller_latency_ms": 0.03,
            }
        ]
        _validate_recomputed_plan(existing, recomputed)
        recomputed[0]["selected_budget"] = 128
        with self.assertRaises(ValueError):
            _validate_recomputed_plan(existing, recomputed)


if __name__ == "__main__":
    unittest.main()
