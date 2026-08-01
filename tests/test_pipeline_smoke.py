import json
import shutil
import tempfile
import unittest
from pathlib import Path

from virel_budget.cli import make_smoke_data
from virel_budget.config import load_config
from virel_budget.pipeline import run_experiment


class PipelineSmokeTest(unittest.TestCase):
    def test_smoke_pipeline_writes_artifacts(self):
        root = Path(tempfile.mkdtemp(prefix="virel-smoke-"))
        try:
            (root / "configs").mkdir()
            make_smoke_data(root / "data" / "smoke")
            config = {
                "run_name": "tmp_smoke",
                "seed": 13,
                "backend": {"name": "deterministic", "profile": "offline-sanity"},
                "dataset": {
                    "name": "smoke",
                    "path": "data/smoke/samples.jsonl",
                    "split_field": "split",
                    "validation_split": "validation",
                    "test_split": "test",
                },
                "interventions": [
                    {"name": "gray", "type": "gray"},
                    {"name": "blur", "type": "blur", "radius": 8.0},
                ],
                "pruning": {"methods": ["random"], "budget_schedule": [16, 32, "full"], "dense_budget": "full"},
                "policy": {"tau_grid": [0.1], "rho_grid": [0.1], "dense_vem_min": 0.0, "prefer_rr": True},
                "cost": {
                    "base_ms": 12.0,
                    "per_token_ms": 0.28,
                    "dense_token_count": 64,
                    "dollar_per_1k_ms": 0.00002,
                    "energy_joule_per_ms": 0.004,
                },
                "outputs": {"dir": "results/tmp_smoke"},
            }
            config_path = root / "configs" / "tmp.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_config(config_path)
            result = run_experiment(loaded)
            self.assertGreater(result["n_records"], 0)
            self.assertGreater(result["n_decisions"], 0)
            for artifact in result["artifacts"].values():
                self.assertTrue(Path(artifact).exists(), artifact)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
