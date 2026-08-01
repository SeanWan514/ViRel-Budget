from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from virel_budget.config import load_config  # noqa: E402
from virel_budget.pipeline import _load_samples, _make_backend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-call fidelity smoke for one extension cell.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    samples = _load_samples(config)
    sample = samples[0]
    method = str(config["pruning"]["methods"][0])
    budget = int(config["pruning"]["budget_schedule"][0])
    backend = _make_backend(config)
    calls = []
    try:
        for call_method, call_budget in (("dense", "full"), (method, budget)):
            started = time.time()
            result = backend.score_options(
                sample,
                sample.image_path,
                call_method,
                call_budget,
                int(config["seed"]),
            )
            calls.append(
                {
                    "method": call_method,
                    "budget": call_budget,
                    "answer": result.answer,
                    "token_count": result.token_count,
                    "latency_ms": result.measured_latency_ms or result.latency_ms,
                    "metadata": result.metadata,
                    "started_epoch": started,
                    "ended_epoch": time.time(),
                }
            )
    finally:
        backend.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "passed",
        "config": args.config,
        "sample_id": sample.sample_id,
        "calls": calls,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
