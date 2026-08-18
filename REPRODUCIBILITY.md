# Reproducibility

## Lightweight verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/audit/verify_release.py
```

This validates the framework, restart behavior, energy attribution, manifest checksums, frozen result schemas, and release inventory without a GPU.

## Full model-backed execution

Full execution requires Linux, CUDA, one NVIDIA GPU with sufficient memory, separately obtained LLaVA-1.5 weights, official FastV and SCOPE implementations, and source-dataset images. Setup helpers are under `scripts/setup/`; frozen configs are under `configs/`.

The official third-party repositories are intentionally not vendored. Apply `patches/fastv_random_pruning.patch` to the pinned FastV checkout for the deterministic Random baseline. Confirm license compatibility before redistribution.

The public repository omits large historical Phase A/Phase B grids. It contains their frozen summaries and the compact raw 210×3 replication artifacts. Paths inside archived protocols reflect the frozen execution environment and are evidence, not portable defaults.

## Reproducing the publication summaries

- Prospective results: inspect `results/prospective/`.
- Replication: inspect or audit `results/replication/`.
- Carbon: inspect `results/carbon/experiment_run_ledger.csv` and its JSON summary.
- Pareto: inspect `results/pareto/`.
- Feature dependence: run `python scripts/analysis/run_feature_ablation.py` or inspect `results/feature_analysis/`.

Do not silently substitute hardware, model revisions, grid intensity, PUE, seeds, or intervention definitions. Record deviations as a new protocol.
