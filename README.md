<h1 align="center">🌿 ViRel-Budget</h1>

<p align="center">
  <strong>Reliability-constrained visual-token budgeting for green vision-language inference</strong><br>
  Accepted for an oral presentation at the ACM MM 2026 GreenMM Workshop
</p>

<p align="center">
  <img alt="Cases" src="https://img.shields.io/badge/evaluation-2%2C100_cases-2ea44f">
  <img alt="Models" src="https://img.shields.io/badge/LLaVA--1.5-7B_%7C_13B-6f42c1">
  <img alt="Pruners" src="https://img.shields.io/badge/pruning-FastV_%7C_SCOPE_%7C_Random-f59e0b">
  <img alt="Replication" src="https://img.shields.io/badge/replication-210_%C3%97_3-0ea5e9">
  <img alt="Energy" src="https://img.shields.io/badge/SCOPE_energy_saving-10.6%E2%80%9311.9%25-16a34a">
  <img alt="Status" src="https://img.shields.io/badge/ACM_MM_GreenMM_2026-Accepted_Oral-0B7A75">
  <img alt="Tests" src="https://img.shields.io/badge/tests-29_passed-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-64748b">
</p>

**ViRel-Budget** asks a concrete systems question: **how aggressively can a vision-language model prune visual tokens while preserving its dense answer and intervention-defined response behavior?** It turns development-only supervision into a frozen controller, executes one selected action per query, and evaluates the complete path using accuracy, unsafe acceptance, GPU energy, latency, memory, carbon estimates, and reliability-adjusted efficiency.

GitHub repository: https://github.com/SeanWan514/ViRel_Budget

> [!IMPORTANT]
> ViRel-Budget measures **intervention-defined visual reliance**; it does not claim to certify universal semantic grounding. Intervention outputs supervise and evaluate the controller but are never available to it at deployment.

## ✨ Release snapshot

| Dimension | Public artifact |
|---|---|
| Evaluation | 1,200-query development pool + 900 sealed prospective queries |
| Backends | LLaVA-1.5-7B and LLaVA-1.5-13B; dense SmolVLM-500M scale audit |
| Pruning | FastV, SCOPE, and deterministic Random control |
| Deployment | Frozen metadata controller; one backend invocation per query |
| Replication | Three group-disjoint, stratified 210-case draws |
| Hardware | One NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Environmental accounting | NVIDIA device telemetry, prospective CodeCarbon, disclosed PUE and grid-intensity scenarios |

## 📊 Main finding

Across the 630-query replication population, SCOPE was the only pruner with positive energy reduction on **every draw**:

| Backend | System | GPU energy vs. dense | 95% CI | Unsafe acceptance | Dense avoidance |
|---|---:|---:|---:|---:|---:|
| 7B | SCOPE | **11.87% lower** | [7.19%, 16.30%] | 0.63% | 33.97% |
| 13B | SCOPE | **10.62% lower** | [7.48%, 13.46%] | 1.27% | 50.00% |
| 7B | FastV | 3.11% higher | [−6.84%, 0.75%] reduction | 0.79% | 61.27% |
| 13B | FastV | 1.35% higher | [−4.13%, 1.39%] reduction | 2.54% | 77.78% |
| 7B | Random | 5.96% higher | [−11.40%, −0.95%] reduction | 2.54% | 55.56% |
| 13B | Random | 3.92% higher | [−7.40%, −0.55%] reduction | 1.43% | 52.22% |

The result separates **token efficiency** from **realized device-energy efficiency**: avoiding dense inference or retaining fewer tokens does not automatically reduce measured energy. The unsafe rates above belong to the separate 630-query replication population; prospective reliability results are reported independently under [`results/prospective`](results/prospective).

See [`results/replication/summary`](results/replication/summary) and [`docs/results.md`](docs/results.md).

## 🧠 Method in one view

1. Construct strict labels that separately record task correctness, dense-answer agreement, intervention-response preservation, and their conjunction.
2. Learn a low-capacity controller from the 1,200-case development pool using deployment-available metadata only.
3. Freeze the controller, risk threshold, action space, fallback, manifests, and measurement boundary.
4. Execute exactly one selected action on each of 900 group-isolated prospective cases.
5. Reveal hidden evaluation labels only after execution is complete.
6. Replicate the principal systems comparison on three independently sampled, group-disjoint 210-case workloads.

Full details: [`docs/methodology.md`](docs/methodology.md), [`docs/datasets.md`](docs/datasets.md), and [`docs/measurement.md`](docs/measurement.md).

## 🗂️ Repository map

```text
ViRel-Budget/
├── virel_budget/          # installable framework and backend adapters
├── scripts/               # data, execution, measurement, analysis, and audits
├── configs/               # frozen development, prospective, replication protocols
├── data/                  # manifests and lightweight test fixtures (not full images)
├── results/               # frozen publication-facing summaries and compact raw replication data
│   └── feature_analysis/  # development-only grouped feature analysis
├── patches/               # deterministic Random pruning patch for official FastV
├── tests/                 # unit and restart/integrity tests
└── docs/                  # methodology, artifacts, measurement, limitations
```

Large historical checkpoints, model weights, source images, telemetry streams, exploratory plots, paper source, handoffs, and obsolete experiments are intentionally excluded from the public release.

## 🚀 Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
virel-budget --help
```

Model-backed reproduction additionally needs the official FastV/SCOPE environments and model weights; see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Saved-result inspection does not require a GPU.

## 🔎 What drives the controller?

A development-only grouped analysis compares question structure, task/source indicators, and semantic-keyword indicators across all six controllers. Task/source indicators have the largest mean absolute standardized coefficient in all 24 budget-specific regressions and are the strongest single feature family for dense avoidance in four of six controllers. Removing them reduces dense avoidance in five controllers by 18.33–32.08 percentage points; 13B FastV is the exception.

This result shows substantial reliance on task identity and does not establish transfer to unseen task families. No image-derived metadata are used. The complete verified outputs are under [`results/feature_analysis`](results/feature_analysis).

## ♻️ Environmental accounting

The recoverable experimental program contains 66 runs, 19.17 GPU-hours, and **4.8198 measured GPU kWh**. Under the disclosed central assumptions (Iceland region, 0.03283 kgCO₂e/kWh, PUE 1.2), this corresponds to **0.1899 kgCO₂e**; the scenario range is 0.0795–0.5398 kgCO₂e. These are operational estimates, not lifecycle carbon measurements. CodeCarbon totals are reported separately because host-wide CPU/RAM estimates do not match the pod allocation.

See [`results/carbon`](results/carbon) for the run ledger and [`docs/measurement.md`](docs/measurement.md) for system boundaries.

## ⚠️ Scope and limitations

- The controller is model/pruner specific and tested on the stated workloads and hardware.
- SCOPE shows replicated device-energy savings; FastV does not on this implementation.
- Carbon is estimated from measured GPU energy plus disclosed regional assumptions.
- The dense SmolVLM-500M run is a model-scale audit, not a like-for-like alternative pruner evaluation.
- Intervention-defined safety measures behavioral preservation, not whether a model uses the correct visual evidence.

See [`docs/limitations.md`](docs/limitations.md) before reusing the claims.

## 📜 Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). Code is released under the [Apache License 2.0](LICENSE). Source images and external model/pruning implementations retain their original licenses and are not redistributed here. Dataset manifests are provided for research reproducibility and remain subject to their source datasets' terms.

## 🙏 Acknowledgments

Sean Wan and Shilin Ou gratefully acknowledge support from the Summer Research Scholars Program at Duke Kunshan University, under the supervision of Prof. Luyao Zhang, and from the Duke Kunshan University Library grant supporting the Open Data Contest project.
