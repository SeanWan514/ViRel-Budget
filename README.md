<h1 align="center">🌿 ViRel-Budget</h1>

<p align="center"><strong>Reliability-aware visual-token budgeting for greener multimodal inference</strong></p>

<p align="center">
  <img alt="Cases" src="https://img.shields.io/badge/evaluation-2%2C100_cases-2ea44f">
  <img alt="Models" src="https://img.shields.io/badge/LLaVA--1.5-7B_%7C_13B-6f42c1">
  <img alt="Pruners" src="https://img.shields.io/badge/pruning-FastV_%7C_SCOPE_%7C_Random-f59e0b">
  <img alt="Replication" src="https://img.shields.io/badge/replication-210_%C3%97_3-0ea5e9">
  <img alt="Energy" src="https://img.shields.io/badge/SCOPE_energy_saving-10.6%E2%80%9311.9%25-16a34a">
  <img alt="Tests" src="https://img.shields.io/badge/tests-29_passed-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-64748b">
</p>

ViRel-Budget studies a precise systems question: **how aggressively can a vision-language model prune visual tokens while preserving its dense answer and intervention-defined response behavior?** It turns that supervision into a frozen, deployment-safe controller, then evaluates the complete one-inference path using accuracy, unsafe acceptance, GPU energy, latency, memory, carbon estimates, and reliability-adjusted efficiency.

> [!IMPORTANT]
> ViRel-Budget measures **intervention-defined visual reliance**; it does not claim to certify universal semantic grounding. Intervention outputs supervise and evaluate the controller but are never available to it at deployment.

## ✨ Release snapshot

| Dimension | Public artifact |
|---|---|
| Evaluation | 1,200-case development pool + 900 sealed prospective cases |
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

The result separates **token efficiency** from **realized device-energy efficiency**: avoiding dense inference or retaining fewer tokens does not automatically reduce measured energy. See [`results/replication/summary`](results/replication/summary) and [`docs/results.md`](docs/results.md).

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
├── patches/               # deterministic Random pruning patch for official FastV
├── tests/                 # unit and restart/integrity tests
└── docs/                  # methodology, artifacts, measurement, limitations
```

Large historical checkpoints, model weights, image corpora, telemetry streams, exploratory plots, paper source, handoffs, and obsolete experiments are intentionally excluded from the public release.

## 🚀 Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
virel-budget --help
```

Model-backed reproduction additionally needs the official FastV/SCOPE environments and model weights; see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Saved-result inspection does not require a GPU.

## ♻️ Environmental accounting

The recoverable experimental program contains 66 runs, 19.17 GPU-hours, and **4.8198 measured GPU kWh**. Under the disclosed central assumptions (Iceland region, 0.03283 kgCO₂e/kWh, PUE 1.2), this corresponds to **0.1899 kgCO₂e**; the scenario range is 0.0795–0.5398 kgCO₂e. These are operational estimates, not lifecycle carbon measurements. CodeCarbon totals are reported separately because host-wide CPU/RAM estimates do not match the pod allocation.

See [`results/carbon`](results/carbon) for the run ledger and [`docs/measurement.md`](docs/measurement.md) for system boundaries.

## ⚠️ Scope and limitations

- The controller is model/pruner specific and tested on the stated workloads and hardware.
- SCOPE shows replicated device-energy savings; FastV does not on this implementation.
- Carbon is estimated from measured GPU energy plus disclosed regional assumptions.
- The dense SmolVLM-500M run is a model-scale audit, not a like-for-like alternative pruner evaluation.
- The blinded qualitative package is prepared, but final human adjudication is a paper-writing task rather than a completed GPU experiment.

See [`docs/limitations.md`](docs/limitations.md) before reusing the claims.

## 📜 Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). Code is released under the [MIT License](LICENSE). Dataset images and external model/pruning implementations retain their original licenses and are not redistributed here.
