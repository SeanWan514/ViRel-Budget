# Energy, carbon, and cost measurement

## Primary device measurement

NVIDIA power telemetry is integrated over the backend call window. Warm-up, CUDA synchronization, fixed execution settings, paired queries, and counterbalanced order reduce measurement bias. Wrapper/model-load energy is retained separately from per-call attribution.

## CodeCarbon

CodeCarbon prospectively instruments the replication and other retained runs. Its energy is a secondary cross-check: it detected host-wide CPU/RAM resources larger than the assigned pod, so those CPU/RAM values are not treated as direct pod measurements.

## Carbon assumptions

- Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Region: RunPod EUR-IS-1; CodeCarbon geolocation indicated Iceland.
- Central grid intensity: 0.03283 kgCO₂e/kWh.
- Central PUE: 1.2.
- Sensitivity: grid intensity 0.015–0.08 kgCO₂e/kWh and PUE 1.1–1.4.
- Cloud cost: USD 2.00/GPU-hour runtime proxy, not a billing statement.

The primary whole-program footprint is location/PUE-adjusted measured GPU energy. Pod idle time outside wrappers, embodied carbon, networking, and storage are excluded. Completed historical runs use direct stored NVIDIA measurements where available; CodeCarbon is never described as having retroactively measured them.

## Functional units

Reports include joules/query, latency/query, throughput, peak GPU memory, correct and intervention-supported answers/kWh, estimated gCO₂e/1,000 queries, dense avoidance at the risk limit, and pairwise Pareto status.
