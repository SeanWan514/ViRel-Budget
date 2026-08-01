# Contributing

Contributions are welcome when they preserve the scientific boundaries of the release.

1. Open an issue describing the proposed change and affected claim.
2. Keep model/pruner adapters isolated under `virel_budget/backends/`.
3. Add tests for controller information flow, restart safety, and accounting changes.
4. Never commit model weights, source-dataset images, credentials, private annotation mappings, or unlicensed third-party code.
5. Run `python -m unittest discover -s tests -v` and `python scripts/audit/verify_release.py` before submitting.

New experimental results must disclose hardware, software revision, sampling seed, run count, measurement boundary, and carbon assumptions.
