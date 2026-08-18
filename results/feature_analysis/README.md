# Development-only feature analysis

This directory contains the grouped feature analysis added for the camera-ready revision. It evaluates seven frozen feature configurations across all six model--method controllers using the original five group-aware folds and the same one-sided 95% Wilson 5% risk rule.

The 17 deployment-available features comprise four question-structure features, six task/source indicators, and seven semantic-keyword indicators. No image-derived metadata, model outputs, gold answers, or intervention results are used.

## Principal result

Task/source indicators have the largest mean absolute standardized coefficient in all 24 budget-specific regressions and are the strongest single feature family for dense avoidance in four of six controllers. Removing them reduces dense avoidance in five controllers by 18.33--32.08 percentage points; 13B FastV is the exception. These development-set results show substantial reliance on task identity and do not demonstrate transfer to unseen task families.

## Reproduce

From the repository root:

```bash
python scripts/analysis/run_feature_ablation.py
```

The script consumes the numeric pre-inference feature matrix and frozen fold assignments in `development_features_and_folds.csv`, together with the published development safety labels. It does not require source images, prospective labels, a GPU, or new VLM inference.

Key outputs:

- `feature_ablation_by_controller.csv`: policy-level results for all 42 configurations;
- `feature_group_effects.csv`: removal and single-family comparisons;
- `feature_group_coefficient_audit.csv`: standardized coefficient summaries;
- `paper_feature_ablation_table.csv`: compact paper-facing summary;
- `feature_ablation_run_manifest.json`: protocol and input checks.

The original full-feature policies reproduce the archived controller behavior exactly at the policy level.
