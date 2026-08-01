# Methodology

## Scientific target

ViRel-Budget does not collapse correctness and visual reliance into one opaque label. For each sample, model, pruning method, and budget it records:

- task correctness;
- agreement with the dense original answer;
- preservation of the dense intervention-response trajectory; and
- strict safety, the conjunction required by the deployment study.

The interventions are targeted counterfactual replacement, irrelevant-image replacement, blur, and the historical constant-gray null. They are supervision/evaluation instruments, not proof of semantic grounding.

## Development and prospective protocol

The original 1,200 cases are treated as development data because the legacy validation/test partition contained related images and repeated questions. Group-aware development procedures are used to fit a low-capacity, cost-sensitive controller. The controller observes only metadata available before inference; it cannot access dense answers, intervention answers, hidden labels, or offline oracle budgets.

Before prospective execution, the feature schema, model, risk threshold, action set, fallback, and manifests are frozen. Each deployment query makes exactly one backend call. Hidden labels from the 900 added cases are revealed only after all six model/pruner executions complete.

## Comparators

The study compares dense inference, fixed visual-token budgets, FastV, SCOPE, deterministic Random pruning, frozen ViRel controllers, and the offline oracle upper bound where appropriate. Random uses a fixed sample-specific seed and the same token counts, isolating informed token selection from pruning quantity.

## Replication gate

The principal GreenMM result uses three stratified, group-disjoint 210-case draws. A model/pruner cell is called strongly green only if:

1. unsafe acceptance remains below the frozen risk criterion;
2. every draw has positive paired energy reduction relative to its matched dense calls; and
3. the pooled paired energy-reduction confidence interval has a positive lower bound.

Only 7B SCOPE and 13B SCOPE satisfy all three conditions.
