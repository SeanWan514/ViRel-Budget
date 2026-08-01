# Results and interpretation

## Prospective validity

All six frozen controllers completed the sealed 900-case protocol with one backend invocation per query. Two of six model/pruner cells met both the prospective risk and measured-energy criteria. Results are reported by model and pruner rather than hidden in a pooled average.

## Replicated systems result

On three group-disjoint 210-case draws, SCOPE reduced call-window GPU energy by 11.87% for 7B and 10.62% for 13B relative to matched dense inference. Both confidence intervals exclude zero, and all six model-by-draw SCOPE reductions are positive.

FastV avoided dense inference more often, but its measured energy was statistically indistinguishable from—or slightly worse than—dense. Random increased energy at both scales. This is a useful negative result: token count, dense avoidance, latency, and energy are distinct systems outcomes.

## Model-scale frontier

The 13B backend improves dense accuracy relative to 7B but uses more energy and memory. The 500M audit has far lower memory but is slower on the tested implementation and does not dominate 7B. The repository therefore reports pairwise Pareto frontiers rather than declaring one model size universally preferable.

## Claim boundary

The supported claim is that ViRel-Budget supplies a deployment-safe reliability/efficiency framework and that SCOPE realizes a replicated device-energy benefit on the tested stack. It is not supported to claim that every pruning method is green or that intervention sensitivity proves universal grounding.
