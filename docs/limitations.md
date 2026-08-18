# Limitations

1. Intervention-defined response preservation is narrower than semantic grounding and can inherit intervention artifacts.
2. The original 1,200-case legacy partition was not group-clean; it is therefore used only for development. Prospective claims come from the separate 900-case pool.
3. Controller transfer across architectures, pruning implementations, datasets, and hardware is not assumed.
4. Energy findings are implementation- and hardware-dependent. FastV did not show replicated energy savings despite high dense avoidance.
5. The whole-program footprint excludes embodied emissions and unrecoverable idle/setup energy.
6. The SmolVLM-500M comparison is dense-only and experienced a disclosed post-inference analysis failure; it supports a limited scale frontier, not full SLM generalization.
7. The development-only feature analysis indicates substantial reliance on task/source identity and does not establish transfer to unseen task families.
