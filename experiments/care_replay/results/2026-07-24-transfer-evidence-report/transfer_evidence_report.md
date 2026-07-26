# CARE 2.0 Transfer Evidence Ladder

This report separates source-schema semantic transfer, source-outcome transfer, warm-start effects, continuous routes, and exact fallback.
All source-schema comparisons below use paired held-out seeds; route selection happened before those seeds were evaluated.

## Source-Schema Semantic Transfer

| Source -> target | Execution | Held-out route | vs matched target LLM | vs strongest BO |
| --- | --- | --- | --- | --- |
| real_moleculenet_esol -> real_moleculenet_lipophilicity | warm-start | `llambo_warmstart_electron_withdrawing_lipophilicity_noprior` | yes | yes |
| real_moleculenet_lipophilicity -> real_moleculenet_freesolv | continuous semantic route | `llm_direct_prior_polar_surface_optimizer_noprior` | no | yes |
| real_matbench_expt_gap -> real_matbench_dielectric | warm-start | `llambo_warmstart_domain_knowledge_high_density_refractivity_noprior` | yes | yes |
| real_matbench_phonons -> real_matbench_dielectric | continuous semantic route | `llm_direct_prior_ionic_covalent_balance_refractive_conservative` | yes | yes |
| real_chemlex_acidamine -> real_buchwald_hartwig | continuous semantic route | `llm_direct_prior_counter_source_acid_halide_reversal_noprior` | yes | yes |

Stable source-schema gains: 4/5 vs matched target-only LLM; 5/5 vs strongest target-only BO.
Among the positive source-schema rows, 2 use a continuous semantic route and 2 use warm-start only.

## Source-Outcome Portfolio

The stricter portfolio contains 3 paths: 1 source-informed deployments, 1 warm-start deployments, 0 continuous-transfer candidates, and 2 exact fallbacks.
A fallback is not counted as a positive transfer result.

## Controls

Random-rule nulls: 3 tested, 1 stable positive. Traditional transfer suites: 3.

## Interpretation

The current evidence supports cross-domain generalization of a routed LLM semantic skill against strong target-only baselines. It does not support the stronger claim that every source-target pair improves, or that every source-outcome gain is continuous after initialization. Those two mechanisms are reported separately so future gains can be attributed rather than overclaimed.
