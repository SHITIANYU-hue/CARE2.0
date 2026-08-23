# CARE 2.0 Submission Evidence Audit

## Primary estimand

Does the online LLM controller improve best-so-far AUC over the same LLM initial design followed by target-only GP-UCB under the same reveal budget?

The matched comparator receives the same LLM-generated initial observations and the same target-evaluation budget. The contrast therefore isolates the online LLM controller from the initial-design effect.

## Current result

Across all 11 routes, the mean route-level AUC delta is +1.0135 with a descriptive bootstrap 95% interval [-0.4433, +2.7821]. The routes contain 6 wins, 2 ties, and 3 losses; the exact two-sided sign-test p-value excluding ties is 0.5078.
The post-freeze chemistry extension contains 3 wins, 1 tie, and 1 loss. Its mean delta is +0.0212, bootstrap 95% interval [-0.5171, +0.4194], and exact two-sided sign-test p=0.6250.

**Decision:** the current result is a route-specific, majority-positive signal. It is not a statistically confirmatory claim of general cross-domain transfer.

## Claim boundary

> The controller produced route-specific positive transfer signals under matched budgets, including a majority-positive post-freeze chemistry extension, but current route-level uncertainty does not support a universal or statistically confirmatory cross-domain claim.

Claims currently supported:

- The LLM made an executable choice on every online round in this suite.
- Six of eleven routes improved best-so-far AUC relative to the matched same-initial GP comparator.
- The post-freeze chemistry extension produced three wins, one tie, and one loss.

Claims not currently supported:

- Statistically significant general positive transfer across domains.
- Universal positive transfer or elimination of negative transfer.
- Superiority of the complete LLM system over the fixed initializer on unseen tasks.
- Independent wet-lab validation or model-weight self-improvement.

## Submission readiness

| Criterion | Status | Evidence |
|---|---|---|
| Matched target-only comparator | PASS | Same initial observations and reveal budget; only the online controller differs. |
| Target-outcome leakage control | PASS | Prompts expose target outcomes only after the selected experiment is revealed. |
| Post-freeze task extension | PARTIAL | Five chemistry routes were added after the high-authority controller was frozen. |
| Repeated stochastic LLM trials | FAIL | Each route currently has one online Opus trajectory for this evidence suite. |
| Confirmatory route-level significance | FAIL | Frozen-extension bootstrap interval crosses zero and the sign test is not significant. |
| Full-system superiority | FAIL | The complete system is less stable because outcome-blind LLM initial design can hurt performance. |
| Independent wet-lab validation | FAIL | The present evidence is finite-pool replay on real datasets, not a new physical campaign. |

## Route-level evidence

| Evidence tier | Route | Domain | Online AUC delta | Outcome |
|---|---|---|---:|---|
| retrospective_development | molecular_freesolv_to_esol | molecular_property | +0.0000 | tie |
| retrospective_development | molecular_freesolv_to_lipophilicity | molecular_property | +6.3250 | win |
| retrospective_development | materials_expt_gap_to_dielectric | materials_property | +6.7451 | win |
| retrospective_development | materials_phonons_to_bulk_modulus | materials_property | +1.1302 | win |
| retrospective_development | materials_dielectric_to_jdft2d | materials_property | -0.7119 | loss |
| retrospective_development | materials_expt_gap_to_mp_gap | materials_property | -2.4464 | loss |
| post_freeze_extension | baumgartner_aniline_to_phenethylamine_alphos | reaction_optimization_cn | +0.6286 | win |
| post_freeze_extension | baumgartner_aniline_to_benzamide_tbuxphos | reaction_optimization_cn | +0.1781 | win |
| post_freeze_extension | baumgartner_aniline_to_phenethylamine_tbubrettphos | reaction_optimization_cn | -0.9806 | loss |
| post_freeze_extension | baumgartner_benzamide_tbubrettphos_to_alphos | reaction_optimization_cn | +0.0000 | tie |
| post_freeze_extension | reizman_cases_123_to_case4 | reaction_optimization_suzuki | +0.2800 | win |

## Statistical boundary

The bootstrap resamples source-target routes and assumes they are exchangeable. Several routes share task families and data-generation structure, so this interval is a sensitivity analysis rather than a hierarchical population estimate. Each route currently contributes one online Opus trajectory; repeated stochastic LLM runs are required to estimate within-route variance.

## Minimum next evidence package

1. Freeze the controller, prompt, candidate-menu policy, and analysis code before any new target is revealed.
2. Run repeated LLM trajectories per route with predeclared seeds and matched GP controls.
3. Add at least one fresh task family not used during prompt or controller development.
4. Evaluate the complete system and the online-controller increment separately.
5. Run a paired wet-lab campaign or an independently held prospective campaign before claiming practical scientific acceleration.
6. Report all negative routes, abstentions, model failures, token cost, and experiment count.

## Provenance

Source route metrics: `experiments/care_replay/results/2026-08-16-opus48-generalization-evidence-v1/combined_metrics.csv`

Generated vector figures: `route_level_auc_deltas.svg` and `phase_evidence_summary.svg`.
