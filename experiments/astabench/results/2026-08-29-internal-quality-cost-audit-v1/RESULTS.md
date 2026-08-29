# CARE 2.0 quality-token audit

This audit adopts AstaBench's quality-cost reporting principle for the
existing CARE online-controller evidence. It does not report an AstaBench
score and must not be presented as one.

- Source: `experiments/care_replay/results/2026-08-23-online-llm-repeated-confirmation-v1/aggregate/route_statistics.csv`
- Routes: 11
- Completed frozen trajectories: 11 / 330
- Mean AUC delta versus same-initial target GP: +1.7532
- Route-bootstrap 95% interval: [+0.4005, +3.3280]
- Win / tie / loss: 6 / 2 / 3
- Recorded model tokens: 1,674,083 total; 150,181 median per route

## Interpretation

The first trajectory on several molecular, materials, and reaction routes
improved early discovery relative to the same-initial target-only GP, but
the route-level interval remains descriptive and the 30-trajectory-per-route
confirmation is incomplete. Negative routes are retained. Token counts expose
a substantial inference burden and motivate the external AstaBench comparison
against simple ReAct under matched resource limits.

No monetary cost is reported: historical provider prices were not frozen, so
converting these tokens to dollars now would create a post-hoc and potentially
incorrect estimate.

## Route table

| Route | Domain | AUC delta | Tokens | Delta / 100k tokens | Outcome |
|---|---|---:|---:|---:|---|
| baumgartner_aniline_to_benzamide_tbuxphos | reaction_optimization_cn | -0.1781 | 170,775 | -0.1043 | loss |
| baumgartner_aniline_to_phenethylamine_alphos | reaction_optimization_cn | -0.7101 | 146,952 | -0.4832 | loss |
| baumgartner_aniline_to_phenethylamine_tbubrettphos | reaction_optimization_cn | +2.3662 | 152,774 | +1.5488 | win |
| baumgartner_benzamide_tbubrettphos_to_alphos | reaction_optimization_cn | +0.0000 | 136,686 | +0.0000 | tie |
| materials_dielectric_to_jdft2d | materials_property | -0.5243 | 162,181 | -0.3233 | loss |
| materials_expt_gap_to_dielectric | materials_property | +6.7451 | 150,181 | +4.4913 | win |
| materials_expt_gap_to_mp_gap | materials_property | +2.4464 | 148,052 | +1.6524 | win |
| materials_phonons_to_bulk_modulus | materials_property | +1.4350 | 149,612 | +0.9592 | win |
| molecular_freesolv_to_esol | molecular_property | +0.0000 | 170,754 | +0.0000 | tie |
| molecular_freesolv_to_lipophilicity | molecular_property | +6.3250 | 172,099 | +3.6752 | win |
| reizman_cases_123_to_case4 | reaction_optimization_suzuki | +1.3800 | 114,017 | +1.2103 | win |

**Claim boundary:** Descriptive first-trajectory quality/token audit. The frozen repeated confirmation remains incomplete, and no monetary cost is inferred without a frozen provider price snapshot.
