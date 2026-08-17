# Opus High-Authority Generalization Evidence

## Main result

The online Opus controller was evaluated on 11 real routes and changed best-so-far AUC by +1.0135 on average: 6 wins, 2 ties, and 3 losses.
The frozen chemistry extension alone produced 3 wins, 1 ties, and 1 losses with mean AUC delta +0.0212.
LLM participation and decision authority were both 100%. Each round exposed 4.99 executable candidates on average, and Opus overrode GP in 45.5% of rounds.

## Domain view

| Task family | Routes | Mean AUC delta | Wins | Ties | Losses |
|---|---:|---:|---:|---:|---:|
| materials_property | 4 | +1.1793 | 2 | 0 | 2 |
| molecular_property | 2 | +3.1625 | 1 | 1 | 0 |
| reaction_optimization_cn | 4 | -0.0435 | 2 | 1 | 1 |
| reaction_optimization_suzuki | 1 | +0.2800 | 1 | 0 | 0 |

## Route-level evidence

| Phase | Route | Task family | Online delta vs same-initial GP | Full delta vs fixed | GP override | Source active |
|---|---|---|---:|---:|---:|---:|
| development | molecular_freesolv_to_esol | molecular_property | +0.0000 | +0.0000 | 30% | 80% |
| development | molecular_freesolv_to_lipophilicity | molecular_property | +6.3250 | +15.7500 | 40% | 60% |
| development | materials_expt_gap_to_dielectric | materials_property | +6.7451 | +15.3780 | 40% | 20% |
| development | materials_phonons_to_bulk_modulus | materials_property | +1.1302 | +0.1165 | 50% | 0% |
| development | materials_dielectric_to_jdft2d | materials_property | -0.7119 | -1.4422 | 60% | 100% |
| development | materials_expt_gap_to_mp_gap | materials_property | -2.4464 | -10.6188 | 30% | 100% |
| frozen_extension | baumgartner_aniline_to_phenethylamine_alphos | reaction_optimization_cn | +0.6286 | +0.8597 | 60% | 0% |
| frozen_extension | baumgartner_aniline_to_benzamide_tbuxphos | reaction_optimization_cn | +0.1781 | +0.4043 | 30% | 70% |
| frozen_extension | baumgartner_aniline_to_phenethylamine_tbubrettphos | reaction_optimization_cn | -0.9806 | -11.3442 | 40% | 30% |
| frozen_extension | baumgartner_benzamide_tbubrettphos_to_alphos | reaction_optimization_cn | +0.0000 | +0.0000 | 80% | 30% |
| frozen_extension | reizman_cases_123_to_case4 | reaction_optimization_suzuki | +0.2800 | -4.8500 | 40% | 60% |

## What the result supports

The online controller shows majority-positive transfer across molecular-property, materials-property, C-N, and Suzuki task families under a matched target-evaluation budget. The frozen chemistry extension is prospective with respect to the high-authority controller and is the main generalization check.

## What the result does not support

The result is not universal positive transfer. Three of eleven routes are negative, and the frozen-extension mean effect is small. The full system is less stable than the online controller because outcome-blind LLM initial design can underperform the fixed source-diverse initializer.

## Rejected diagnostic

After the frozen extension, the same routes were rerun with a compiled initial design. This was a diagnostic, not a second holdout. It yielded 1 wins, 3 ties, and 1 losses with mean online delta -0.0217, so it is not adopted as the default controller.

## Audit boundary

Every route keeps the frozen initial record, full proposer and critic requests/responses, target reveals, per-case summary, and SHA-256 fingerprints. Target outcomes enter the prompt only after the corresponding experiment has been selected.
