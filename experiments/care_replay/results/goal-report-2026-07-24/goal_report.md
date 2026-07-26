# CARE 2.0 Cross-Domain Goal Report

This report separates source-outcome transfer, target-only baselines, LLM routes, and matched random nulls. All source-outcome comparisons use held-out rows from frozen summaries.

## Source-Outcome Paths

| Source -> target | Route candidate | Deployed | Strongest target BO | Beats BO | Beats matched LLM |
| --- | --- | --- | --- | --- | --- |
| real_chemlex_acidamine -> real_buchwald_hartwig | reaction_role_source_outcome | llm_transfer_router | target_acquisition_portfolio | yes | yes |
| real_matbench_dielectric -> real_matbench_expt_gap | shared_descriptor_source_outcome | llm_transfer_router | mixed_kernel_gp_ei | yes | yes |
| real_matbench_expt_gap -> real_matbench_dielectric | shared_descriptor_source_outcome | llm_transfer_router | target_acquisition_portfolio | yes | yes |
| real_matbench_phonons -> real_matbench_dielectric | shared_descriptor_source_outcome | matched_target_only_llm | target_acquisition_portfolio | no | no |
| real_moleculenet_esol -> real_moleculenet_lipophilicity | schema_aligned_source_outcome | matched_target_only_llm | gp_ucb | no | no |
| real_moleculenet_freesolv -> real_moleculenet_lipophilicity | shared_descriptor_source_outcome | llm_transfer_router | target_acquisition_portfolio | yes | yes |
| real_moleculenet_lipophilicity -> real_moleculenet_freesolv | shared_descriptor_source_outcome | matched_target_only_llm | target_acquisition_portfolio | no | no |

Route proposals are schema-only candidates; the calibration gate still decides whether source-outcome transfer is deployed.

## Random Nulls

| Target | Selected null | Final delta | AUC delta | Win rate |
| --- | --- | ---: | ---: | ---: |
| real_matbench_phonons | llm_semantic_random_rule_r2_enhance_chalcogenide_phonon_response | -1.3163 | -0.6295 | 36.7% |
| real_moleculenet_esol | llm_semantic_random_rule_r1_preserve_low_rotatable_bond_structures | 0.4924 | 0.2297 | 36.7% |
| real_moleculenet_esol | llambo_warmstart_random_rule_r1_counter_hypothesis_low_hbond_donors | 1.1143 | 2.1257 | 56.7% |

## Traditional Transfer Baselines

These are fixed weighted-kernel transfer runs, reported descriptively because their task pairs and seed counts are not the same as the frozen seven-pair CARE suite.

| Pair | Mode | Final delta | AUC delta | Primary gain |
| --- | --- | ---: | ---: | --- |
| suzuki_bh | transfer_weighted_gp_ucb_scale_ensemble_0p5_1_1p5_2_3_4 | 1.0484 | 0.6326 | no |
| suzuki_chemlex | transfer_weighted_gp_ucb_scale_ensemble_0p5_1_1p5_2_3_4 | 2.4961 | 0.8783 | no |
| chemlex_bh | transfer_weighted_gp_ucb_scale_ensemble_1_1p5_2_3_4_6 | -1.8382 | -1.3454 | no |

## Interpretation

The report does not classify a fallback as a positive transfer gain. A random null with a positive mean but a confidence interval crossing zero is also not treated as stable generalization. A warm-start null can still beat GP-UCB, which is why initialization controls are reported separately. Traditional weighted-kernel rows are kept separate from the frozen CARE claim because their confidence intervals cross zero in this descriptive batch.
