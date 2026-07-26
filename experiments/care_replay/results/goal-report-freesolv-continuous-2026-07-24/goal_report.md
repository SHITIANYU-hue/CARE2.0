# CARE 2.0 Cross-Domain Goal Report

This report separates source-outcome transfer, target-only baselines, LLM routes, and matched random nulls. All source-outcome comparisons use held-out rows from frozen summaries.

## Source-Outcome Paths

| Source -> target | Route candidate | Deployed | Strongest target BO | Beats BO | Beats matched LLM |
| --- | --- | --- | --- | --- | --- |
| real_moleculenet_esol -> real_moleculenet_freesolv_continuous | schema_aligned_source_outcome | matched_target_only_llm | target_acquisition_portfolio | no | no |
| real_moleculenet_lipophilicity -> real_moleculenet_freesolv_continuous | shared_descriptor_source_outcome | matched_target_only_llm | target_acquisition_portfolio | no | no |

Route proposals are schema-only candidates; the calibration gate still decides whether source-outcome transfer is deployed.

## Random Nulls

| Target | Selected null | Final delta | AUC delta | Win rate |
| --- | --- | ---: | ---: | ---: |

## Traditional Transfer Baselines

These are fixed weighted-kernel transfer runs, reported descriptively because their task pairs and seed counts are not the same as the frozen seven-pair CARE suite.

| Pair | Mode | Final delta | AUC delta | Primary gain |
| --- | --- | ---: | ---: | --- |

## Interpretation

The report does not classify a fallback as a positive transfer gain. A random null with a positive mean but a confidence interval crossing zero is also not treated as stable generalization. A warm-start null can still beat GP-UCB, which is why initialization controls are reported separately. Traditional weighted-kernel rows are kept separate from the frozen CARE claim because their confidence intervals cross zero in this descriptive batch.
