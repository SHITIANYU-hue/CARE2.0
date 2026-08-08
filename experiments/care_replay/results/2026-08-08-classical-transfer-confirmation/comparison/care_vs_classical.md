# CARE vs classical transfer BO

The classical opponent is selected on calibration seeds and frozen before held-out comparison.

| Pair | Selected classical arm | CARE final delta [95% CI] | CARE AUC delta [95% CI] |
| --- | --- | ---: | ---: |
| molecular_freesolv_to_lipophilicity | target_gp_ucb | 1.978 [0.958, 2.997] | 1.578 [0.621, 2.534] |
| molecular_lipophilicity_to_freesolv | target_gp_ucb | 35.023 [32.100, 37.946] | 42.995 [40.287, 45.703] |
| materials_expt_gap_to_dielectric | multitask_gp_icm | 4.472 [0.937, 8.008] | 2.884 [0.321, 5.447] |
| materials_dielectric_to_expt_gap | rgpe | -33.250 [-38.511, -27.989] | -44.838 [-49.274, -40.403] |
| materials_phonons_to_dielectric | multitask_gp_icm | 4.329 [0.998, 7.659] | 0.045 [-1.911, 2.002] |

## Calibration-selected CARE/classical portfolio

| Pair | Frozen portfolio arm | Final delta vs target GP [95% CI] | AUC delta vs target GP [95% CI] |
| --- | --- | ---: | ---: |
| molecular_freesolv_to_lipophilicity | care_source_outcome_router | 1.978 [0.958, 2.997] | 1.578 [0.621, 2.534] |
| molecular_lipophilicity_to_freesolv | care_source_outcome_router | 35.023 [32.100, 37.946] | 42.995 [40.287, 45.703] |
| materials_expt_gap_to_dielectric | care_source_outcome_router | 5.859 [2.241, 9.478] | 3.530 [0.883, 6.178] |
| materials_dielectric_to_expt_gap | rgpe | 41.697 [37.089, 46.306] | 49.757 [46.111, 53.403] |
| materials_phonons_to_dielectric | multitask_gp_icm | 2.127 [-0.523, 4.776] | 2.897 [1.020, 4.775] |

A positive delta favors CARE. This table does not credit LLM causality; that requires the separate fixed data-only control.
The hybrid portfolio is an offline calibration result and is not a zero-target-cost deployment gate.
