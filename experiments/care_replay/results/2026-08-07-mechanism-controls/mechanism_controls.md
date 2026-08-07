# CARE 2.0 Mechanism-Control Summary

All effects are paired held-out composite deltas with normal 95% confidence intervals. Positive values favor the left-hand route.

| Source -> target | Deployed | vs strongest BO (final) | Warm-start effect | Post-init source-outcome effect | LLM increment vs fixed data-only | Action change |
|---|---|---:|---:|---:|---:|---:|
| real_chemlex_acidamine -> real_buchwald_hartwig | llm_transfer_router | 11.452 [9.025, 13.878] | 22.460 [18.196, 26.723] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 |
| real_matbench_dielectric -> real_matbench_expt_gap | llm_transfer_router | 42.461 [37.932, 46.990] | 85.616 [75.374, 95.859] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.385 |
| real_matbench_expt_gap -> real_matbench_dielectric | matched_target_only_llm | 10.017 [6.562, 13.471] | -0.912 [-6.080, 4.255] | 0.000 [0.000, 0.000] | 29.424 [29.424, 29.424] | 0.000 |
| real_matbench_phonons -> real_matbench_dielectric | matched_target_only_llm | 3.369 [-0.331, 7.069] | 3.361 [-1.565, 8.287] | -1.296 [-2.516, -0.075] | -2.642 [-6.142, 0.859] | 0.019 |
| real_moleculenet_esol -> real_moleculenet_lipophilicity | matched_target_only_llm | 1.089 [0.165, 2.013] | 0.789 [-0.686, 2.265] | -0.145 [-0.565, 0.275] | -0.223 [-1.573, 1.127] | 0.090 |
| real_moleculenet_freesolv -> real_moleculenet_lipophilicity | matched_target_only_llm | 2.126 [1.087, 3.165] | 0.937 [-0.586, 2.459] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 |
| real_moleculenet_lipophilicity -> real_moleculenet_freesolv | matched_target_only_llm | 31.760 [29.109, 34.411] | -1.173 [-1.408, -0.938] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.078 |

## Interpretation

- Warm-start effect compares the same target-only policy after replacing its initial design with the source-informed design.
- Post-init source-outcome effect compares the full route with the same source-informed initialization but zero transfer mass after initialization.
- LLM increment compares the full LLM patch with a deterministic equal-role data-only transfer patch.
- These runs are offline replay model selection. A positive result does not make the calibration protocol wet-lab deployment ready.
