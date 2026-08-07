# CARE 2.0 Mechanism-Control Summary

All effects are paired held-out composite deltas with normal 95% confidence intervals. Positive values favor the left-hand route.

| Source -> target | Deployed | vs strongest BO (final) | Warm-start effect | Post-init source-outcome effect | LLM increment vs fixed data-only | Action change |
|---|---|---:|---:|---:|---:|---:|
| real_chemlex_acidamine -> real_buchwald_hartwig | matched_target_only_llm | -3.369 [-6.792, 0.053] | 4.169 [0.283, 8.055] | 1.850 [-1.611, 5.310] | 1.510 [-1.861, 4.880] | 0.277 |
| real_matbench_dielectric -> real_matbench_expt_gap | matched_target_only_llm | 10.877 [0.495, 21.260] | 0.000 [0.000, 0.000] | 6.856 [0.107, 13.605] | 6.880 [0.597, 13.164] | 0.074 |
| real_matbench_expt_gap -> real_matbench_dielectric | matched_target_only_llm | 3.401 [-2.056, 8.858] | 0.000 [0.000, 0.000] | 1.231 [-0.496, 2.959] | -0.293 [-3.017, 2.431] | 0.018 |
| real_matbench_phonons -> real_matbench_dielectric | matched_target_only_llm | 3.375 [-2.527, 9.278] | 0.000 [0.000, 0.000] | 0.439 [-3.483, 4.362] | -0.499 [-1.289, 0.290] | 0.074 |
| real_moleculenet_esol -> real_moleculenet_lipophilicity | matched_target_only_llm | 2.865 [1.523, 4.207] | 0.000 [0.000, 0.000] | 0.078 [-0.109, 0.264] | 0.011 [-0.199, 0.221] | 0.074 |
| real_moleculenet_freesolv -> real_moleculenet_lipophilicity | matched_target_only_llm | 1.890 [0.627, 3.153] | 0.000 [0.000, 0.000] | -0.476 [-1.153, 0.201] | 0.059 [-0.243, 0.362] | 0.082 |
| real_moleculenet_lipophilicity -> real_moleculenet_freesolv | matched_target_only_llm | 34.658 [30.024, 39.292] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.100 |

## Interpretation

- Warm-start effect compares the same target-only policy after replacing its initial design with the source-informed design.
- Post-init source-outcome effect compares the full route with the same source-informed initialization but zero transfer mass after initialization.
- LLM increment compares the full LLM patch with a deterministic equal-role data-only transfer patch.
- These runs are offline replay model selection. A positive result does not make the calibration protocol wet-lab deployment ready.
