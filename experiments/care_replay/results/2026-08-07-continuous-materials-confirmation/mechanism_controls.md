# CARE 2.0 Mechanism-Control Summary

All effects are paired held-out composite deltas with normal 95% confidence intervals. Positive values favor the left-hand route.

| Source -> target | Deployed | vs strongest BO (final) | Warm-start effect | Post-init source-outcome effect | LLM increment vs fixed data-only | Action change |
|---|---|---:|---:|---:|---:|---:|
| real_matbench_dielectric -> real_matbench_expt_gap | matched_target_only_llm | 4.514 [-1.844, 10.872] | 0.000 [0.000, 0.000] | 0.347 [-2.066, 2.761] | 1.305 [-2.125, 4.734] | 0.045 |

## Interpretation

- Warm-start effect compares the same target-only policy after replacing its initial design with the source-informed design.
- Post-init source-outcome effect compares the full route with the same source-informed initialization but zero transfer mass after initialization.
- LLM increment compares the full LLM patch with a deterministic equal-role data-only transfer patch.
- These runs are offline replay model selection. A positive result does not make the calibration protocol wet-lab deployment ready.
