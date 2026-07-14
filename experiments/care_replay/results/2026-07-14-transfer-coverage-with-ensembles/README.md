# Transfer Coverage Portfolio With Scale Ensembles

This run extends the 2026-07-12 transfer coverage portfolio with three acquisition-level scale-ensemble policies.

The goal is not to claim that transfer always helps. The goal is to test whether a more robust executable transfer policy can improve the number of source-target pairs where CARE 2.0 beats strong target-only search.

## What changed

Added `weighted_kernel_ensemble` sources for:

- `real_suzuki_miyaura -> real_buchwald_hartwig`
- `real_suzuki_miyaura -> real_chemlex_acidamine`
- `real_chemlex_acidamine -> real_buchwald_hartwig`

Each ensemble averages normalized GP-UCB ranks from multiple transfer-weight scales instead of selecting a single scale by hand.

## Summary

```json
{
  "pair_count": 11,
  "policy_count": 137,
  "missing_inputs": [],
  "best_transfer_positive_final_vs_public": 7,
  "best_transfer_positive_final_vs_gp_ucb": 4,
  "best_transfer_positive_final_vs_best_target": 3,
  "calibrated_selector_positive_heldout_final_vs_gp_ucb": 4,
  "transfer_only_selector_positive_heldout_final_vs_gp_ucb": 3,
  "transfer_only_selector_positive_heldout_final_vs_best_target": 2
}
```

Compared with the previous portfolio, the scale ensembles improve selector coverage against GP-UCB:

- calibrated selector held-out positive vs GP-UCB: `3/11 -> 4/11`;
- transfer-only selector held-out positive vs GP-UCB: `2/11 -> 3/11`;
- transfer-only selector held-out positive vs best target-only baseline: `1/11 -> 2/11`.

The strict risk-aware selector remains conservative. It selects transfer in 2 pairs when judged against GP-UCB, and both selected pairs stay positive on held-out seeds.

## Main pair-level changes

`Suzuki-Miyaura -> Buchwald-Hartwig` is the most useful new result. The best executable transfer policy becomes:

```text
weighted_kernel_ensemble:transfer_weighted_gp_ucb_scale_ensemble_0p5_1_1p5_2_3_4
```

It improves over GP-UCB by `+1.0484` final best and `+0.6326` AUC across 50 seeds. On held-out seeds selected by calibration, the gain is `+1.5501` final best and `+0.4664` AUC.

`Suzuki-Miyaura -> ChemLex` remains strongly positive. The single-scale `transfer_weighted_gp_ucb_scale_1` is still the best full-mean policy, but the new ensemble is also positive: `+2.4961` final best and `+0.8783` AUC vs GP-UCB.

`ChemLex -> Buchwald-Hartwig` is a negative transfer boundary. The high-scale ensemble is worse than GP-UCB by `-1.8382` final best and `-1.3454` AUC. This is useful evidence for the gate: ChemLex source knowledge should not be blindly reused for Buchwald-Hartwig.

## Files

- `summary.json`: aggregate counts.
- `pair_summary.csv`: best policy and selector behavior per source-target pair.
- `policy_summary.csv`: per-policy full/calibration/held-out metrics.
