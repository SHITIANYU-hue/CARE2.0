# Scale-Ensemble Acquisition Transfer Follow-Up

This follow-up tests whether acquisition-level transfer can be made less brittle than a single hand-picked GP-kernel scale.

Instead of choosing one transfer-weight scale, the new policy runs several transfer-weighted GP-UCB acquisitions in parallel and averages normalized candidate ranks. The source transfer card still only changes public categorical-kernel weights; it does not read hidden target outcomes or directly select candidate ids.

## Policy

New mode:

```text
transfer_weighted_gp_ucb_scale_ensemble_*
```

Implementation:

- compute GP-UCB scores for each transfer-weight scale;
- convert each scale's scores to normalized ranks;
- average the normalized ranks;
- select the candidate with the highest average rank score.

## 50-Seed Results

| Pair | Ensemble scales | Delta final vs GP-UCB | Delta AUC vs GP-UCB | Delta top-10 vs GP-UCB |
| --- | --- | ---: | ---: | ---: |
| Suzuki -> Buchwald-Hartwig | 0.5, 1, 1.5, 2, 3, 4 | +1.0484 | +0.6326 | -0.06 |
| Suzuki -> ChemLex | 0.5, 1, 1.5, 2, 3, 4 | +2.4961 | +0.8783 | +0.02 |
| ChemLex -> Buchwald-Hartwig | 1, 1.5, 2, 3, 4, 6 | -1.8382 | -1.3454 | -0.04 |

## Interpretation

The useful result is not that every ensemble helps. It does not.

The important change is that `Suzuki -> Buchwald-Hartwig` now has a stronger acquisition-level win over GP-UCB than the earlier single-scale result. The old best single-scale transfer was only about `+0.3001 final best / +0.3162 AUC` over GP-UCB. The scale ensemble increases that to `+1.0484 final best / +0.6326 AUC`.

`Suzuki -> ChemLex` stays strongly positive, though the old single-scale `scale=1` still has the best full-mean final/AUC. The ensemble is useful as a more robust executable policy, not as the absolute oracle.

`ChemLex -> Buchwald-Hartwig` is a negative result. High-scale transfer looked promising in held-out oracle inspection, but the executable high-scale ensemble is worse than GP-UCB. This should be kept as a boundary case: ChemLex source knowledge does not yet provide a stable reusable acquisition skill for Buchwald-Hartwig.

## Files

- `scale_ensemble_summary.csv`: compact comparison against GP-UCB.
- `transfer_weighted_kernel_real_suzuki_miyaura_to_real_buchwald_hartwig_scale_ensemble_mid_50seed_metrics.csv`
- `transfer_weighted_kernel_real_suzuki_miyaura_to_real_chemlex_acidamine_scale_ensemble_mid_50seed_metrics.csv`
- `transfer_weighted_kernel_real_chemlex_acidamine_to_real_buchwald_hartwig_scale_ensemble_high_50seed_metrics.csv`

