# Transfer Significance Audit

This snapshot separates clear transfer gains from boundary results. The goal is
not to find another larger-looking mean, but to check whether each transfer
claim survives paired seed-level uncertainty.

## Method

- Unit of comparison: paired replay seeds.
- Test statistic: transfer metric minus baseline metric for the same seed.
- Confidence interval: non-parametric bootstrap over paired seed deltas.
- Default bootstrap: 5000 samples, random seed 17.
- Metrics checked: `final_best`, `best_so_far_auc`, `top10_hit`, and
  `bad_intervention_count` when available.

For `final_best`, `best_so_far_auc`, and `top10_hit`, a positive delta is good.
For `bad_intervention_count`, a positive delta is a risk cost, not a win.

## Headline Findings

Two transfer settings are now statistically positive over the public incumbent:

| Setting | Metric | Mean delta | 95% bootstrap CI |
| --- | ---: | ---: | ---: |
| FreeSolv -> Lipophilicity value-prior transfer | final best | +2.7550 | [+1.4375, +4.0025] |
| FreeSolv -> Lipophilicity value-prior transfer | AUC | +2.4000 | [+1.3405, +3.4733] |
| FreeSolv -> Lipophilicity value-prior transfer | top-10 hit | +0.2600 | [+0.1400, +0.3800] |
| Suzuki-Miyaura -> Buchwald-Hartwig role transfer | final best | +2.4389 | [+0.8078, +4.3094] |
| Suzuki-Miyaura -> Buchwald-Hartwig role transfer | AUC | +1.1067 | [+0.1809, +2.2873] |
| Suzuki-Miyaura -> Buchwald-Hartwig role transfer | top-10 hit | +0.0800 | [+0.0200, +0.1600] |

These are the current best evidence for meaningful CARE 2.0 transfer. The
molecular-property result is the cleanest because source and target share a real
SMILES-derived descriptor vocabulary. The reaction result is the strongest HTE
case because source knowledge transfers at the role level rather than through
dataset-local component IDs.

## Strong-Baseline Boundary Checks

The stronger GP-UCB comparisons are not yet statistically significant:

| Setting | Metric | Mean delta | 95% bootstrap CI |
| --- | ---: | ---: | ---: |
| Suzuki -> BH transfer-weighted GP kernel vs GP-UCB | final best | +0.3483 | [-0.7797, +1.5775] |
| Suzuki -> BH transfer-weighted GP kernel vs GP-UCB | AUC | +0.3162 | [-0.6464, +1.4234] |
| FreeSolv -> Lipophilicity hybrid value prior vs GP-UCB, 5 rounds | final best | +0.1462 | [-0.6987, +0.9587] |
| FreeSolv -> Lipophilicity hybrid value prior vs GP-UCB, 5 rounds | AUC | +0.3118 | [-0.2967, +0.9308] |

The right interpretation is: transfer can clearly beat the public incumbent in
two real settings, but it has not yet produced a robust win over the stronger
target-only GP-UCB baseline.

## New Target-Calibrated Hybrid Check

This snapshot also includes a fresh 100-seed low-budget run for safer
MoleculeNet hybrid transfer:

```bash
python3 experiments/care_replay/scripts/run_hybrid_surrogate_transfer.py \
  --source-dataset real_moleculenet_freesolv \
  --target-dataset real_moleculenet_lipophilicity \
  --seeds 100 \
  --rounds 5 \
  --initial 5 \
  --source-observations 192 \
  --modes gp_ucb,hybrid_value_prior_gp_ucb_gate_v1,hybrid_value_prior_gp_ucb_strict_gate_v1,hybrid_value_prior_gp_ucb_target_calibrated_gate_v1,hybrid_value_prior_gp_ucb_adaptive_gate_v1 \
  --output-tag target_calibrated_value_prior_lowbudget5_100seed
```

The target-calibrated mode is safer but not stronger:

| Mode | Final best | AUC | Top-10 hit | Bad interventions |
| --- | ---: | ---: | ---: | ---: |
| `gp_ucb` | 86.4712 | 84.9870 | 0.0400 | 0.0000 |
| `hybrid_value_prior_gp_ucb_gate_v1` | 86.6175 | 85.2987 | 0.0900 | 1.4000 |
| `hybrid_value_prior_gp_ucb_target_calibrated_gate_v1` | 86.5900 | 85.0035 | 0.0400 | 0.5100 |

So target calibration reduces bad interventions from 1.40 to 0.51, but it also
removes most of the early-discovery advantage. This is useful as a safety
diagnosis, not as the new headline.

## Files

- `transfer_significance_summary.csv`: flat paired bootstrap summary.
- `transfer_significance_summary.json`: structured version of the same audit.
- `tables/`: curated metrics from the new target-calibrated hybrid run.
- `runs/`: aggregate JSON summary for the new target-calibrated hybrid run.

Regenerate the audit with:

```bash
python3 experiments/care_replay/scripts/build_transfer_significance_summary.py
```
