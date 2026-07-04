# Hybrid Transfer Budget Sweep

This snapshot extends the 2026-07-03 hybrid surrogate experiment. The baseline
is still target-only mixed-kernel GP-UCB. CARE transfer is allowed to modify the
GP-UCB acquisition through the FreeSolv -> Lipophilicity shared-descriptor
value-prior skill.

The purpose is not to claim a large final-best win. The cleaner result is that
the transferred value prior improves early discovery: it raises best-so-far AUC
and top-10 hit rate over a strong target-only GP-UCB baseline, especially when
the target has a small experiment budget.

## Setting

- Source: `real_moleculenet_freesolv`
- Target: `real_moleculenet_lipophilicity`
- Base incumbent: `mixed_kernel_gp_ucb`
- CARE skill: source transfer card + shared SMILES descriptor value priors
- Source observations: 192
- Initial target observations: 5
- Main seeds: 100
- Budgets tested: 3, 5, and 10 reveal rounds

## Main Results

| Budget | Mode | Final Best | Delta Final | AUC | Delta AUC | Top-10 Hit | Delta Top-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 rounds | `gp_ucb` | 88.9038 | 0.0000 | 86.5433 | 0.0000 | 0.1100 | 0.0000 |
| 10 rounds | `hybrid_value_prior_gp_ucb_gate_v1` | 88.9225 | +0.0187 | 86.7037 | +0.1604 | 0.2100 | +0.1000 |
| 5 rounds | `gp_ucb` | 86.4712 | 0.0000 | 84.9870 | 0.0000 | 0.0400 | 0.0000 |
| 5 rounds | `hybrid_value_prior_gp_ucb_gate_v1` | 86.6175 | +0.1463 | 85.2987 | +0.3117 | 0.0900 | +0.0500 |
| 3 rounds | `gp_ucb` | 85.1813 | 0.0000 | 84.1817 | 0.0000 | 0.0300 | 0.0000 |
| 3 rounds | `hybrid_value_prior_gp_ucb_gate_v1` | 85.4587 | +0.2774 | 84.5908 | +0.4091 | 0.0500 | +0.0200 |

The 10-round final-best delta is tiny, but the top-10 hit rate nearly doubles
from 0.11 to 0.21. Under smaller budgets, the final-best and AUC deltas become
larger. That is the most useful interpretation: the current CARE transfer
skill helps the optimizer reach promising Lipophilicity regions earlier, rather
than fully replacing the target-only GP model.

## Warm-Start Ablation

We also tested whether transfer should only act during the first few rounds and
then hand control back to GP-UCB. This did not improve the result.

| Mode | Final Best | Delta Final | AUC | Delta AUC | Top-10 Hit | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gp_ucb` | 88.4775 | 0.0000 | 86.4578 | 0.0000 | 0.0800 | 0.0000 |
| `hybrid_value_prior_gp_ucb_gate_v1` | 88.8775 | +0.4000 | 86.7270 | +0.2692 | 0.1600 | 3.4000 |
| `hybrid_value_prior_gp_ucb_warm3_gate_v1` | 88.4825 | +0.0050 | 86.5000 | +0.0422 | 0.1200 | 0.7800 |
| `hybrid_value_prior_gp_ucb_warm5_gate_v1` | 88.2700 | -0.2075 | 86.5915 | +0.1337 | 0.1400 | 1.6400 |

Warm-start reduces bad interventions, but it also removes most of the gain.
For this source-target pair, the useful behavior comes from letting the
value-prior skill stay active throughout the short replay.

## Interpretation

This is a positive but bounded result. It is stronger than the original
public-incumbent comparison because GP-UCB is a much tougher target-only
baseline. It also shows why the metric choice matters: final best is nearly
tied at 10 rounds, while AUC and top-10 hit show clearer early-discovery gains.

What we should not overclaim:

- This is not a universal cross-domain win; reverse FreeSolv and ESOL source
  directions still show negative or weak transfer.
- The current meta-gate is still simple. A threshold over source-card features
  looked good on calibration seeds but did not reliably improve held-out final
  best.
- The LLM is not yet optimizing this rule. These runs are deterministic
  transfer-card experiments over a GP-UCB incumbent.

The next useful experiment is a real rule-level proposer: use calibration data
or an LLM to choose transfer strength, descriptor whitelist, and gate threshold,
then evaluate on held-out seeds and target tasks.

## Files

- `hybrid_transfer_budget_summary.csv`: compact comparison across budgets and
  warm-start ablations.
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_100seed_metrics.csv`
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_100seed_summary.json`
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_lowbudget5_100seed_metrics.csv`
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_lowbudget5_100seed_summary.json`
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_lowbudget3_100seed_metrics.csv`
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_lowbudget3_100seed_summary.json`
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_warmstart_50seed_metrics.csv`
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_warmstart_50seed_summary.json`
