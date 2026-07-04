# Surrogate Baseline Comparison

This snapshot adds dependency-free surrogate baselines to the CARE 2.0 replay
harness. The purpose is to compare current CARE transfer results against
stronger target-only optimizers, not only against random search and the
hand-written public incumbent.

## Baselines

- `random`: uniform random selection from unrevealed candidates.
- `public_incumbent`: the current transparent target-only evidence baseline.
- `mixed_kernel_gp_ucb`: Gaussian-process-style mixed categorical/numeric kernel
  surrogate with UCB acquisition.
- `mixed_kernel_gp_ei`: same surrogate with expected-improvement acquisition.
- `knn_ucb`: weighted nearest-neighbor surrogate with an uncertainty bonus.

The GP-style baselines are implemented without `numpy`, `scipy`, or
`sklearn`, so the replay remains runnable in the minimal repository
environment. They use only public candidate features and revealed target
observations.

## Readout

The added baselines change the interpretation in a useful way.

For `FreeSolv -> Lipophilicity`, CARE transfer with shared descriptor value
priors remains the strongest result in this comparison:

| Family | Mode | Final Best | Delta vs Incumbent | AUC | Delta AUC | Top-10 Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CARE transfer | `transfer_value_prior_gate_v1` | 90.0625 | +2.7550 | 87.9660 | +2.4000 | 0.3000 |
| Surrogate | `mixed_kernel_gp_ucb` | 88.4775 | +1.1700 | 86.4578 | +0.8918 | 0.0800 |
| Surrogate | `mixed_kernel_gp_ei` | 88.4675 | +1.1600 | 86.3153 | +0.7493 | 0.0600 |
| Surrogate | `knn_ucb` | 87.7075 | +0.4000 | 86.1762 | +0.6102 | 0.0800 |
| Target-only | `public_incumbent` | 87.3075 | 0.0000 | 85.5660 | 0.0000 | 0.0400 |

For `Suzuki-Miyaura -> Buchwald-Hartwig`, the result is more nuanced.
Transfer clearly beats the public incumbent, but GP-UCB is currently stronger:

| Family | Mode | Final Best | Delta vs Incumbent | AUC | Delta AUC | Top-10 Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Surrogate | `mixed_kernel_gp_ucb` | 91.1145 | +4.4668 | 82.9700 | +2.9987 | 0.3200 |
| CARE transfer | `transfer_gate_v1` | 89.0866 | +2.4389 | 81.0779 | +1.1066 | 0.2400 |
| Surrogate | `mixed_kernel_gp_ei` | 88.3858 | +1.7381 | 81.5814 | +1.6101 | 0.2800 |
| Surrogate | `knn_ucb` | 87.3065 | +0.6588 | 81.0692 | +1.0979 | 0.1200 |
| Target-only | `public_incumbent` | 86.6477 | 0.0000 | 79.9713 | 0.0000 | 0.1600 |

This is a good pressure test. The current reaction transfer result should be
reported as positive against incumbent, but not as stronger than all target-only
surrogate baselines. The molecular transfer result is stronger: it beats the
added surrogate baselines on both final best and AUC.

## Next Experiment

The next CARE 2.0 target should be hybrid rather than either/or:

1. Use GP-UCB or a similar surrogate as a stronger incumbent.
2. Let the transfer card modify the surrogate acquisition, not just the current
   hand-written public incumbent.
3. Let the LLM propose rule-level changes, such as role-map confidence,
   support thresholds, transfer discount, and shared-descriptor eligibility.
4. Evaluate those rule updates on held-out seeds so LLM participation is not
   just prompt-level score nudging.

## Files

- `surrogate_vs_transfer_summary.csv`: compact comparison against the existing
  50-seed transfer advantage sweep.
- `surrogate_baselines_real_buchwald_hartwig_50seed_metrics.csv`: per-seed
  surrogate baseline metrics.
- `surrogate_baselines_real_buchwald_hartwig_50seed_summary.json`: aggregate
  Buchwald-Hartwig summary.
- `surrogate_baselines_real_moleculenet_lipophilicity_50seed_metrics.csv`:
  per-seed surrogate baseline metrics.
- `surrogate_baselines_real_moleculenet_lipophilicity_50seed_summary.json`:
  aggregate Lipophilicity summary.
