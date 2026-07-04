# Incumbent Rule Ablation

This snapshot checks whether the current public incumbent is unusually strong
and which part of the rule contributes most to target-only replay performance.

The incumbent is not a CARE 1.0 reproduction or a named literature baseline. It
is the current transparent target-only control used in the CARE 2.0 replay
harness.

## Setting

- Datasets: `real_buchwald_hartwig`,
  `real_moleculenet_lipophilicity`
- Seeds: 50
- Initial observations: 5
- Reveal budget: 10
- Compared rules:
  - `random`: uniform random unrevealed candidate.
  - `group_only`: smoothed group mean plus uncertainty.
  - `factor_only`: smoothed decision-factor means only.
  - `factor_ucb`: factor means plus UCB-like uncertainty.
  - `no_condition_prior`: current incumbent without the `x1/x2/x3` prior.
  - `public_incumbent`: current target-only baseline.

## Summary

| Dataset | Mode | Final Best | Delta vs Incumbent | AUC | Delta AUC | Top-10 Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Buchwald-Hartwig | random | 82.1728 | -4.4749 | 77.4347 | -2.5366 | 0.0600 |
| Buchwald-Hartwig | group_only | 86.1706 | -0.4771 | 82.9826 | +3.0113 | 0.0600 |
| Buchwald-Hartwig | factor_only | 86.3136 | -0.3341 | 80.0357 | +0.0644 | 0.2400 |
| Buchwald-Hartwig | factor_ucb | 86.7960 | +0.1483 | 79.5502 | -0.4211 | 0.2000 |
| Buchwald-Hartwig | no_condition_prior | 86.1992 | -0.4485 | 80.4621 | +0.4908 | 0.2000 |
| Buchwald-Hartwig | public_incumbent | 86.6477 | 0.0000 | 79.9713 | 0.0000 | 0.1600 |
| Lipophilicity | random | 85.7650 | -1.5425 | 84.2917 | -1.2743 | 0.0200 |
| Lipophilicity | group_only | 86.3200 | -0.9875 | 84.4772 | -1.0888 | 0.0000 |
| Lipophilicity | factor_only | 87.5225 | +0.2150 | 85.6342 | +0.0682 | 0.0800 |
| Lipophilicity | factor_ucb | 87.4550 | +0.1475 | 85.8572 | +0.2912 | 0.1600 |
| Lipophilicity | no_condition_prior | 87.1875 | -0.1200 | 85.3805 | -0.1855 | 0.0800 |
| Lipophilicity | public_incumbent | 87.3075 | 0.0000 | 85.5660 | 0.0000 | 0.0400 |

## Readout

The current incumbent is a strong target-only baseline because it beats random
search by a clear margin on both datasets. It does not look like an unbeatable
oracle. On Buchwald-Hartwig, several simple variants sit in the same range. On
Lipophilicity, `factor_only` and `factor_ucb` slightly beat the full incumbent
on final best or AUC.

This makes the LLM result easier to interpret. The LLM variants are not losing
only because the incumbent is artificially too strong. They are also limited by
their current role: the model proposes bounded factor-level adjustments inside a
hand-written schema, while the deterministic transfer rule is already calibrated
for that schema.

The next LLM experiment should let the model propose rule-level updates, such as
role-map confidence, support thresholds, effect thresholds, transfer discount,
and shared-descriptor value-prior eligibility. Those proposals should be chosen
on calibration seeds and evaluated on held-out seeds.

## Files

- `incumbent_rule_ablation_summary.csv`: compact cross-dataset summary.
- `incumbent_ablation_real_buchwald_hartwig_50seed_metrics.csv`: per-seed
  Buchwald-Hartwig metrics.
- `incumbent_ablation_real_buchwald_hartwig_50seed_summary.json`: aggregate
  Buchwald-Hartwig summary.
- `incumbent_ablation_real_moleculenet_lipophilicity_50seed_metrics.csv`:
  per-seed Lipophilicity metrics.
- `incumbent_ablation_real_moleculenet_lipophilicity_50seed_summary.json`:
  aggregate Lipophilicity summary.
