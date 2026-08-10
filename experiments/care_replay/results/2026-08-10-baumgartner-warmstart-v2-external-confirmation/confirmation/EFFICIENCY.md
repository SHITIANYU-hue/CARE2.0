# External confirmation efficiency

## Outcome quality

| Policy | Best-so-far AUC | Final best yield |
|---|---:|---:|
| Frozen v2 transfer | 98.0612 | 100.0000 |
| Space filling | 89.4001 | 91.8190 |
| Random initial, 100-seed mean | 88.8402 | 94.7673 |

The frozen route improves AUC by 8.6611 over the stronger deterministic/mean
baseline and improves final best yield by 5.2327 over the random mean.

## Observation efficiency

The frozen route reaches 100% yield at observation 5: three initial experiments
plus two target-only GP-UCB reveals.

- 46/100 random-initial runs reach 100% within 13 total observations.
- Successful random runs require a median of 7 observations.
- The frozen route therefore saves 2 observations relative to the median
  successful random run.
- Space filling never reaches 95% within the budget.

## Uncertainty

Random-initial AUC is 88.8402 with a normal 95% CI of [87.3443, 90.3361]. The
frozen route is above 85 of 100 random runs; 15 random runs match or exceed it.
Because transfer and space filling are deterministic and there is only one
external target campaign, this is a practical-effect report rather than a
task-level significance claim.
