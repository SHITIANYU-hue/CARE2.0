# Online LLM portfolio versus rule-fixed baselines

This audit replays all 17 routes declared in the analysis configuration from the exact online-LLM initial candidates, target pool and 10-reveal budget.

## Aggregate results

| Comparator | Mean AUC delta | Route-bootstrap 95% CI | Win / tie / loss | Route sign-test p |
|---|---:|---:|---:|---:|
| Target-only GP-UCB | +0.762 | [-0.501, +2.091] | 8 / 4 / 5 | 0.5811 |
| Rule-fixed RGPE | +0.987 | [-2.901, +5.960] | 8 / 2 / 7 | 1.0000 |
| Strongest realized baseline (post-hoc diagnostic) | -1.821 | [-4.498, +0.604] | 3 / 2 / 12 | 0.0352 |

## Interpretation

The online LLM controller has an equal-route mean of +0.762 against target-only GP-UCB, with 8 wins, 4 ties and 5 losses. The route-bootstrap interval and exact route sign test must be interpreted together because the portfolio is small and the effect is heterogeneous.

Against the rule-fixed classical transfer comparator, the equal-route mean is +0.987; the interval is [-2.901, +5.960] and 7 routes lose. The evidence does not establish stable superiority over classical transfer BO.

## Claim boundary

This audit covers every route declared in the analysis configuration and applies one comparator rule uniformly. However, the online LLM trajectories were observed before this audit, each route has one completed trajectory, and route-bootstrap intervals quantify benchmark-route variation rather than LLM stochastic variation. Confirmatory claims require the already frozen 30-replicate protocol or a prospectively collected wet-lab test.
