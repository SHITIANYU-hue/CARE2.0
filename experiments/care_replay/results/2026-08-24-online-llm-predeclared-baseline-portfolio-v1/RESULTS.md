# Online LLM portfolio versus rule-fixed baselines

This audit replays all 11 routes declared in the analysis configuration from the exact online-LLM initial candidates, target pool and 10-reveal budget.

## Aggregate results

| Comparator | Mean AUC delta | Route-bootstrap 95% CI | Win / tie / loss | Route sign-test p |
|---|---:|---:|---:|---:|
| Target-only GP-UCB | +1.753 | [+0.415, +3.315] | 6 / 2 / 3 | 0.5078 |
| Rule-fixed RGPE | +3.229 | [-1.844, +10.141] | 6 / 1 / 4 | 0.7539 |
| Strongest realized baseline (post-hoc diagnostic) | -0.550 | [-3.838, +2.193] | 3 / 1 / 7 | 0.3438 |

## Interpretation

The online LLM controller has an equal-route mean of +1.753 against target-only GP-UCB, with 6 wins, 2 ties and 3 losses. The route-bootstrap interval and exact route sign test must be interpreted together because the portfolio is small and the effect is heterogeneous.

Against the rule-fixed classical transfer comparator, the equal-route mean is +3.229; the interval is [-1.844, +10.141] and 4 routes lose. The evidence does not establish stable superiority over classical transfer BO.

## Claim boundary

This audit covers every route declared in the analysis configuration and applies one comparator rule uniformly. However, the online LLM trajectories were observed before this audit, each route has one completed trajectory, and route-bootstrap intervals quantify benchmark-route variation rather than LLM stochastic variation. Confirmatory claims require the already frozen 30-replicate protocol or a prospectively collected wet-lab test.
