# Bounded online-LLM decision-impact pilot

The pilot completed 6 real online trajectories across 6 routes. It is not a confirmatory analysis.

| Diagnostic | Result |
|---|---:|
| LLM selected GP rank one | 6 / 6 |
| LLM overrode GP rank one | 0 / 6 |
| LLM kept source transfer active | 3 / 6 |
| Critic changed proposer choice | 0 / 6 |
| Online LLM mean AUC delta vs same-start GP | +0.0000 |
| Online LLM win / tie / loss | 0 / 6 / 0 |
| Complete-system mean AUC delta vs fixed initial design | -0.0087 |
| Complete-system win / tie / loss | 3 / 1 / 2 |
| Total tokens | 99,243 |
| Mean tokens per trajectory | 16,540.5 |
| Nominal later LLM calls avoided by bounded authority | 108 |

## Interpretation

Every valid LLM call produced an auditable hypothesis and critic decision, but every executable decision selected the target-only GP-UCB rank-one candidate. The online module therefore had zero functional decision impact in this pilot.

This result separates model participation from causal contribution. The next method-development question is not whether the LLM can emit reasoning, but whether a frozen semantic skill or challenger policy changes actions and improves held-out outcomes.

## Claim boundary

One completed trajectory per route is an operational pilot only. The frozen protocol disables confirmatory inference until all declared replicates finish; no confidence interval or superiority claim is made.
