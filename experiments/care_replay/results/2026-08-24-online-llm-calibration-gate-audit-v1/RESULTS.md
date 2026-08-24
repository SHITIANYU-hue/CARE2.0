# Prediction-error calibration gate audit

The gate is evaluated after three LLM-guided target reveals. If the first-three mean absolute prediction error exceeds a leave-one-route-out threshold, or the LLM explicitly falsifies or abandons the transfer hypothesis, target-only GP-UCB continues from the accumulated observations.

| Method | Mean AUC delta vs target GP | Route-bootstrap 95% CI | Win / tie / loss |
|---|---:|---:|---:|
| Original online LLM | +1.753 | [+0.406, +3.326] | 6 / 2 / 3 |
| Cross-validated calibration gate | +2.036 | [+0.425, +3.901] | 5 / 4 / 2 |

The gate switched on 8 of 11 held-out route evaluations. It reduced the negative-transfer count against target GP from 3 to 2 and changed the equal-route mean by +0.283 AUC.

Against the strongest realized baseline selected post hoc, the gated controller remained -0.267 AUC on average with 2 wins, 2 ties and 7 losses. The gate is therefore a negative-transfer control, not evidence of universal baseline superiority.

## Claim boundary

This is a retrospective leave-one-route-out audit on one trajectory per route. The held-out route never contributes to its threshold selection, and the fallback uses only observations available by the gate round, but prospective repeated confirmation is still required.
