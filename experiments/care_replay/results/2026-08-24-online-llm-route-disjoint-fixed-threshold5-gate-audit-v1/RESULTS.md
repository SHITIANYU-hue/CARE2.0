# Prediction-error calibration gate audit

The gate is evaluated after 3 LLM-guided target reveals. If their mean absolute prediction error exceeds the fixed global threshold, or the LLM explicitly falsifies or abandons the transfer hypothesis, target-only GP-UCB continues from the accumulated observations.

| Method | Mean AUC delta vs target GP | Route-bootstrap 95% CI | Win / tie / loss |
|---|---:|---:|---:|
| Original online LLM | -1.056 | [-3.010, +0.154] | 2 / 2 / 2 |
| Fixed global calibration gate | -1.968 | [-6.438, +0.873] | 2 / 2 / 2 |

The gate switched on 6 of 6 route-disjoint retrospective replays. It reduced the negative-transfer count against target GP from 2 to 2 and changed the equal-route mean by -0.912 AUC.

In this counterfactual replay, switching removed 42 later LLM-guided rounds, corresponding to 42 nominal proposer/critic calls. These are calls that the executable gate would avoid; they are not claimed as already realized API savings.

Against the strongest realized baseline selected post hoc, the gated controller remained -5.063 AUC on average with 1 wins, 1 ties and 4 losses. The gate is therefore a negative-transfer control, not evidence of universal baseline superiority.

## Claim boundary

The evaluation routes do not overlap the routes used to select the fixed threshold. However, these are previously completed development trajectories with known replay outcomes, so this is route-disjoint retrospective evidence rather than prospective external validation.
