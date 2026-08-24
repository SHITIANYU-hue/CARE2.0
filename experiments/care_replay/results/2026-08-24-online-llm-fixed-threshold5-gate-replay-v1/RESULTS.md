# Prediction-error calibration gate audit

The gate is evaluated after three LLM-guided target reveals. If the first-three mean absolute prediction error exceeds the fixed global threshold, or the LLM explicitly falsifies or abandons the transfer hypothesis, target-only GP-UCB continues from the accumulated observations.

| Method | Mean AUC delta vs target GP | Route-bootstrap 95% CI | Win / tie / loss |
|---|---:|---:|---:|
| Original online LLM | +1.753 | [+0.406, +3.326] | 6 / 2 / 3 |
| Fixed global calibration gate | +2.135 | [+0.515, +4.001] | 6 / 3 / 2 |

The gate switched on 10 of 11 retrospective route replays. It reduced the negative-transfer count against target GP from 3 to 2 and changed the equal-route mean by +0.381 AUC.

In this counterfactual replay, switching removed 70 later LLM-guided rounds, corresponding to 140 nominal proposer/critic calls. These are calls that the executable gate would avoid; they are not claimed as already realized API savings.

Against the strongest realized baseline selected post hoc, the gated controller remained -0.168 AUC on average with 3 wins, 1 ties and 7 losses. The gate is therefore a negative-transfer control, not evidence of universal baseline superiority.

## Claim boundary

This is a retrospective replay of one previously observed trajectory per route under a single fixed threshold. It validates executable semantics and motivates the frozen repeated protocol, but it is not prospective or external validation.
