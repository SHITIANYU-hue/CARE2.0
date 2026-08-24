# Bounded online-LLM authority audit

The controller hands execution to target-only GP-UCB after 1 LLM-guided target reveal while preserving all observations accumulated so far.

| Method | Mean AUC delta vs target GP | Route-bootstrap 95% CI | Win / tie / loss |
|---|---:|---:|---:|
| Original online LLM | -1.056 | [-3.010, +0.154] | 2 / 2 / 2 |
| Bounded-authority controller | +0.612 | [-0.442, +1.793] | 2 / 3 / 1 |

The gate switched on 6 of 6 route-disjoint retrospective replays. It reduced the negative-transfer count against target GP from 2 to 1 and changed the equal-route mean by +1.668 AUC.

In this counterfactual replay, switching removed 54 later LLM-guided rounds, corresponding to 54 nominal proposer/critic calls. These are calls that the executable gate would avoid; they are not claimed as already realized API savings.

Against the strongest realized baseline selected post hoc, the gated controller remained -2.483 AUC on average with 1 wins, 1 ties and 4 losses. The gate is therefore a negative-transfer control, not evidence of universal baseline superiority.

## Claim boundary

The evaluation routes do not overlap the routes used to select the fixed one-round authority limit. However, these are previously completed development trajectories with known replay outcomes, so this is route-disjoint retrospective evidence rather than prospective external validation.
