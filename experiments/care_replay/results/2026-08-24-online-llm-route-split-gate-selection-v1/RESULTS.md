# Route-split online-LLM controller selection audit

The controller was selected using only the 11 training-route metrics. The six evaluation-route outcomes were previously known, but they do not enter the selection key; this is a leakage-controlled retrospective audit rather than a blinded holdout.

Selected policy: `bounded_authority_r1`. It permits 1 online LLM-guided reveal and then continues with target-only GP-UCB from all accumulated observations.

| Panel / method | Mean AUC delta vs target GP [95% route-bootstrap CI] | Win / tie / loss |
|---|---:|---:|
| Training / selected controller | +1.188 [+0.000, +2.951] | 2 / 9 / 0 |
| Evaluation / full online LLM | -1.056 [-3.010, +0.154] | 2 / 2 / 2 |
| Evaluation / fixed round-3 threshold-5 gate | -1.968 [-6.438, +0.873] | 2 / 2 / 2 |
| Evaluation / selected controller | +0.612 [-0.442, +1.793] | 2 / 3 / 1 |

On the six route-disjoint trajectories, the selected controller changed AUC by +1.668 versus the original full online-LLM trajectory. It still had 1 loss against target-only GP-UCB, so this is evidence of improved robustness, not universal positive transfer.

## Development-route selection stability

Leaving out each development route in turn selected 2 distinct policies. The most frequent policy was `bounded_authority_r1` in 90.9% of folds. Applied to each omitted route, the re-selected policies averaged +1.141 AUC with 2 / 8 / 1 wins / ties / losses versus target-only GP-UCB.

## Claim boundary

The controller is selected using only the 11 training routes and then evaluated on six route-disjoint trajectories. All trajectories are previously completed retrospective replays, so this is a leakage-controlled development audit rather than prospective or external validation.
