# CARE 2.0 Zero-Shot Hypothesis Transfer Matrix

This report uses frozen LLM hypotheses, no target calibration, and matched random-rule nulls. Every hypothesis is retained; no target outcome is used to choose the reported winner.

## Pair-Level Results

| Source -> target | Seeds | Hypotheses | Stable positive vs mixed GP-EI | Stable negative | Best composite hypothesis |
| --- | ---: | ---: | ---: | ---: | --- |
| real_suzuki_miyaura -> real_buchwald_hartwig | 30 | 4 | 0 | 2 | `ligand_mw_influence` |
| real_suzuki_miyaura -> real_chemlex_acidamine | 10 | 5 | 0 | 0 | `acid_amine_coupling_hypothesis` |
| real_matbench_expt_gap -> real_matbench_dielectric | 10 | 5 | 0 | 1 | `transition_metal_effect` |
| real_moleculenet_esol -> real_moleculenet_freesolv_continuous | 10 | 4 | 2 | 0 | `hydrogen_bonding_effect` |
| real_moleculenet_freesolv -> real_moleculenet_lipophilicity | 10 | 5 | 0 | 0 | `impact_of_halogen_substitution_on_lipophilicity` |

## Protocol Audit

Runs: 5; frozen hypotheses: 23; pair-level stable gains: 1/5; pair-level stable harms: 2/5.
All runs have zero target calibration and zero pre-decision target outcomes: True.
Each run reports GP-UCB, mixed-kernel GP-EI, every LLM hypothesis, and a same-structure random null. The random null is a control, not a competing LLM proposal.

## Interpretation

This matrix is a zero-shot hypothesis audit, not a tuned leaderboard. A positive row is only called stable when at least one primary metric's normal 95% CI is above zero versus mixed-kernel GP-EI. The aggregate is reported at the source-target-pair level, so a few favorable hypotheses cannot be presented as universal transfer.
