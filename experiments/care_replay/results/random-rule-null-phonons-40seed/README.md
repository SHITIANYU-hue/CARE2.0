# Matched Random-Rule Null: Matbench Phonons

This is the first null-control pass for the semantic-skill claim. It uses the
same target-only GP-UCB anchor, initial observations, reveal budget, and paired
seed protocol as the frozen semantic replay.

## Protocol

- Target: `real_matbench_phonons`
- Calibration: seeds `30000`-`30009` (10)
- Held-out: seeds `31000`-`31029` (30)
- Initial observations: 5
- Reveal rounds: 10
- Random replicates: 3
- LLM skill structures retained per replicate: 5
- Null transformation: preserve each skill's field set, rule count, condition
  arity, and execution hyperparameters; randomize condition values and rule
  signs.

The best random route is selected using calibration seeds only. Its held-out
comparison is therefore not a selection result.

## Result

The calibration-selected random route was
`llm_semantic_random_rule_r2_enhance_chalcogenide_phonon_response`.

| Held-out metric | Delta vs GP-UCB |
| --- | ---: |
| Final Best | -1.3163 |
| Best-so-far AUC | -0.6295 |
| Final + AUC | -1.9458 |
| Final Best win rate | 36.7% |

This is a control result, not evidence that the LLM semantic route is already
causal. It shows that a matched random rule structure did not reproduce the
positive held-out behavior in this first pass. The larger 30/100-seed and
multi-domain null runs remain part of the confirmation plan.
