# Matched Random-Rule Null: MoleculeNet ESOL

This is a first cross-domain null-control pass using the same protocol as the
phonons control: 10 calibration seeds, 30 held-out seeds, 3 random replicates,
5 frozen skill structures, 5 initial observations, and 10 reveal rounds.

The calibration-selected random route was
`llm_semantic_random_rule_r1_preserve_low_rotatable_bond_structures`.

| Held-out metric | Delta vs GP-UCB |
| --- | ---: |
| Final Best | +0.4924 |
| Best-so-far AUC | +0.2297 |
| Final + AUC | +0.7221 |
| Final Best win rate | 36.7% |

The positive mean is not a stable win: the paired win rate is below 50% and
the small first-pass sample is not a significance claim. This result is kept
as a warning that a random null can sometimes produce a positive mean on a
finite seed slice; larger held-out runs are required before attributing an
ESOL gain to LLM content.
