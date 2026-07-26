# Matched random-rule + warm-start null: ESOL

This is a control for the claim that a semantic LLM rule is responsible for
the gain. It preserves the LLM skill shape, randomizes rule values and signs,
then also runs the matched LLAMBO-style warm-start. The route is selected on
10 calibration seeds and evaluated on 30 disjoint held-out seeds with 3 random
replicates, 5 skills, 5 initial observations, and 10 reveal rounds.

The selected null was
`llambo_warmstart_random_rule_r1_counter_hypothesis_low_hbond_donors`.
Against GP-UCB on held-out seeds:

| Metric | Delta | Approx. 95% CI |
| --- | ---: | ---: |
| Final best | +1.1143 | `[+0.374, +1.854]` |
| Best-so-far AUC | +2.1257 | `[+0.998, +3.254]` |
| Final best win rate | 56.7% | |

This is a control result, not evidence that random rules are scientifically
meaningful. It shows that warm-start and initialization geometry can explain
part of an apparent gain against GP-UCB. Any CARE claim must therefore compare
against matched warm-start controls and the strongest target-only acquisition
baseline, not only against random initialization.
