# Response to the 2026-08-06 code review

## Bottom line

The review is substantially correct about the current evidence boundary. The
headline gains are reproducible, but the available matched controls do not yet
support a broad claim that post-initialization transfer or LLM reasoning caused
those gains. The implementation does execute measured source outcomes after
initialization, so the statement that there is no transfer mechanism at all is
too strong. The accurate statement is that these mechanisms exist in code but
their incremental benefit has not been established on the inspected routes.

## Accepted findings

1. **Warm-start attribution:** ChemLex to Buchwald-Hartwig and the audited
   materials direction are exactly reproduced by matched source-informed
   warm-start-only controls. They must be reported as initialization transfer,
   not continuous transfer.
2. **LLM attribution:** LLM-generated numeric patch fields are executed, but an
   LLM record alone is not evidence of an LLM gain. Mechanism and failure text
   are advisory-only. Causal LLM credit requires a matched non-LLM transfer
   control.
3. **Calibration cost:** tens of replay seeds are valid for offline model
   selection but not a practical real-lab deployment gate. Held-out evaluation
   prevents result selection bias; it does not make calibration free.
4. **Accumulation:** the current artifact does not update a shared skill library
   across completed tasks. Calling it a reusable, accumulating skill would be
   premature.
5. **Missing baselines:** target-only BO is necessary but insufficient. This
   was a valid gap in the reviewed commit. RGPE and a two-task ICM GP are now
   included wherever source and target have the same ordered feature space.

## Overstated findings

The review says post-initialization transfer is merely "adding features whose
coefficients are refit." That is incomplete. Measured source outcomes also
produce neighbor, additive, and interaction priors, change categorical kernel
geometry, and can alter the acquisition ranking through a bounded expert
mixture. These are real transfer operators. The problem is empirical
attribution: on the matched routes inspected so far, they did not improve over
the same warm start.

The review also says the LLM output is not read except for a rule direction.
That is true for the older semantic-rule path, but not for the source-outcome
kernel path. The latter executes LLM-generated scales, role multipliers, GP
exploration settings, prior strengths, calibration settings, interaction
settings, and confidence. Whether those choices improve results remains an
experimental question, now tested against a fixed data-only control.

## Changes made

- Every canonical pair now runs `source_warmstart_only`, with the same
  source-informed initial design and zero post-initialization transfer mass.
- Every canonical pair now runs `fixed_data_only_transfer`, with measured
  source outcomes and a deterministic equal-role patch but no LLM patch choices.
- Each summary automatically reports initialization effect,
  post-initialization effect, LLM patch increment, exact seed-match rate,
  transfer-active rounds, and acquisition-action changes.
- Calibration output is explicitly marked `offline_replay_model_selection` and
  `real_experiment_deployment_ready: false`.
- Each run reports the evaluated strategy count and gross policy-replay reveal
  equivalents; this is explicitly not presented as a wet-lab cost model.
- The `TransferSkill` execution contract now distinguishes executable numeric
  fields from advisory-only text and identifies the artifact as a frozen
  source-outcome transfer policy.

## Classical baseline follow-up

A classic transfer baseline was added as a separate implementation rather than
by relabeling the existing router. RGPE and conventional multi-task GP normally
assume a compatible input space, so the confirmation suite is split correctly:

- RGPE and a two-task ICM GP run on five same-space molecular/materials pairs;
- all methods use identical initial observations, reveal budgets, source
  histories, and disjoint 30/100 calibration/held-out seeds;
- heterogeneous reaction pairs remain excluded until a declared mapping or a
  heterogeneous-domain transfer method is implemented.

The result is not a blanket CARE win. Classical transfer is exceptionally
strong for Dielectric to Band Gap and harmful on several other routes. A
calibration-selected CARE/classical portfolio has positive mean AUC gains on
all five pairs, but all CARE source-transfer candidates were rejected. The
positive CARE arms are target-only fallbacks, so this result supports model
routing rather than a causal source-transfer or LLM-patch claim. See
`results/2026-08-08-classical-transfer-confirmation/`.

Work still remains on a zero-target-cost router, accumulated transfer memory,
and a declared heterogeneous-domain baseline for reaction tasks.

Primary references: [RGPE](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/18-AUTOML-RGPE.pdf),
[transfer GP for Bayesian optimization](https://proceedings.mlr.press/v151/tighineanu22a.html),
and [multi-task GP over heterogeneous input domains](https://arxiv.org/abs/2202.12636).
