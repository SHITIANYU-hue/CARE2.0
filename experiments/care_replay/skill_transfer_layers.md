# CARE 2.0 Skill Transfer Layers

This note defines the CARE 2.0 direction behind the current transfer experiments:
skill transfer should not be limited to acquisition-level score adjustment.

The current `transfer_weighted_gp_kernel` experiment is useful because it gives a
measurable positive result over GP-UCB, but it is only one layer of a broader
transferable skill artifact.

## Layer 1: Representation Transfer

This layer answers: what can be mapped from source to target?

Examples:

- Suzuki `ligand`, `reagent`, `reactant_1`, `solvent` -> Buchwald-Hartwig
  `ligand`, `base`, `aryl_halide`, `additive`.
- FreeSolv SMILES descriptor bins -> Lipophilicity SMILES descriptor bins.

This layer should avoid transferring dataset-local labels unless the vocabulary
is explicitly shared. For example, `L01` in one reaction table is not assumed to
mean the same ligand as `L01` in another table.

## Layer 2: Mechanism Or Hypothesis Transfer

This layer answers: what scientific claim is being reused?

Examples:

- Certain reaction roles are more informative for cross-coupling search.
- Shared molecular descriptors can define reusable property-search geometry.
- A source domain may identify which factors are high-risk for negative
  transfer, even when exact factor values cannot be reused.

This layer is where the skill becomes more than a numerical tweak. It should
record the hypothesis, expected scope, and known failure modes.

## Layer 3: Model Transfer

This layer answers: how does the source knowledge change the target model?

The current implemented example is:

- Transfer-card role confidence reweights categorical fields in the mixed-kernel
  GP-UCB surrogate.

Other possible bindings:

- feature transforms,
- embedding priors,
- posterior mean priors,
- uncertainty priors,
- task-specific kernels,
- descriptor selection.

## Layer 4: Acquisition Transfer

This layer answers: how does the skill affect the next experiment choice?

Examples:

- reweight GP-UCB or GP-EI,
- change exploration strength,
- filter candidate windows,
- rerank only inside a high-quality candidate set,
- apply bounded transfer value priors.

This is the layer we can measure most directly with replay metrics, but it
should be downstream of representation, mechanism, and model choices.

## Layer 5: Gate And Risk Transfer

This layer answers: when should the transfer be trusted?

Examples:

- minimum source support,
- minimum target confirmation,
- role confidence threshold,
- allowed acquisition loss,
- negative-transfer detection,
- stricter gate after bad interventions.

This layer is needed because the current experiments show both positive and
negative transfer. CARE 2.0 should learn when to reuse a skill, not just how to
reuse it.

## Layer 6: Workflow Transfer

This layer answers: what operational procedure is reusable?

Examples:

- finite-pool replay protocol,
- budget and seed setup,
- audit log schema,
- raw data boundary,
- batch-selection constraints,
- wetlab protocol constraints,
- report-ready result snapshot format.

Workflow transfer is important if CARE 2.0 is meant to become a general platform
for chemistry, materials, and drug-discovery tasks rather than a single replay
script.

## Current Concrete Artifact

The first concrete transferable skill artifact is:

- Skill: `skill.transfer-weighted-gp-kernel`
- Result snapshot:
  `experiments/care_replay/results/2026-07-04-transfer-weighted-kernel/`
- Main positive setting: `transfer_weighted_gp_ucb_scale_1p5`

Current 50-seed results:

| Transfer | Baseline | Transfer Skill | Delta Final | Delta AUC |
| --- | ---: | ---: | ---: | ---: |
| Suzuki-Miyaura -> Buchwald-Hartwig | 91.1145 / 82.9700 | 91.4146 / 83.2862 | +0.3001 | +0.3162 |
| FreeSolv -> Lipophilicity | 88.4775 / 86.4578 | 88.5650 / 86.6808 | +0.0875 | +0.2230 |

The result should be framed conservatively: it is a small positive signal that
skill transfer can optimize a strong acquisition baseline. It is not yet a
large, universal transfer advantage.

## Next Step

The next technical step is to calibrate transferable skill parameters rather
than hand-picking them. Candidate parameters include transfer scale, field
weights, source support, role-confidence discount, posterior mean versus
uncertainty binding, candidate filtering radius, and gate thresholds.

LLM participation should move toward proposing structured skill artifacts and
parameter priors. The calibration loop and gate should decide whether those
proposals are reusable.
