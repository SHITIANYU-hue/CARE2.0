# Rule Provenance and Incumbent Ablation Notes

This note answers a narrow but important question: what are the current CARE 2.0
replay rules, where did they come from, and how should we interpret the fact
that deterministic transfer rules currently outperform the LLM variants?

## Short Answer

The current replay harness is not a reproduction of CARE 1.0 paper numbers, and
the incumbent rule is not a community-standard chemistry baseline. It is a
transparent target-only evidence baseline written for this replay harness.

That matters for interpretation. We should not describe the incumbent as
CARE 1.0, Bayesian optimization, or a literature consensus. The right wording is
closer to:

> The incumbent is a strong public-observation baseline that uses only revealed
> target observations and public candidate descriptors. Transfer gains are
> measured against this baseline, so positive gains are meaningful, but the
> baseline itself is an engineered replay control rather than a named external
> method.

## Current Incumbent Rule

For each unrevealed target candidate, the incumbent estimates a score from:

1. Smoothed mean of previously observed candidates in the same coarse group.
2. Smoothed means for each observed decision-factor value, such as ligand/base
   or molecular descriptor bins.
3. A UCB-like uncertainty bonus that favors factors with less support.
4. A lightweight prior over public continuous features `x1`, `x2`, and `x3`.

The current scoring path lives in
`experiments/care_replay/scripts/run_synthetic_suzuki.py::public_incumbent_scores`.
It does not use hidden target outcomes for unrevealed rows.

This makes the incumbent stronger than random search, but it is not unbeatable.
The 50-seed ablation below shows that simpler variants are often in the same
range, and on Lipophilicity the factor-only variants slightly exceed the full
incumbent on some metrics.

| Dataset | Mode | Final Best | Delta vs Incumbent | AUC | Delta AUC | Top-10 Hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Buchwald-Hartwig | random | 82.1728 | -4.4749 | 77.4347 | -2.5366 | 0.0600 |
| Buchwald-Hartwig | group_only | 86.1706 | -0.4771 | 82.9826 | +3.0113 | 0.0600 |
| Buchwald-Hartwig | factor_only | 86.3136 | -0.3341 | 80.0357 | +0.0644 | 0.2400 |
| Buchwald-Hartwig | factor_ucb | 86.7960 | +0.1483 | 79.5502 | -0.4211 | 0.2000 |
| Buchwald-Hartwig | no_condition_prior | 86.1992 | -0.4485 | 80.4621 | +0.4908 | 0.2000 |
| Buchwald-Hartwig | public_incumbent | 86.6477 | 0.0000 | 79.9713 | 0.0000 | 0.1600 |
| Lipophilicity | random | 85.7650 | -1.5425 | 84.2917 | -1.2743 | 0.0200 |
| Lipophilicity | group_only | 86.3200 | -0.9875 | 84.4772 | -1.0888 | 0.0000 |
| Lipophilicity | factor_only | 87.5225 | +0.2150 | 85.6342 | +0.0682 | 0.0800 |
| Lipophilicity | factor_ucb | 87.4550 | +0.1475 | 85.8572 | +0.2912 | 0.1600 |
| Lipophilicity | no_condition_prior | 87.1875 | -0.1200 | 85.3805 | -0.1855 | 0.0800 |
| Lipophilicity | public_incumbent | 87.3075 | 0.0000 | 85.5660 | 0.0000 | 0.0400 |

## Current Transfer Rule

The deterministic transfer rule is also engineered for the replay harness. It
is not copied from CARE 1.0. Its purpose is to test whether reusable evidence can
move from a source task to a target task under a strict audit boundary.

The transfer path has three pieces:

1. A manually declared role map, such as Suzuki ligand to Buchwald-Hartwig
   ligand, or MoleculeNet descriptor bin to the same descriptor bin.
2. A transfer card compiled from revealed source observations. This card stores
   role-level support, effect magnitude, confidence, and transfer weight.
3. A target-side adjustment rule. The source card can weight which target roles
   matter, but target observations decide the direction of most adjustments.

For reaction HTE tasks, the rule does not directly transfer source-local values
such as a particular ligand ID unless the field vocabulary is known to be shared.
For MoleculeNet tasks, shared descriptor bins are allowed to create conservative
value priors because the vocabulary is genuinely shared.

The gate then decides whether the transfer challenger can override the
incumbent. It checks the margin, acquisition loss, row-order stability, and
maximum bounded adjustment.

## Why LLM Transfer Is Behind the Fixed Rule

The current LLM variants do make real LLM calls. They are not mocked. However,
their role is still narrow:

- `llm_transfer_gate_v1` asks the model to propose bounded factor-level
  adjustments from the transfer card and revealed target evidence.
- `llm_audit_transfer_gate_v1` asks the model to audit an already proposed
  challenger.

The model is not yet allowed to rewrite the rule, tune thresholds, create new
role maps, or evolve a reusable skill artifact across runs. In other words, the
LLM is currently a constrained proposer/auditor inside a hand-written policy,
while the deterministic rule is already exhaustive and calibrated for this
schema. That is a plausible reason the fixed rule still wins.

The next LLM experiment should move one level up: let the model propose a
structured rule update, then evaluate that update on held-out seeds and target
pairs. Good candidates include:

- Role-map review and confidence assignment.
- Threshold tuning for minimum support, effect size, and transfer discount.
- Selection of which shared descriptor fields are safe for value priors.
- Risk policy changes for when to use strict gate versus ordinary gate.

The LLM should still not see hidden target outcomes for unrevealed candidates.
Rule proposals should be trained or selected on a small calibration split and
then evaluated on held-out seeds.

## Reporting Guidance

The current strongest evidence is that cross-domain transfer is possible and can
be large in selected settings:

- FreeSolv to Lipophilicity shows a strong shared-descriptor transfer gain.
- Suzuki-Miyaura to Buchwald-Hartwig shows a positive reaction-HTE transfer
  gain.

The baseline story now has three layers:

1. Basic replay baselines: random search and public incumbent.
2. Incumbent ablations: group-only, factor-only, factor-UCB, and no-condition
   prior variants.
3. Stronger target-only surrogate baselines: dependency-free mixed-kernel
   GP-UCB, mixed-kernel GP-EI, and kNN-UCB.

The surrogate baseline comparison sharpens the claim. FreeSolv to
Lipophilicity transfer remains stronger than the added GP-UCB / GP-EI / kNN-UCB
baselines. Suzuki-Miyaura to Buchwald-Hartwig transfer beats the public
incumbent, but current GP-UCB is stronger than the additive transfer gate on
that target. The reaction HTE story should therefore distinguish between
ordinary transfer-over-incumbent and acquisition-level transfer-over-GP-UCB.

The first hybrid experiment uses GP-UCB itself as the incumbent acquisition and
lets transfer cards apply bounded acquisition adjustments. This gives a small
positive signal for FreeSolv to Lipophilicity when shared descriptor value
priors are enabled. It does not yet beat GP-UCB on Buchwald-Hartwig final best,
although it slightly improves AUC. The next rule work should therefore focus on
how transfer modifies the acquisition function, not just whether transfer can
override a hand-written incumbent.

The transfer-weighted kernel experiment is the first acquisition-level positive
result against GP-UCB. It uses the same mixed-kernel GP-UCB acquisition, but
reweights categorical kernel fields using source-to-target transfer-card role
confidence. This does not transfer hidden outcomes and does not directly select
candidate ids. On Suzuki-Miyaura to Buchwald-Hartwig, `scale=1.5` improves
50-seed final best from 91.1145 to 91.4146 and AUC from 82.9700 to 83.2862.
On FreeSolv to Lipophilicity, the same moderate scale improves final best from
88.4775 to 88.5650 and AUC from 86.4578 to 86.6808. The gain is small, and
larger or smaller scales can hurt, so this should be framed as evidence that
skill optimization can beat the stronger baseline in selected settings, not as
a solved optimizer.

At the same time, negative or flat transfer should remain in the internal
report. It helps make the story credible: CARE 2.0 is not claiming universal
transfer. The current claim should be that the platform can express and test
transfer, that some real transfer directions are already positive, and that the
next technical work is calibration of acquisition-level skill parameters and
LLM-driven rule evolution.
