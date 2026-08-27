# CARE 2.0 photocatalysis pilot

## What was added

This pilot adds the public 10-dimensional photocatalytic hydrogen-evolution
benchmark to the CARE finite-pool replay harness. The pinned table contains
1,109 formulations and a measured hydrogen-evolution-rate target. The adapter
retains all ten continuous variables as normalized numeric features and adds
coarse, outcome-blind bins for LLM-readable hypotheses. It does not load the
upstream pickle model.

This is a real photocatalysis benchmark, but it is not the exact
photocatalytic nitroarene-to-amine Buchwald-Hartwig task discussed by the team.
The latter remains a separate, mechanism-near follow-up.

## Frozen protocol

- Target-only baselines: 5 random initial observations, 20 additional reveals,
  30 seeds.
- LLM generation: one successful Claude Opus 5 call, target schema only, no
  target outcomes in the prompt. Six skills were frozen before replay.
- Zero-shot check: 30 new seeds, no target calibration, with three matched
  random-rule replicas per skill.
- Calibrated strategy check: 30 calibration seeds select among frozen routes;
  50 disjoint held-out seeds provide the reported comparison.
- The strongest accepted LLM route uses the
  `silicate_dispersant_optimum_window` skill only to choose the initial batch;
  all later choices use the target-only GP acquisition.

## Results

The target-only baseline run shows that the dataset is learnable under the
shared budget. Mean final best HER was 20.61 for random selection, 20.82 for
the transparent incumbent, 24.15 for GP-UCB, 22.51 for GP-EI, and 22.25 for
kNN-UCB. GP-UCB reached a finite-pool top-10 candidate in 63% of the 30 runs,
versus 23% for random selection.

The zero-shot LLM direct-prior policies were negative against GP-UCB. The best
of the six still had a mean AUC delta of -1.35 µmol h⁻¹. This falsifies the
idea that a strong language model's static chemistry prior should steer every
round of this task.

Calibration instead selected an LLM warm-start. On 50 disjoint held-out
seeds, the selected route improved mean final best HER by 0.55 µmol h⁻¹ and
mean best-so-far AUC by 0.74 µmol h⁻¹ versus GP-UCB. Win rates were 54% and
64%, respectively. Both 95% confidence intervals cross zero, so this is a
positive but not statistically confirmed signal. It must not be reported as
a significant improvement.

## Why the LLM helped only at warm-start

Several Opus rules are directionally consistent with the full table: dye-added
formulations average 1.09 µmol h⁻¹ versus 11.60 for dye-free formulations;
surfactant-added formulations average 3.91 versus 11.22 for surfactant-free
formulations; and no-added-NaOH formulations average 5.02 versus 14.65 in the
medium-NaOH bin. These broad rules help avoid poor regions in the initial
batch.

They are too coarse for continuous steering. The global best formulation, for
example, lies in the low L-cysteine bin even though that bin has a poor pool
average. A static rule that penalizes the whole bin can therefore suppress an
important local exception. Once a few target outcomes exist, the numeric GP
is better suited to resolve these local interactions. The result supports a
bounded role for the LLM: propose an interpretable initial hypothesis, then
hand control to the data-driven optimizer.

## Files

- `llm_target_only_skills_opus5.json`: first Opus response; retained failure
  trace because the response exceeded the token budget and produced invalid
  JSON.
- `llm_target_only_skills_opus5_v2.json`: successful prompt, raw response,
  usage metadata, and six normalized executable skills.
- `zero_shot_opus5_30seed/`: zero-shot metrics, summary, and compressed full
  per-round audits.
- `calibrated_warmstart/`: calibration/held-out summary, metrics, and paired
  representative LLM/GP traces for seed 63000.
- `rule_associations.csv`: post-hoc descriptive associations between each
  frozen LLM rule and the full finite pool. These associations were not used
  to generate or select the rules.
- `photocatalysis_pilot_results.png` and `.pdf`: figures generated from the
  recorded numeric results by `plot_results.py`.

## Sources and boundary

Dataset and benchmark methodology:
<https://github.com/Ablatif6c/llm-closed-loop-experiments> and
<https://doi.org/10.1039/D5DD00520E>.

Mechanism-near nitroarene photo-flow reference for the next phase:
<https://doi.org/10.1039/D5RE00131E>.

The current pilot establishes infrastructure and a bounded warm-start signal.
It does not establish transfer to photocatalytic nitroarene C-N coupling and
does not reproduce the upstream BORA continuous-surrogate benchmark.
