# Cross-domain LLM Semantic Skill Evaluation

Date: 2026-07-21

This snapshot asks a stricter CARE 2.0 question: can an LLM compile source-task
evidence and a public target schema into an executable skill that improves a
strong target-only Bayesian-optimization policy on real scientific data?

## System

One offline CommonStack `openai/gpt-4o-mini` call proposes a small library of
semantic skills. Each skill contains public field/value rules, coefficient
priors, and an exploration schedule. A strict compiler rejects unknown fields
and values. During replay, the rules become binary features in a Bayesian ridge
surrogate fitted only to target outcomes revealed in earlier rounds. Its rank
is blended with target-only GP-UCB and mixed-kernel GP-EI.

The sequence is:

`source evidence + public target schema -> LLM skill library -> strict compiler -> calibration -> frozen skill -> held-out replay`

The LLM is not called per seed or per round. Five successful generation calls
were made in this study (13,324 prompt tokens and 9,136 completion tokens); the
replay itself makes zero model calls.

## Protocol

- Real datasets: Dreher-Doyle Buchwald-Hartwig HTE, Matbench experimental band
  gap, ChemLex Acid-Amine wetlab, and the earlier MoleculeNet snapshot.
- Strong target-only baselines: GP-UCB, mixed-kernel GP-EI, and a target
  acquisition portfolio.
- Development: 30 calibration seeds and 50 held-out development seeds.
- Confirmation: the selected skill is frozen, then evaluated on 500 fresh
  seeds (`14000-14499`). The 50 confirmation calibration seeds only select the
  target-only comparison anchor.
- Statistics: paired seed deltas with normal 95% confidence intervals.
- Leakage boundary: the LLM sees source summaries and public target field/value
  counts, never target outcomes or candidate identifiers. Each replay round
  fits only outcomes revealed in prior rounds.

The comparison baseline in each confirmation is the strongest of the three
target-only policies on that held-out result table. This is descriptive and
conservative, but the intervals are not adjusted for multiple comparisons.

## Cross-domain Result

The molecular rows come from the companion
[`../2026-07-21-low-cost-llm-transfer`](../2026-07-21-low-cost-llm-transfer/README.md)
snapshot. Reaction and materials rows use the semantic compiler introduced
here.

| Domain / target | Frozen LLM skill | Seeds | Strongest target-only baseline | Delta final best | Delta AUC | Delta top-10 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Molecular / FreeSolv | exploration-first kernel skill | 500 | target portfolio | +1.4326 `[+0.1224, +2.7428]` | +1.3840 `[+0.5789, +2.1892]` | +0.0580 `[+0.0076, +0.1084]` |
| Molecular / Lipophilicity | same acquisition schedule | 300 | target portfolio | +0.4850 `[-0.0602, +1.0302]` | +0.6066 `[+0.1298, +1.0835]` | +0.0400 `[-0.0031, +0.0831]` |
| Reaction / Buchwald-Hartwig | `high_mw_ligand_effect` | 500 | mixed-kernel GP-EI | +3.1208 `[+2.2012, +4.0403]` | +2.8588 `[+1.9602, +3.7574]` | +0.1760 `[+0.1276, +0.2244]` |
| Materials / Matbench band gap | `counter_transition_metal` | 500 | mixed-kernel GP-EI | +7.2240 `[+4.5309, +9.9171]` | +5.9302 `[+4.1592, +7.7011]` | +0.1640 `[+0.1175, +0.2105]` |

The result now contains statistically positive held-out examples in molecular
property, reaction HTE, and materials-property tasks. It does not establish
that one universal skill works on every dataset: the platform selected
different executable skill forms and ChemLex remained negative.

## What Drives The New Gains

The schedule-only ablation keeps the LLM's GP-UCB/EI mixture and exploration
schedule but removes its semantic surrogate. The comparison below is full
semantic skill minus schedule-only on exactly the same 500 seeds.

| Target | Delta final best | Delta AUC | Delta top-10 |
| --- | ---: | ---: | ---: |
| Buchwald-Hartwig | +2.7453 `[+1.8877, +3.6029]` | +2.8244 `[+2.0539, +3.5949]` | +0.1880 `[+0.1377, +0.2383]` |
| Matbench band gap | +6.6775 `[+4.0058, +9.3492]` | +5.0652 `[+3.3546, +6.7758]` | +0.1480 `[+0.1018, +0.1942]` |

The main reaction and materials gains therefore require the semantic rule
features; acquisition tuning alone is not sufficient.

The no-prior ablation keeps the same LLM-proposed rule features and online
fitting, but sets the LLM's initial rule-coefficient direction to zero. The
comparison is full semantic skill minus no-prior on the same seeds.

| Target | Delta final best | Delta AUC | Delta top-10 |
| --- | ---: | ---: | ---: |
| Buchwald-Hartwig | +1.6546 `[+0.9286, +2.3807]` | +1.9929 `[+1.3724, +2.6134]` | +0.0940 `[+0.0449, +0.1431]` |
| Matbench band gap | +2.0540 `[+0.4165, +3.6915]` | +1.3780 `[+0.4065, +2.3495]` | +0.0340 `[+0.0046, +0.0634]` |

Both components matter. The LLM-proposed feature partition remains useful
after its prior is removed, while the LLM's proposed direction adds a separate
significant increment. Online target evidence is still required to update the
coefficients and uncertainty at every round.

## Negative Result: ChemLex

The first ChemLex calibration rejected all five LLM skills and fell back to
GP-UCB. On its independent 50-seed development split, every semantic skill had
lower AUC than mixed-kernel GP-EI. For example, `acid_length_effect` produced an
AUC delta of -5.0042 `[-9.4717, -0.5366]`.

A second prompt added 37 public motif fields: carbonyl/amide/nitrile flags,
N/O bins, formal charge, aromatic heteroatoms, and coupling-reagent families.
Calibration selected `acid_ring_token_skill`, but on a new 50-seed held-out
split it produced -2.9792 final best `[-8.2129, +2.2545]` and -0.2556 AUC
`[-5.2182, +4.7071]` versus GP-UCB. No confirmatory 500-seed claim was made.

The failure remains useful: richer string-level motifs still do not capture the
substrate/reagent interactions in this wetlab task, and the held-out check
prevents an unstable calibration win from becoming a deployed claim. A learned
reaction representation is needed before retrying ChemLex.

## Interpretation Boundary

This is evidence for LLM-assisted representation and search-policy transfer,
not a direct leaderboard comparison with an external autonomous scientist.
The controlled advantages over common LLM-scientist demonstrations are the
strong target-only BO baselines, frozen pre-held-out decisions, 500 paired
seeds, explicit negative-transfer fallback, and complete traces. A direct
external-system claim would require running those systems on the same finite
pools, budgets, and seed splits.

The selected fields occur in the source-to-target role maps, but this snapshot
does not isolate source evidence from the LLM's pretrained domain knowledge.
That is a remaining transfer-causality ablation.

## Reproducibility Files

- `pair_summary.csv`: headline comparisons and paired ablations.
- `paired_semantic_ablation.json`: direct same-seed semantic ablations.
- `run_manifest.json`: model usage, seed ranges, selected skills, and files.
- `model_calls/`: prompts, raw responses, normalized skills, and API traces.
- `raw_metrics/`: every per-mode, per-seed metric table.
- `raw_summaries/`: calibration, selection, held-out, and CI summaries.
- `audit_archives/`: compressed per-round traces, including matched rules,
  fitted coefficients, uncertainty, target anchors, and selected candidates.

No API credential is stored in this directory.
