# Low-cost LLM Skill Transfer Evaluation

Date: 2026-07-21

This snapshot tests a narrower, falsifiable CARE 2.0 claim: can an LLM generate
an executable optimization skill once, freeze it before held-out evaluation,
and reuse it across real finite-pool scientific tasks without calling the LLM
for every seed or revealing hidden target outcomes?

## Protocol

- Skill generator: CommonStack `openai/gpt-4o-mini` in native JSON mode.
- LLM calls: 12 successful generation/refinement calls in total.
- Recorded usage: 32,575 prompt tokens and 15,807 completion tokens.
- Estimated generation cost: USD 0.0144 at USD 0.15/M input and USD 0.60/M
  output. Replay itself makes zero model calls per seed.
- Selection boundary: calibration seeds may choose a target-only anchor and an
  LLM patch. The chosen patch is frozen before independent held-out seeds.
- Strong target-only baselines: GP-UCB, mixed-kernel GP-EI, and a target
  acquisition portfolio. Each result is compared with the strongest of these
  baselines on the reported held-out split.
- Uncertainty: paired seed deltas with normal 95% confidence intervals.
- Leakage boundary: the LLM sees public schemas, source summaries, and transfer
  cards. It does not see unrevealed target outcomes or candidate identifiers.

The API key is supplied through an environment variable and is not present in
any record, trace, audit, or manifest in this directory.

## Main Confirmatory Result

The strongest run freezes the cheap-model `exploration_patch_high_beta` skill
generated from ESOL -> FreeSolv, disables the source-value prior, and executes
the transferred kernel/acquisition policy on 500 fresh held-out seeds.

| Target | Comparison | Held-out seeds | Delta final best | Delta AUC | Delta top-10 |
| --- | --- | ---: | ---: | ---: | ---: |
| FreeSolv | LLM transfer vs strongest target portfolio | 500 | +1.4326 `[+0.1224, +2.7428]` | +1.3840 `[+0.5789, +2.1892]` | +0.0580 `[+0.0076, +0.1084]` |

All three confidence intervals are positive. This is the clearest current
evidence that a frozen LLM-generated search skill can outperform the strong
target-only portfolio on a real molecular-property replay.

## What Produced The Gain

The patch changes two parts of GP search:

1. It transfers a source-informed categorical-kernel role map.
2. It uses an exploration-first beta schedule that settles later in replay.

The same-seed ablation separates these effects.

| Comparison on FreeSolv | Delta final best | Delta AUC | Delta top-10 |
| --- | ---: | ---: | ---: |
| Target-only LLM schedule vs strongest target portfolio | +0.9189 `[-0.3605, +2.1984]` | +0.9242 `[+0.1028, +1.7455]` | +0.0700 `[+0.0191, +0.1209]` |
| Transfer kernel vs target-only LLM schedule | +0.5137 `[-0.4796, +1.5069]` | +0.4598 `[-0.1220, +1.0417]` | -0.0120 `[-0.0504, +0.0264]` |

The defensible causal reading is therefore limited. The LLM-generated
acquisition schedule is useful; the transferred kernel has a positive mean
increment, but that increment is not yet statistically confirmed. The exact
SMILES source-value prior and its cold-start variant did not add value in the
paired ablations, so the main confirmation disables them.

## Frozen-skill Generalization

To test reuse rather than pair-specific prompt tuning, the exact same cheap LLM
schedule was frozen and applied to four new targets with no new LLM call and no
source prior or transfer kernel.

| Target | Domain | Seeds | Strongest baseline | Delta final best | Delta AUC | Readout |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Lipophilicity | molecular property | 300 | target portfolio | +0.4850 `[-0.0602, +1.0302]` | +0.6066 `[+0.1298, +1.0835]` | significant AUC gain |
| Buchwald-Hartwig | reaction HTE | 300 | GP-UCB | -0.5372 `[-1.5996, +0.5252]` | -0.5316 `[-1.4105, +0.3472]` | no gain |
| Matbench band gap | materials | 300 | target portfolio | -1.7554 `[-4.6500, +1.1392]` | -0.1885 `[-1.9272, +1.5502]` | no gain |
| ChemLex acid-amine | reaction wet lab | 300 | GP-UCB | -0.3197 `[-2.2451, +1.6057]` | +0.6314 `[-0.9904, +2.2533]` | inconclusive |

Together with the FreeSolv confirmation, the schedule improves AUC on two
independent molecular-property targets. It does not yet generalize across the
reaction and materials domains. The present result supports domain-level
generalization, not universal cross-domain transfer.

## Negative and Boundary Results

- A stronger GPT-5.6 patch did not confirm on FreeSolv -> Lipophilicity: its
  200-seed final and AUC deltas were not significant.
- Suzuki -> Buchwald-Hartwig semantic-descriptor transfer was neutral/negative
  under the frozen selector.
- Suzuki -> ChemLex and dielectric -> band-gap patches did not beat the
  strongest target-only baseline reliably.
- Calibration often rejected all LLM patches and fell back exactly to the
  target-only anchor. This is intended negative-transfer protection.
- The target-calibrated source prior and exact-identity cold start were not the
  source of the main gain.

## Reproducibility Files

- `pair_summary.csv`: compact paired comparisons and confidence intervals.
- `run_manifest.json`: datasets, seed ranges, model usage, feature flags, and
  archive inventory.
- `model_calls/`: complete prompts, parsed responses, normalized patches, usage,
  and API event traces. No credentials are stored.
- `raw_metrics/`: every per-mode, per-seed metric table used in this study.
- `raw_summaries/`: selector and calibration summaries from development runs.
- `audit_archives/`: compressed per-round audit logs for the main confirmation,
  target-schedule ablation, universal-schedule runs, and boundary studies.

## Conclusion

This round establishes a useful but narrower result than universal CARE 2.0
transfer. A low-cost LLM can generate an auditable acquisition skill once, and
that skill shows reproducible molecular-property generalization while costing
about 1.4 cents to generate. The next experiment should learn a domain router
from calibration evidence and improve the scientific representation; prompt
changes alone are unlikely to make the same schedule work for HTE and materials.
