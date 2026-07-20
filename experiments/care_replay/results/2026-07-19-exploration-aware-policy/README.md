# Live LLM exploration and transfer follow-up

This report records the July 19 live-model experiments. The model was called
through CommonStack as `openai/gpt-5.6-sol`. API keys are not stored in the
repository. All reported target outcomes come from finite-pool replay after a
candidate is selected; prompts never contain unrevealed target values.

## What changed

The implementation now includes:

1. compact prompts containing only public decision fields and revealed target
   evidence;
2. explicit `explore`, `exploit`, and `avoid` intents, counter-hypotheses, and
   evidence-calibrated confidence;
3. response-shape repair for valid JSON nested under `decision_summary`;
4. reproducible held-out seed ranges through `--seed-start`;
5. LLM-generated kernel-skill portfolios with target-only scale-0 anchors;
6. a calibration selector with a bounded practical-equivalence margin;
7. an online transfer router that requires at least 10 target observations and
   online quality of at least 0.15 before a transfer skill may change the
   target-only candidate.

The legacy modes remain available as ablations.

## 1. Per-round LLM exploration policy

The exploration policy was calibrated on seeds 0-2, then frozen and evaluated
on seeds 3-7. Each task used 8 initial observations and 4 live LLM calls per
seed. Sixty calls were made in the frozen evaluation; three returned no JSON
because the 1,000-token completion budget was consumed by model reasoning.

| Target | Delta Final vs incumbent | Delta AUC | Interpretation |
| --- | ---: | ---: | --- |
| Buchwald-Hartwig | +0.0328 | +0.3546 | Essentially tied |
| ChemLex acid-amine | 0.0000 | 0.0000 | Gate rejected every change |
| FreeSolv | -1.3933 | -0.6517 | Small negative transfer |

The large Buchwald-Hartwig gain seen on calibration seeds did not reproduce on
held-out seeds. This arm therefore does not establish generalization.

## 2. LLM-generated transfer skills

For each source-target pair, one LLM call saw source transfer-card evidence and
the public target schema, then generated eight kernel-skill patches. A patch
specifies role weights, transfer scales, a GP-UCB beta schedule, optional source
prior calibration, confidence, and a failure condition. Target outcomes were
not shown during skill generation.

Development runs exposed two problems with choosing one fixed patch:

- `Suzuki -> BH`: the generated `safe_multiscale_anchor` patch was positive on
  held-out seeds versus GP-UCB (`+1.5301` Final, `+0.7862` AUC), but a naive
  calibration selector chose a different patch on a 0.06-point calibration
  tie.
- `Suzuki -> ChemLex`: `anchored_multiscale` was positive versus GP-UCB
  (`+8.3512` Final, `+3.0222` AUC), while an overly wide one-standard-error set
  selected a weaker high-confidence patch.
- `FreeSolv -> Lipophilicity`: the selected fixed patch was negative versus
  GP-UCB (`-2.5625` Final, `-1.6992` AUC).
- `Dielectric -> band gap`: the selected fixed patch was strongly negative
  (`-14.4376` Final, `-10.3535` AUC).

These are development results, not final evidence. They show that the LLM can
generate useful patches, but fixed calibration selection is not reliable
enough across domains.

## 3. Frozen online router

The final router was frozen after the development analysis above and evaluated
on new seeds 30-49. It starts from an equal-rank GP-UCB/GP-EI target portfolio.
Transfer can alter a selection only after target warm-up and only when
prequential target evidence supports at least one LLM-generated skill.

| Pair | Strongest target-only | Delta Final | Delta AUC | Router behavior |
| --- | --- | ---: | ---: | --- |
| Suzuki -> BH | GP-UCB | -2.0346 | -1.3248 | Fell back to target portfolio |
| Suzuki -> ChemLex | Target portfolio | 0.0000 | 0.0000 | Exact safe fallback |
| FreeSolv -> Lipophilicity | GP-EI | -0.0813 | -0.6570 | Near fallback; one small deviation |
| Dielectric -> band gap | GP-EI | -2.3375 | -0.2414 | Exact portfolio fallback, but EI was stronger |

Against GP-UCB alone, the router was positive on ChemLex (`+3.0790` Final,
`+3.4142` AUC) and materials (`+1.1438`, `+0.9281`). Those numbers should not be
presented as wins against the strongest target-only baseline. None of the
20-seed confidence intervals establishes a statistically significant advantage
over the strongest target-only method.

## Current conclusion

The live calls verify that an LLM can turn source evidence into structured,
executable transfer skills spanning reaction optimization, molecular
properties, and materials. The online router substantially reduces the severe
negative transfer observed with a fixed skill. The stronger claim, that LLM
transfer consistently improves over the best target-only optimizer across
domains, is not yet supported.

The next technical priority is target-anchor routing: select GP-UCB, GP-EI, or
their portfolio from target-only prequential evidence before adding transfer.
Only after that anchor is competitive should the LLM skill receive residual
transfer mass. Larger frozen evaluations should use saved LLM records so model
generation is not repeated.

## Reproducibility files

`live_policy/` contains held-out summaries, per-seed metrics, and API event
traces for the per-round LLM policy. `live_transfer/` contains completed LLM
records plus the 20-seed frozen-router summaries and metrics. LLM records retain
the exact prompt payload, raw response, normalized patches, and usage metadata;
they contain no API key.
