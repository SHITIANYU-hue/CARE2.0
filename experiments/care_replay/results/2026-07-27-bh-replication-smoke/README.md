# Frozen BH semantic-skill replication smoke

Date: 2026-07-27

This is a small independent replication of the Suzuki -> Buchwald-Hartwig
semantic-skill result. It uses a frozen `gpt-4o-mini` record generated before
the replay, a preselected `high_mw_ligand_effect` skill, and new non-overlapping
seeds. It is a smoke replication, not a replacement for the 500-seed
confirmatory result from 2026-07-21.

## Protocol

- Source: real Suzuki-Miyaura reaction HTE schema.
- Target: real Buchwald-Hartwig reaction HTE data.
- Calibration: 10 seeds (`82000-82009`), used only to choose the target-only
  anchor and strategy route.
- Held out: 30 seeds (`83000-83029`), never used for selection.
- Budget: 5 initial observations plus 10 reveal rounds.
- Frozen skill: `high_mw_ligand_effect`.
- Compared methods: GP-UCB, mixed-kernel GP-EI, target acquisition portfolio,
  semantic skill, LLM direct prior, and LLAMBO-style warm start.
- LLM calls during replay: zero. The one recorded generation call is retained
  in `model_calls/`.

## Held-out result

The selected semantic skill was better than the strongest target-only method,
mixed-kernel GP-EI, on all three reported metrics in this 30-seed smoke:

| Metric | Delta vs. mixed-kernel GP-EI | 95% CI |
| --- | ---: | ---: |
| Final best | +2.5801 | [+0.4088, +4.7515] |
| Best-so-far AUC | +3.2158 | [+0.1176, +6.3140] |
| Top-10 hit | +0.1667 | [+0.0044, +0.3289] |

The strategy router selected the LLM direct-prior route on calibration. On
held-out seeds, CARE's router and the direct-prior route were identical. Thus
this run supports the semantic skill's cross-task gain over classical BO, but
does not establish an additional router gain over the direct-prior LLM
baseline.

## Interpretation boundary

This result is encouraging but small. It should be combined with the existing
500-seed frozen confirmations before making a headline claim. The broader
archive still contains negative and fallback routes, including ChemLex and
FreeSolv controls; those are part of the evidence for the calibration gate.

## Reproduction

The replay can be regenerated with:

```bash
python3 experiments/care_replay/scripts/run_calibrated_llm_semantic_selector.py \
  --llm-record experiments/care_replay/results/2026-07-21-cross-domain-semantic-skills/model_calls/suzuki_to_bh_semantic_record.json \
  --target-dataset real_buchwald_hartwig \
  --calibration-seed-start 82000 --calibration-seeds 10 \
  --heldout-seed-start 83000 --heldout-seeds 30 \
  --skill-variants record --preselected-skill-id high_mw_ligand_effect \
  --include-llm-baselines --output-tag 20260727_replication_bh_smoke
```

`raw_metrics/` contains per-seed results, `raw_summaries/` contains the
selection and confidence intervals, `model_calls/` contains the frozen LLM
record, and `audit_archives/` contains the per-round audit traces.
