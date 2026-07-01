# Real ChemLex Acid-Amine Wetlab Replay

Date: 2026-06-30

This snapshot replaces the synthetic ChemLex-style control with the public
ChemLex Acid-Amine wetlab data from the updated Zenodo record:

```text
https://zenodo.org/records/17596563
```

The adapter uses `Chemlex_Acidamine_Wetlab_Data.xlsx` v3 with 11,669 rows. The
decision-facing fields are anonymized categorical labels for acid, amine,
reagent, and solvent. The hidden target is `conversion_value`, revealed only
after a candidate is selected.

## Commands

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py \
  --dataset real_chemlex_acidamine \
  --seeds 30 \
  --rounds 10 \
  --initial 5 \
  --modes no_care_random,incumbent,no_gate,gate_v1,gate_v2
```

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py \
  --dataset real_chemlex_acidamine \
  --seeds 5 \
  --rounds 6 \
  --initial 5 \
  --modes no_care_random,incumbent,llm_no_gate,llm_gate_v1 \
  --llm-model openai/gpt-4o-mini \
  --output-tag llm_commonstack_5seed
```

The LLM key was supplied through the process environment and is not stored in
the repository.

## Files

- `tables/real_chemlex_acidamine_metrics.csv`: 30-seed non-LLM baseline and CARE replay.
- `tables/real_chemlex_acidamine_llm_commonstack_5seed_metrics.csv`: 5-seed real LLM replay.
- `chemlex_non_llm_summary.csv`: compact non-LLM aggregate.
- `chemlex_llm_summary.csv`: compact LLM aggregate.
- `runs/*_summary.json`: aggregate JSON summaries.

Per-seed audit logs and knowledge snapshots are kept under
`experiments/care_replay/outputs/runs/real_chemlex_acidamine*`.

## Main Result

The synthetic ChemLex control was too optimistic. On the real ChemLex wetlab
table, the current public observation model is weak: random search beats the
incumbent, gated CARE variants, and the first LLM proposer.

| Run | Mode | Final Best Mean | AUC Mean | Note |
| --- | --- | ---: | ---: | --- |
| 30-seed non-LLM | no_care_random | 90.3810 | 82.8549 | strongest current baseline |
| 30-seed non-LLM | incumbent | 82.0203 | 75.7267 | weak public observation model |
| 30-seed non-LLM | gate_v1/gate_v2 | 82.1300 | 76.0638 | tiny change, not enough |
| 5-seed LLM | no_care_random | 89.6800 | 72.7893 | strongest in LLM run |
| 5-seed LLM | incumbent | 72.0440 | 70.1367 | weak |
| 5-seed LLM | llm_no_gate/llm_gate_v1 | 72.0440 | 70.1367 | 30 real LLM calls, no improvement |

## Interpretation

This is a useful negative result. It shows why the real ChemLex data should be
used instead of the synthetic ChemLex control when discussing CARE 2.0
performance. The current factor-evidence model uses sparse categorical evidence
over acid, amine, reagent, and solvent labels. That is not strong enough for the
real wetlab table.

The next ChemLex experiment should add chemically meaningful descriptors or a
stronger candidate-ranking interface. Until then, real ChemLex should be
reported as a diagnostic dataset, not as a positive CARE result.
