# Zero-shot Suzuki -> ChemLex

This is a 10-seed smoke expansion using the same no-target-calibration
protocol as the Buchwald-Hartwig audit. The frozen source record is
`outputs/llm_semantic/suzuki_to_chemlex_semantic_record.json`.

- Target calibration seeds: `0`
- Pre-decision target outcomes: `0`
- Online budget: `5` initial observations + `8` rounds per seed
- Baselines: GP-UCB and mixed-kernel GP-EI
- Matched random nulls: 2 replicates per LLM skill

`counter_hypothesis_branching` and `reagent_effectiveness` have positive
mean composite deltas against mixed-kernel GP-EI (`+22.35` and `+19.18`), but
both normal 95% intervals cross zero. This is a candidate signal, not a stable
transfer claim. The complete metrics and per-seed traces are in
`summary.json`, `metrics.csv`, and `audits.jsonl`.
