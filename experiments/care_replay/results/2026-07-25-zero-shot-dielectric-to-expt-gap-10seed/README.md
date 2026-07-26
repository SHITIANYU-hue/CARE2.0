# Zero-shot Matbench dielectric -> experimental gap

This is a 10-seed material-domain smoke expansion with no target calibration
and no target-based skill selection. The frozen source record is
`outputs/llm_semantic/dielectric_to_bandgap_semantic_record.json`.

- Target calibration seeds: `0`
- Pre-decision target outcomes: `0`
- Online budget: `5` initial observations + `8` rounds per seed
- Baselines: GP-UCB and mixed-kernel GP-EI
- Matched random nulls: 2 replicates per LLM skill

`low_mean_atomic_number` has the largest positive mean composite delta versus
mixed-kernel GP-EI (`+15.28`), while `high_chalcogenide_effect` is clearly
negative (`-23.97`, CI entirely below zero). The positive candidates all have
intervals crossing zero, so this domain currently demonstrates both a possible
signal and negative transfer, not reliable generalization.
