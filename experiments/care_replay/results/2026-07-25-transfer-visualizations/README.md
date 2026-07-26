# CARE 2.0 Transfer Visualizations

These figures are generated from the frozen source-outcome transfer archive at
`experiments/care_replay/results/2026-07-24-source-outcome-transfer/`.

## Figures

- `transfer_role_weight_heatmap.png/pdf`: mean role multiplier across the
  candidate LLM patches for each source-target pair. `1.00` is neutral; values
  above or below one show relative up- or down-weighting. The row prefix marks
  deployed positive transfer (`+`), rejected negative transfer (`-`), and
  rejected uncertain transfer (`~`).
- `transfer_matrix.png/pdf`: source rows and target columns. Green cells are
  deployed positive transfer; red cells are raw negative routes rejected by the
  gate; gray cells are unstable routes that fell back to target-only.
- `transfer_graph.png/pdf`: directed source-to-target migration graph across
  reaction, materials, and molecular tasks. Edge labels show the composite
  held-out signal.
- `transfer_evidence_forest.png/pdf`: route-level composite delta with its 95%
  confidence interval, making the zero-gain boundary and uncertain routes
  explicit.
- `transfer_metric_profile.png/pdf`: side-by-side comparison of composite
  signal, deployed final-best delta, and top-10 rounds saved.
- `transfer_domain_coverage.png/pdf`: source-domain by target-domain coverage;
  blank cells are domain combinations not evaluated in this archive.
- `transfer_visualization_data.json`: the exact records used to render the
  figures.
- `reasoning_trace_index.json`: index of model-call records, representative
  reasoning traces, and zero-shot replay audits.

The trace files are structured audit records: prompt metadata, model outputs,
compiled skills or patches, selected candidates, revealed outcomes, and
replay diagnostics. They are not claimed to be hidden model chain-of-thought.

## Rebuild

```bash
python3 experiments/care_replay/scripts/build_transfer_visualizations.py \
  --source-outcome-root experiments/care_replay/results/2026-07-24-source-outcome-transfer \
  --output-dir experiments/care_replay/results/2026-07-25-transfer-visualizations

python3 experiments/care_replay/scripts/build_reasoning_trace_index.py \
  --repo-root . \
  --source-outcome-root experiments/care_replay/results/2026-07-24-source-outcome-transfer \
  --model-call-root experiments/care_replay/results/2026-07-25-hypothesis-generation \
  --zero-shot-root experiments/care_replay/results/2026-07-25-hypothesis-zero-shot-suzuki-to-bh-30seed \
  --zero-shot-root experiments/care_replay/results/2026-07-25-hypothesis-zero-shot-suzuki-to-chemlex-10seed \
  --zero-shot-root experiments/care_replay/results/2026-07-25-hypothesis-zero-shot-expt-gap-to-dielectric-10seed \
  --zero-shot-root experiments/care_replay/results/2026-07-25-hypothesis-zero-shot-esol-to-freesolv-10seed \
  --zero-shot-root experiments/care_replay/results/2026-07-25-hypothesis-zero-shot-freesolv-to-lipophilicity-10seed \
  --output experiments/care_replay/results/2026-07-25-transfer-visualizations/reasoning_trace_index.json
```
