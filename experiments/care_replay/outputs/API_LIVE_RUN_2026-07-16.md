# CARE 2.0 Live LLM Run

Date: 2026-07-16

## Purpose

This run verifies that the newly supplied Common Stack key can make real LLM calls and that the CARE 2.0 replay harness records the resulting reasoning traces. It is an offline replay benchmark, not a new wet-lab experiment.

## Configuration

- Model: `openai/gpt-5.6-sol`
- LLM mode: API
- Seeds: 3
- Initial observations: 8
- Replay rounds: 8
- Transfer pairs: `bh -> suzuki`, `suzuki -> chemlex`, `real_chemlex_train -> real_chemlex_test`
- Real CHEMLEX split: `Stratified_Split_Both_Unseen`
- CHEMLEX source candidates: 4,960
- CHEMLEX target candidates: 2,234
- LLM calls: 18 total, 18 successful, 0 fallback

The API smoke test also returned the exact sentinel response `CARE_LLM_LIVE` from the selected model.

## What the LLM returned

For each source-transfer and target-only proposal, the model returned structured JSON containing:

- confidence;
- per-feature weights for electronics, activation, temperature, solvation, stability, and compatibility;
- pairwise interaction terms.

These proposals were then passed through the existing CARE 2.0 replay policies. The trace file records the prompt context, raw response, parsed proposal, selected patches, and downstream audit information without storing the API key.

## Results

The comparison below is against the strong target-only `kernel_ucb` acquisition baseline. Positive values favor `care_transfer`.

| Source -> target | Final-best delta | AUC delta | Win rate | 95% interval for final-best delta |
| --- | ---: | ---: | ---: | ---: |
| BH -> Suzuki | -1.856 | -2.294 | 1/3 | [-8.267, 4.556] |
| Suzuki -> ChemLex | +0.868 | +0.015 | 1/3 | [-4.785, 6.522] |
| CHEMLEX train -> test | +3.303 | +1.468 | 1/3 | [-3.171, 9.778] |
| All three pairs | +0.772 | -0.270 | 1/3 | pairwise |

## Interpretation

The main result of this run is that the live LLM path is now working and auditable: every proposal came from `gpt-5.6-sol`, rather than the proxy or fallback path. The performance signal is mixed. The real CHEMLEX split shows the clearest positive direction, Suzuki -> ChemLex is nearly flat to mildly positive, and BH -> Suzuki is negative in this small run. With three seeds, none of these differences should be treated as a stable claim yet.

The next statistically meaningful check is to hold the prompt and policy fixed and expand these same pairs to 30 or 50 seeds. The trace should also be retained so that any gain can be attributed to the LLM proposal, the acquisition geometry, or the CARE gate rather than to a hidden fallback.

## Artifacts

- `outputs/runs/transfer_generalization_live_gpt56sol_20260716_llm_trace.jsonl`
- `outputs/runs/transfer_generalization_live_gpt56sol_20260716_metadata.json`
- `outputs/runs/transfer_generalization_live_gpt56sol_20260716_proposals.json`
- `outputs/runs/transfer_generalization_live_gpt56sol_20260716_audit_seed0.jsonl`
- `outputs/tables/transfer_generalization_live_gpt56sol_20260716_aggregate.csv`
- `outputs/tables/transfer_generalization_live_gpt56sol_20260716_comparison.csv`
- `outputs/tables/transfer_generalization_live_gpt56sol_20260716_runs.csv`
