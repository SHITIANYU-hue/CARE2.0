# CARE 2.0 Live LLM Run: 10 Seeds

Date: 2026-07-16

## Purpose

This run expands the live API replay from three to ten seeds so that the LLM trace and transfer signal can be inspected across more random initializations. It remains an offline replay benchmark, not a new wet-lab experiment.

## Configuration

- Model: `openai/gpt-5.6-sol`
- LLM mode: API
- Seeds: 10
- Initial observations: 8
- Replay rounds: 8
- Transfer pairs: `bh -> suzuki`, `suzuki -> chemlex`, `real_chemlex_train -> real_chemlex_test`
- Real CHEMLEX split: `Stratified_Split_Both_Unseen`
- CHEMLEX source candidates: 4,960
- CHEMLEX target candidates: 2,234
- Paired benchmark runs: 30
- LLM calls and traces: 60 total, 60 successful, 0 fallback

Each pair/seed produces one source-transfer proposal and one target-only proposal. The trace records confidence, feature weights, interaction terms, raw response, parsed response, and downstream audit metadata without storing the API key.

## Results

The comparison below is against the strong target-only `kernel_ucb` acquisition baseline. Positive values favor `care_transfer`.

| Source -> target | Final-best delta | AUC delta | Win rate | 95% interval for final-best delta |
| --- | ---: | ---: | ---: | ---: |
| BH -> Suzuki | +0.640 | +2.406 | 2/10 | [-1.881, 3.161] |
| Suzuki -> ChemLex | -0.587 | -0.342 | 4/10 | [-2.811, 1.637] |
| CHEMLEX train -> test | +2.805 | +2.359 | 2/10 | [-6.277, 11.887] |
| All three pairs | +0.953 | +1.474 | 8/30 | pairwise |

## Interpretation

The 10-seed run gives a more useful picture than the earlier three-seed smoke test. The aggregate AUC direction is positive for BH -> Suzuki and the real CHEMLEX split, while Suzuki -> ChemLex remains slightly negative. However, every reported final-best interval includes zero and the per-pair win rates are modest. The current evidence supports a live, auditable LLM transfer path with a promising but not yet statistically established gain; it does not support claiming uniform improvement across domains.

## Artifacts

- `outputs/runs/transfer_generalization_live_gpt56sol_20260716_10seed_llm_trace.jsonl`
- `outputs/runs/transfer_generalization_live_gpt56sol_20260716_10seed_metadata.json`
- `outputs/runs/transfer_generalization_live_gpt56sol_20260716_10seed_proposals.json`
- `outputs/runs/transfer_generalization_live_gpt56sol_20260716_10seed_audit_seed0.jsonl`
- `outputs/tables/transfer_generalization_live_gpt56sol_20260716_10seed_aggregate.csv`
- `outputs/tables/transfer_generalization_live_gpt56sol_20260716_10seed_comparison.csv`
- `outputs/tables/transfer_generalization_live_gpt56sol_20260716_10seed_runs.csv`
