# CARE 2.0 Live LLM Trace Index

Date: 2026-07-16  
Branch base: `target-calibrated-transfer`  
Model: `openai/gpt-5.6-sol` via the Common Stack API

This directory records the live-API replay used to audit CARE 2.0's transfer path. The LLM was asked to infer priors over semantic roles and interactions from source and target task summaries. It did not receive candidate IDs or oracle outcomes. The downstream replay then compared target-only, direct LLM transfer, CARE transfer, and kernel-UCB acquisition.

## Runs

- `transfer_generalization_live_gpt56sol_20260716_*`: earlier 3-seed live replay.
- `transfer_generalization_live_gpt56sol_20260716_10seed_*`: 10-seed extension with 60 LLM calls, all successful and with no fallback.

Each run includes:

- `llm_trace.jsonl`: prompt, raw response, parsed response, proposal, and trace status for every LLM call.
- `metadata.json`: model, task pairs, replay settings, method definitions, and trace counts.
- `proposals.json`: serialized proposals consumed by the replay.
- `audit_seed0.jsonl`: downstream selection and audit events for the first replay seed.
- matching CSV files under `outputs/tables/`: aggregate metrics, baseline comparisons, and per-seed results.

## 10-seed headline

Against the strong target-only `kernel_ucb` baseline, CARE transfer had an aggregate final-best delta of `+0.953` and AUC delta of `+1.474` across 30 paired runs. The pairwise final-best deltas were `+0.640` for BH -> Suzuki, `-0.587` for Suzuki -> ChemLex, and `+2.805` for the real CHEMLEX train -> test split. The confidence intervals include zero, so this is evidence of a live and auditable transfer path, not a claim of uniform improvement.

The logs contain no API key. They are offline replay artifacts and do not represent a new wet-lab experiment.
