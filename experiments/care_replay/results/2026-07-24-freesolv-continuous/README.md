# Continuous FreeSolv cross-task extension

This is an extension of the frozen source-outcome protocol to the continuous
FreeSolv adapter. It is kept separate from the headline seven-pair result
because it uses 30 calibration seeds and 50 held-out seeds rather than the
headline 50/100 protocol.

Both paths use real molecular-property data, the same source-history boundary,
the same target-only baselines, and the same calibration gate:

| Source -> target | Route proposal | Deployed route |
| --- | --- | --- |
| Lipophilicity -> FreeSolv continuous | shared descriptor source-outcome | matched target-only LLM |
| ESOL -> FreeSolv continuous | schema-aligned source-outcome | matched target-only LLM |

The raw `llm_transfer_router` was stronger than the target-only BO baselines on
both held-out paths. For Lipophilicity -> FreeSolv continuous, its delta versus
the strongest target-only BO was Final Best `+3.7734` (95% CI `[+0.7118,
+6.8351]`) and AUC `+3.5521` (95% CI `[+0.7612, +6.3430]`). For ESOL ->
FreeSolv continuous, the corresponding deltas were Final Best `+3.4653`
(`[+0.3760, +6.5546]`) and AUC `+3.4558` (`[+0.7759, +6.1356]`).

However, neither raw route beat the matched target-only LLM with a confidence
interval above zero:

| Source -> target | Final delta vs matched LLM | AUC delta vs matched LLM |
| --- | ---: | ---: |
| Lipophilicity -> FreeSolv continuous | `+0.3985` `[-0.4028, +1.1998]` | `+0.1845` `[-0.6518, +1.0209]` |
| ESOL -> FreeSolv continuous | `+0.0904` `[-0.1790, +0.3597]` | `+0.0882` `[-0.1455, +0.3219]` |

The frozen gate therefore correctly rejected both source routes. This is a
useful result: source transfer can beat classical BO while still failing to
show an incremental gain over a strong target-only LLM. The experiment should
not be reported as a positive CARE transfer result, but it does expand the
matrix and preserves the full audit trail.

Artifacts:

- `raw_summaries/`: calibration and held-out decisions;
- `raw_metrics/`: per-seed metrics for all routes and baselines;
- `audit_archives/`: 1,060 JSONL audit files;
- `logs/`: one log per pair;
- `SHA256SUMS`: archive integrity manifest;
- `../goal-report-freesolv-continuous-2026-07-24/`: machine-readable and Markdown report.
