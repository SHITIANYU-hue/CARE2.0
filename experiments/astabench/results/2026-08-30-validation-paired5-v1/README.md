# AstaBench validation paired-5 audit

This evidence package compares the frozen CARE solver with the stock ReAct
solver on the same five AstaBench DiscoveryBench validation samples. Both arms
used `anthropic/claude-opus-5`, the same CommonStack endpoint, one epoch, a
60,000-token per-sample limit, and at most two concurrent samples. The public
test split was not opened.

## Result

CARE returned the required two-key JSON and called `submit` on 5/5 samples.
ReAct did so on 1/5. CARE used 176,412 total tokens versus 306,728 for ReAct,
a 42.5% reduction. CARE also used 15 Python calls versus 35 for ReAct.

| Arm | Completed | Strict JSON + submit | Python calls | Model calls | Total tokens | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CARE | 5/5 | 5/5 | 15 | 20 | 176,412 | 14m 33s |
| ReAct | 5/5 | 1/5 | 35 | 40 | 306,728 | 17m 28s |

## Claim boundary

This is a structural-completion and resource-efficiency result, not an official
DiscoveryBench scientific-quality score. AstaBench's scorer requested
`gpt-4o-2024-08-06`, which was unavailable through the configured endpoint, so
the run logs contain no official scorer values. One CARE sample also encountered
a dataset-path problem and submitted an unresolved conclusion instead of
inventing evidence. Its valid format is not evidence that its answer was
scientifically correct.

## Files

- `per_sample.csv`: paired per-sample structural and resource measurements.
- `summary.json`: protocol, aggregate counts, token usage, and claim boundary.
- `raw_logs/care.eval`: native Inspect evaluation log for CARE.
- `raw_logs/react.eval`: native Inspect evaluation log for ReAct.
- `SHA256SUMS`: hashes for the evidence files.

The run used repository commit `ecf6b96`. The Inspect run metadata marks the
working tree dirty because local run configuration was present; the solver and
benchmark protocol were the committed versions at that revision.
