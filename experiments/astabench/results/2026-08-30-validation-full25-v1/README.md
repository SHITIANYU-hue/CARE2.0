# AstaBench DiscoveryBench full validation generation check

## What was run

This is the complete 25-sample DiscoveryBench validation split, one epoch per arm. CARE and the official ReAct baseline used the same Opus 5 endpoint, sample order, Python tool, two-sample concurrency, 36-message limit, 60,000-token sample limit, and 8,192-byte outer tool-output limit.

The official DiscoveryBench quality scorer requires `gpt-4o-2024-08-06`, which was unavailable on the configured endpoint. Both arms were therefore run with `--no-score`. The result below measures structural completion and resource use, not scientific-answer correctness. The test split was not loaded or executed.

## Result

| Measure | CARE | Official ReAct |
| --- | ---: | ---: |
| Validation samples | 25 | 25 |
| Strict `{hypothesis, workflow}` JSON | 23/25 | 12/25 |
| Samples that called `submit` | 25/25 | 12/25 |
| Sample errors | 0 | 2 |
| Python calls | 69 | 165 |
| Model calls | 94 | 176 |
| Total tokens | 718,084 | 1,213,851 |
| Wall-clock time | 44:57 | 86:28 |

CARE used 40.8% fewer tokens, 58.2% fewer Python calls, and 46.6% fewer model calls. In the paired strict-format comparison, both arms succeeded on 12 samples, CARE alone succeeded on 11, ReAct alone succeeded on none, and neither was strict on two.

CARE still had two non-strict outputs and they remain counted as failures: one submitted an additional `budget_note` key and one submitted only `hypothesis`. ReAct recorded two sample errors after an empty Python tool result was passed back as an empty tool message and rejected by the compatible API. No sample was deleted or rerun after inspecting its outcome.

## Interpretation

The expanded validation run confirms that the five-sample pilot was not only a small-sample formatting accident: CARE more reliably reaches an auditable submission while using fewer analysis and model calls. It does not establish that CARE's hypotheses are scientifically better. That comparison must wait for an available official scorer or a preregistered human/expert evaluation.

## Files

- `summary.json`: aggregate paired counts, usage, time, and the claim boundary.
- `per_sample.csv`: aligned per-sample structural and resource metrics.
- `raw_logs/care.eval`: native Inspect log for CARE, including all traces.
- `raw_logs/react.eval`: native Inspect log for the official ReAct baseline.
- `SHA256SUMS`: integrity hashes for the package.

The package can be regenerated with `scripts/summarize_unscored_paired_logs.py`.
