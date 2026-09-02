# AstaBench DiscoveryBench official HMS rescoring

## What was scored

The previously frozen 25-sample validation generation logs for CARE and the
official ReAct baseline were rescored without rerunning either agent. The
scoring run used AstaBench 0.5.3's official DiscoveryBench HMS implementation
and its required OpenAI judge snapshot, `gpt-4o-2024-08-06`, through the direct
OpenAI endpoint.

Both arms therefore retain the same Opus 5 generations, task order, tools, and
resource limits reported in
`../2026-08-30-validation-full25-v1/`. Only the previously missing quality
scoring stage was added. The test split was not loaded or scored.

## Official quality result

| Measure | CARE | Official ReAct |
| --- | ---: | ---: |
| Mean HMS | 0.1827 | 0.1683 |
| Pairwise wins | 7 | 7 |
| Pairwise ties | 11 | 11 |

The paired mean difference, CARE minus ReAct, is `+0.0143`. A 200,000-resample
task bootstrap gives a 95% interval of `[-0.1260, +0.1563]`. The interval
crosses zero, so this validation run does not establish scientific-quality
superiority.

HMS is AstaBench's multiplicative context-variable-relation score in the
`[0, 1]` range. It should not be read as ordinary percent accuracy.

## Combined interpretation

The official quality result changes the earlier claim boundary. We no longer
need to say that scientific quality was unscored: CARE's mean official HMS is
slightly higher, but statistically inconclusive on these 25 validation tasks.

The separate execution result remains clear: CARE reached strict JSON plus
submission on 23/25 tasks versus 12/25 for ReAct, used 40.8% fewer tokens,
58.2% fewer Python calls, and 46.6% fewer model calls. The defensible conclusion
is therefore that CARE improves workflow reliability and resource efficiency
while maintaining comparable measured scientific quality; quality superiority
has not yet been demonstrated.

## Reproduction

Set a direct OpenAI API key with access to the frozen judge snapshot. Do not
store the key in the repository.

```bash
cd experiments/astabench
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"

uv run inspect score \
  results/2026-08-30-validation-full25-v1/raw_logs/care.eval \
  --scorer configurable_discoverybench_scorer.py@score_discoverybench_configurable \
  --stream 4 --output-file care-scored-official.eval
```

The ReAct log is scored with the same command and judge. Aggregate the paired
logs with `scripts/summarize_scored_paired_logs.py`.

## Files

- `summary.json`: paired means, bootstrap interval, win/tie/loss counts, and
  claim boundary.
- `per_sample.csv`: all 25 paired HMS values and differences.
- `raw_logs/care-scored-official.eval`: native scored CARE Inspect log.
- `raw_logs/react-scored-official.eval`: native scored ReAct Inspect log.
- `SHA256SUMS`: integrity hashes for the evidence package.
