# Live RSI replication extension: blocked launch

This directory retains the failed operational launch of the frozen v2 replication extension.

- Planned cases: 9 (3 tasks × 3 new model replicates).
- Generation attempts made: 18.
- Valid generations: 0.
- Scientific trajectories produced: 0.
- All cases failed before initial skill generation with HTTP 429.
- A separate minimal provider probe returned `rate_limit_exceeded` with `Access key max cost limit exceeded (cap 100)`; see `execution/provider_probe.json`.

No efficacy result can be computed from this launch. It must not be pooled with the completed 2026-09-09 experiment. The v3 config preserves the scientific design and adds transport backoff plus immediate termination for non-recoverable provider errors.
