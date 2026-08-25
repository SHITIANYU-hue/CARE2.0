# opus48_flip2_hydro_external_family_v1

- Evidence class: `new_biology_task_family_wild_type_holdout`
- Model: `anthropic/claude-opus-4-8`
- Primary comparator: the same LLM-selected initial observations followed by target-only GP-UCB.
- Target outcomes are revealed only after each candidate is selected.

## Aggregate

Across 1 completed routes, the online LLM controller changed best-so-far AUC by +0.0000 on average (0 wins, 1 ties, 0 losses).
Mean LLM participation was 8%; mean decision authority was 8%; the GP default was overridden in 0% of rounds.

## Routes

| Route | Domain | AUC delta vs GP | Final delta vs GP | GP override | Source active |
|---|---|---:|---:|---:|---:|
| flip2_hydro_p01053_p0a9x9_to_p06241 | protein_engineering | +0.0000 | +0.0000 | 0% | 0% |

## Interpretation boundary

A positive delta isolates the online LLM controller because both methods start from the same initial observations and use the same target budget. It does not by itself prove universal cross-domain transfer.
`Source active` reports whether the LLM continued to rely on source-task evidence. A route can improve after source transfer is stopped; that is a useful routing result, but it is not positive source-outcome transfer.

Each case directory contains the frozen initial record, full proposal/critic trace, summary, and SHA-256 fingerprints.
