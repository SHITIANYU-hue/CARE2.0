# opus48_high_authority_chemistry_holdout_v1

- Evidence class: `post_freeze_real_chemistry_route_extension`
- Model: `anthropic/claude-opus-4-8`
- Primary comparator: the same LLM-selected initial observations followed by target-only GP-UCB.
- Target outcomes are revealed only after each candidate is selected.

## Aggregate

Across 5 completed routes, the online LLM controller changed best-so-far AUC by +0.0212 on average (3 wins, 1 ties, 1 losses).
Mean LLM participation was 100%; mean decision authority was 100%; the GP default was overridden in 50% of rounds.

## Routes

| Route | Domain | AUC delta vs GP | Final delta vs GP | GP override | Source active |
|---|---|---:|---:|---:|---:|
| baumgartner_aniline_to_phenethylamine_alphos | reaction_optimization_cn | +0.6286 | +0.0000 | 60% | 0% |
| baumgartner_aniline_to_benzamide_tbuxphos | reaction_optimization_cn | +0.1781 | +0.0000 | 30% | 70% |
| baumgartner_aniline_to_phenethylamine_tbubrettphos | reaction_optimization_cn | -0.9806 | +0.0000 | 40% | 30% |
| baumgartner_benzamide_tbubrettphos_to_alphos | reaction_optimization_cn | +0.0000 | +0.0000 | 80% | 30% |
| reizman_cases_123_to_case4 | reaction_optimization_suzuki | +0.2800 | +0.4000 | 40% | 60% |

## Interpretation boundary

A positive delta isolates the online LLM controller because both methods start from the same initial observations and use the same target budget. It does not by itself prove universal cross-domain transfer.
`Source active` reports whether the LLM continued to rely on source-task evidence. A route can improve after source transfer is stopped; that is a useful routing result, but it is not positive source-outcome transfer.

Each case directory contains the frozen initial record, full proposal/critic trace, summary, and SHA-256 fingerprints.
