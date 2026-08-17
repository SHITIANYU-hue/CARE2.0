# opus48_high_authority_development_v1

- Evidence class: `retrospective_controller_development_on_previously_analyzed_targets`
- Model: `anthropic/claude-opus-4-8`
- Primary comparator: the same LLM-selected initial observations followed by target-only GP-UCB.
- Target outcomes are revealed only after each candidate is selected.

## Aggregate

Across 6 completed routes, the online LLM controller changed best-so-far AUC by +1.8403 on average (3 wins, 1 ties, 2 losses).
Mean LLM participation was 100%; mean decision authority was 100%; the GP default was overridden in 42% of rounds.

## Routes

| Route | Domain | AUC delta vs GP | Final delta vs GP | GP override | Source active |
|---|---|---:|---:|---:|---:|
| molecular_freesolv_to_esol | molecular_property | +0.0000 | +0.0000 | 30% | 80% |
| molecular_freesolv_to_lipophilicity | molecular_property | +6.3250 | +4.8750 | 40% | 60% |
| materials_expt_gap_to_dielectric | materials_property | +6.7451 | +0.0000 | 40% | 20% |
| materials_phonons_to_bulk_modulus | materials_property | +1.1302 | +0.0000 | 50% | 0% |
| materials_dielectric_to_jdft2d | materials_property | -0.7119 | -5.2431 | 60% | 100% |
| materials_expt_gap_to_mp_gap | materials_property | -2.4464 | -24.4638 | 30% | 100% |

## Interpretation boundary

A positive delta isolates the online LLM controller because both methods start from the same initial observations and use the same target budget. It does not by itself prove universal cross-domain transfer.
`Source active` reports whether the LLM continued to rely on source-task evidence. A route can improve after source transfer is stopped; that is a useful routing result, but it is not positive source-outcome transfer.

Each case directory contains the frozen initial record, full proposal/critic trace, summary, and SHA-256 fingerprints.
