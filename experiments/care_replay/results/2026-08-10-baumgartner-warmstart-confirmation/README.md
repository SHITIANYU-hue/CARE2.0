# Baumgartner multi-source warm-start confirmation

This archive evaluates source-guided initial experiment design on 13 real
Baumgartner C-N optimization campaigns. The transfer route uses completed
source outcomes to select three diverse initial candidates, then switches to
the same target-only GP-UCB used by every baseline.

## Frozen development selection

Nine development campaigns were used to compare eight source-scope/diversity
policies against both random initialization and deterministic space filling.
The statistical unit is the campaign. For each campaign, the reference is the
stronger of the two non-transfer baselines.

The selected route was frozen as:

- sources: same substrate first, otherwise same precatalyst;
- initial design: source rank consensus plus `0.85` diversity weight;
- primary metric: best-so-far AUC over ten target-only GP-UCB reveals;
- mean development AUC delta: `+1.320699`;
- task-level 95% CI: `[+0.315177, +2.326222]`;
- task win rate: `7/9`;
- task non-loss rate: `9/9`.

The server selection record is byte-identical to the local preflight record
(`SHA-256 ad22d60effbf6e89424203a8abe6d430b6920708f60f6b04f6a2bbdf4ab249d0`).

## Confirmation status

The four evaluation campaigns remain outside development selection. Confirmation
is run only after the code, configuration, selection record, and reusable skill
have been committed. Results are added under `confirmation/` without changing
the frozen route.

## Files

- `configs/`: frozen benchmark configuration.
- `development/selection_record.json`: complete policy comparison and route.
- `development/development_metrics.csv`: raw per-task/per-mode metrics.
- `development/development_audits.tar.gz`: every candidate reveal and outcome.
- `skill_bank/`: exact reusable skill compiled from development evidence.

## Claim boundary

This is a campaign-disjoint test of initial-design transfer within a family of
C-N reaction campaigns. It is not evidence for arbitrary cross-domain transfer,
LLM superiority, or continuous source use after target observations begin.
