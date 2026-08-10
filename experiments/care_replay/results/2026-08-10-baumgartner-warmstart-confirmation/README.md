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

## Confirmation result

The four evaluation campaigns were run only after the code, configuration,
selection record, and reusable skill were committed. V1 did **not** generalize:

- best-so-far AUC delta: `-1.204666`;
- task-level 95% CI: `[-2.927358, +0.518025]`;
- AUC task win/non-loss rates: `1/4` and `1/4`;
- final-best delta: `-1.610000`.

The largest failure was unseen-substrate Morpholine-tBuBrettPhos. Development
tasks almost always had a same-substrate source, while the Morpholine tasks fell
back to same-precatalyst sources. That fallback had not been isolated during
development selection. V1 is retained as negative-transfer evidence and the
motivation for the v2 source-quality floor.

## Files

- `configs/`: frozen benchmark configuration.
- `development/selection_record.json`: complete policy comparison and route.
- `development/development_metrics.csv`: raw per-task/per-mode metrics.
- `development/development_audits.tar.gz`: every candidate reveal and outcome.
- `confirmation/`: frozen v1 metrics, summary, stdout, and all audits.
- `skill_bank/`: exact reusable skill compiled from development evidence.

## Claim boundary

This is a campaign-disjoint test of initial-design transfer within a family of
C-N reaction campaigns. It is not evidence for arbitrary cross-domain transfer,
LLM superiority, or continuous source use after target observations begin.
