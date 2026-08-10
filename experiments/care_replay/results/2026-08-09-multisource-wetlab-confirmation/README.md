# Multi-source wet-lab confirmation

This archive is the first CARE 2.0 confirmation that uses several completed
real reaction campaigns to make a frozen routing decision for a new campaign.
The run was executed on the project GPU server from commit `9a18089` with
Python 3.8.10.

## Protocol

- Real Reizman Suzuki data: cases 1-3 are development sources; case 4 is the
  untouched confirmation target.
- Mixed search space: catalyst identity plus residence time, temperature, and
  catalyst loading.
- Development selection: leave one of cases 1-3 out, 30 seeds per fold.
- Confirmation: 100 new seeds on case 4.
- Budget: three matched initial observations plus twelve sequential reveals.
- Primary metric: best-so-far AUC under the matched reveal budget.
- Admission gate: a source route must have development AUC 95% CI lower bound
  greater than zero; otherwise the exact target-only GP-UCB route is used.

No case-4 outcome was loaded to choose a method or weight. The configuration
fingerprint is checked before confirmation starts.

## Development decision

None of the six source-informed candidates passed the frozen AUC gate across 90
development folds. The least harmful source prior had AUC delta `-0.518` with
95% CI `[-1.320, +0.285]`. The selected deployment route was therefore
`target_gp_ucb`.

## Case-4 result

| Policy | Final best | AUC | Top-10 hit | AUC delta vs target GP-UCB |
| --- | ---: | ---: | ---: | ---: |
| target GP-UCB | 97.224 | 93.956 | 0.87 | 0.000 |
| multi-source RGPE | 94.517 | 93.051 | 0.97 | -0.905 `[-1.638, -0.172]` |
| multi-source ICM-BMA | 95.113 | 93.126 | 0.91 | -0.830 `[-1.546, -0.114]` |
| selected source-rank prior | 97.045 | 94.293 | 0.85 | +0.337 `[-0.225, +0.899]` |
| frozen deployment route | 97.224 | 93.956 | 0.87 | 0.000 |

The source-rank prior has a positive mean AUC delta on case 4, but its interval
crosses zero and it failed the task-disjoint development gate. It is retained
as an inconclusive exploratory result, not selected after seeing case 4.

RGPE significantly increases the probability of touching a top-10 condition by
`+0.10` with CI `[+0.029, +0.171]`, but it reduces both AUC and final best yield.
This is a real objective trade-off, not an overall positive-transfer result.

The main supported result is calibrated abstention: the frozen CARE route
introduces zero negative transfer on the unseen task. This archive does not
establish a positive multi-source transfer gain or an LLM contribution. The
`SKILL.md` artifact was compiled deterministically from development evidence;
automatic LLM trace-to-skill evolution remains future work.

## Artifacts

- `development/development_metrics.csv`: 630 development rows.
- `development/selection_record.json`: frozen route, comparisons, and config
  fingerprint.
- `confirmation/reizman_cases_123_to_case4_metrics.csv`: 500 confirmation rows.
- `confirmation/reizman_cases_123_to_case4_summary.json`: aggregate metrics and
  paired confidence intervals.
- `confirmation/audits.tar.gz`: 500 per-policy/per-seed JSONL audits, twelve
  sequential events each.
- `skill_bank/`: the exact agent-readable skill and evidence used at the
  development/confirmation boundary.
- `SERVER_RUN.txt`: server environment and executed commit.
- `SHA256SUMS`: integrity hashes for archived evidence.

The full experiment definition and claim boundary are documented in
`../../MULTISOURCE_WETLAB_PROTOCOL.md`.
