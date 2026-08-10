# Baumgartner C-N multi-source warm-start protocol

## Question

Can outcomes from completed C-N optimization campaigns improve the first three
experiments selected for a new campaign, after which a target-only GP-UCB policy
receives the same ten target reveals as every baseline?

This benchmark tests **initial-design transfer**. It does not claim that source
outcomes remain inside the acquisition function after target observations begin.

## Data and split

The benchmark uses 13 real Baumgartner C-N campaigns from the public
`sustainable-processes/multitask` supplementary workbook pinned in
`data/raw/baumgartner_cn/`.

- Development: the first nine campaigns in workbook order.
- Evaluation: the final four campaigns in workbook order.
- Development and evaluation campaign IDs are disjoint.
- Evaluation outcomes are not loaded by `calibrate`.

The split is exploratory rather than a prospective preregistration because the
workbook was inspected while the benchmark was built. Candidate-level evaluation
outcomes were not used to select the frozen route.

## Compared policies

All policies reveal three initial candidates and then run the same target-only
GP-UCB for ten additional reveals.

1. `target_gp_random_initial`: random initial design, averaged over frozen seeds.
2. `target_gp_space_filling`: deterministic mixed-variable farthest-point design.
3. `multisource_diverse_warmstart`: source outcome ranks plus mixed-space diversity.
4. `development_selected_route`: the route frozen on development tasks.

For the transfer policy, a GP is fitted independently to each eligible source
campaign. Predictions are converted to ranks and aggregated by their median.
The first initial candidate maximizes this source consensus. Subsequent initial
candidates maximize

```text
(1 - diversity_weight) * source_rank_prior
  + diversity_weight * distance_to_selected_design
```

Source campaigns are selected using public campaign descriptors only. The frozen
candidate scopes are: all sources, same substrate first, or same precatalyst first.

## Development selection

Eight source-scope/diversity candidates are compared on nine development tasks.
For each task and metric, the reference is whichever is stronger: mean random
initialization or deterministic space filling. The statistical unit is the task,
not the random seed.

The transfer route is accepted only when the development-task 95% confidence
interval for mean best-so-far AUC improvement is above zero and the task non-loss
rate is at least 0.8. Otherwise the stronger non-transfer route is deployed.

The frozen development selection is:

- source scope: `same_substrate_then_precatalyst`;
- diversity weight: `0.85`;
- AUC delta versus the stronger per-task baseline: `+1.3207`;
- task-level 95% CI: `[+0.3152, +2.3262]`;
- task win/non-loss rates: `7/9` and `9/9`.

These are development results, not confirmation results.

## Confirmation discipline

`confirm` checks the SHA-256 fingerprint of the frozen JSON configuration. It
evaluates the selected route, the best source route, random initialization over
100 frozen seeds, and space filling on four separate campaigns. Every candidate
reveal and revealed target value is recorded in per-task JSONL audit logs.

Primary metric: mean best-so-far yield across the ten GP-UCB reveals. Secondary
metrics: final best yield, simple regret, and top-10 hit.

## Interpretation boundary

A positive result supports reusable source knowledge for initial experiment
design within a family of real C-N campaigns. It does not by itself establish
arbitrary chemistry-to-materials transfer, LLM superiority, or continuous source
outcome use throughout Bayesian optimization.
