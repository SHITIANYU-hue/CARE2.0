# online_llm_bounded_authority_route_disjoint_confirmation_v1

- Protocol complete: `false`
- Claim decision: `not_evaluated_incomplete_protocol`
- Completed trajectories: `6` / `180`
- Failed trajectories: `1`

Partial execution is operational evidence only. No confirmatory claim is evaluated until all declared trajectories are complete.

## Primary estimand

Equal-route-weighted mean of within-route mean best-so-far AUC deltas for the bounded-authority controller versus the matched same-initial target-only GP-UCB comparator.

## Route statistics

| Route | Tier | n | Mean AUC delta | 95% CI | W/T/L | BH-adjusted sign p |
|---|---|---:|---:|---:|---:|---:|
| molecular_lipophilicity_to_freesolv | route_disjoint_repeated_confirmation | 1 | 0.0 | not estimable | 0/1/0 | NA |
| materials_bulk_to_shear_modulus | route_disjoint_repeated_confirmation | 1 | 0.0 | not estimable | 0/1/0 | NA |
| materials_phonons_to_perovskites | route_disjoint_repeated_confirmation | 1 | 0.0 | not estimable | 0/1/0 | NA |
| materials_expt_gap_to_steels | route_disjoint_repeated_confirmation | 1 | 0.0 | not estimable | 0/1/0 | NA |
| molecular_freesolv_to_bace | route_disjoint_repeated_confirmation | 1 | 0.0 | not estimable | 0/1/0 | NA |
| materials_perovskites_to_mp_e_form | route_disjoint_repeated_confirmation | 1 | 0.0 | not estimable | 0/1/0 | NA |

## Interpretation

The repeated protocol estimates stochastic variation of the frozen online controller. Development and post-freeze routes remain labeled separately, and completion of this protocol does not convert retrospective routes into independent external validation.
