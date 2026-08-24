# online_llm_outcome_semantics_robustness_v1

- Protocol complete: `false`
- Claim decision: `not_evaluated_incomplete_protocol`
- Completed trajectories: `1` / `330`
- Failed trajectories: `0`

Partial execution is operational evidence only. No confirmatory claim is evaluated until all declared trajectories are complete.

## Primary estimand

Equal-route-weighted mean of within-route mean best-so-far AUC deltas versus the same-initial target-only GP-UCB comparator.

## Route statistics

| Route | Tier | n | Mean AUC delta | 95% CI | W/T/L | BH-adjusted sign p |
|---|---|---:|---:|---:|---:|---:|
| molecular_freesolv_to_esol | retrospective_development | 0 | NA | not estimable | 0/0/0 | NA |
| molecular_freesolv_to_lipophilicity | retrospective_development | 0 | NA | not estimable | 0/0/0 | NA |
| materials_expt_gap_to_dielectric | retrospective_development | 0 | NA | not estimable | 0/0/0 | NA |
| materials_phonons_to_bulk_modulus | retrospective_development | 1 | 3.08253 | not estimable | 1/0/0 | 1.0 |
| materials_dielectric_to_jdft2d | retrospective_development | 0 | NA | not estimable | 0/0/0 | NA |
| materials_expt_gap_to_mp_gap | retrospective_development | 0 | NA | not estimable | 0/0/0 | NA |
| baumgartner_aniline_to_phenethylamine_alphos | post_freeze_extension | 0 | NA | not estimable | 0/0/0 | NA |
| baumgartner_aniline_to_benzamide_tbuxphos | post_freeze_extension | 0 | NA | not estimable | 0/0/0 | NA |
| baumgartner_aniline_to_phenethylamine_tbubrettphos | post_freeze_extension | 0 | NA | not estimable | 0/0/0 | NA |
| baumgartner_benzamide_tbubrettphos_to_alphos | post_freeze_extension | 0 | NA | not estimable | 0/0/0 | NA |
| reizman_cases_123_to_case4 | post_freeze_extension | 0 | NA | not estimable | 0/0/0 | NA |

## Interpretation

The repeated protocol estimates stochastic variation of the frozen online controller. Development and post-freeze routes remain labeled separately, and completion of this protocol does not convert retrospective routes into independent external validation.
