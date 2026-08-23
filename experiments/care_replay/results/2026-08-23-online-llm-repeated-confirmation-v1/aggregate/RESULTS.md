# online_llm_repeated_confirmation_v1

- Protocol complete: `false`
- Claim decision: `not_evaluated_incomplete_protocol`
- Completed trajectories: `11` / `330`
- Failed trajectories: `10`

Partial execution is operational evidence only. No confirmatory claim is evaluated until all declared trajectories are complete.

## Primary estimand

Equal-route-weighted mean of within-route mean best-so-far AUC deltas versus the same-initial target-only GP-UCB comparator.

## Route statistics

| Route | Tier | n | Mean AUC delta | 95% CI | W/T/L | BH-adjusted sign p |
|---|---|---:|---:|---:|---:|---:|
| molecular_freesolv_to_esol | retrospective_development | 1 | 0.0 | not estimable | 0/1/0 | NA |
| molecular_freesolv_to_lipophilicity | retrospective_development | 1 | 6.325 | not estimable | 1/0/0 | 1.0 |
| materials_expt_gap_to_dielectric | retrospective_development | 1 | 6.74515 | not estimable | 1/0/0 | 1.0 |
| materials_phonons_to_bulk_modulus | retrospective_development | 1 | 1.43505 | not estimable | 1/0/0 | 1.0 |
| materials_dielectric_to_jdft2d | retrospective_development | 1 | -0.52431 | not estimable | 0/0/1 | 1.0 |
| materials_expt_gap_to_mp_gap | retrospective_development | 1 | 2.44638 | not estimable | 1/0/0 | 1.0 |
| baumgartner_aniline_to_phenethylamine_alphos | post_freeze_extension | 1 | -0.71009 | not estimable | 0/0/1 | 1.0 |
| baumgartner_aniline_to_benzamide_tbuxphos | post_freeze_extension | 1 | -0.17809 | not estimable | 0/0/1 | 1.0 |
| baumgartner_aniline_to_phenethylamine_tbubrettphos | post_freeze_extension | 1 | 2.36616 | not estimable | 1/0/0 | 1.0 |
| baumgartner_benzamide_tbubrettphos_to_alphos | post_freeze_extension | 1 | 0.0 | not estimable | 0/1/0 | NA |
| reizman_cases_123_to_case4 | post_freeze_extension | 1 | 1.38 | not estimable | 1/0/0 | 1.0 |

## Interpretation

The repeated protocol estimates stochastic variation of the frozen online controller. Development and post-freeze routes remain labeled separately, and completion of this protocol does not convert retrospective routes into independent external validation.
