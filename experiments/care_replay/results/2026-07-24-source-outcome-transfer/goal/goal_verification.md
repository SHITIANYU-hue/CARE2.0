# CARE 2.0 source-outcome transfer held-out report

| Source -> target | Route | Final delta | AUC delta | Deployed composite 95% CI | Raw route composite | Result |
|---|---:|---:|---:|---:|---:|---|
| real_chemlex_acidamine -> real_buchwald_hartwig | transfer | +9.197 | +11.057 | [+15.847, +24.659] | +20.253 | significant positive |
| real_matbench_dielectric -> real_matbench_expt_gap | transfer | +32.699 | +48.028 | [+70.960, +90.494] | +80.727 | significant positive |
| real_matbench_expt_gap -> real_matbench_dielectric | transfer | +18.065 | +23.723 | [+36.825, +46.750] | +41.787 | significant positive |
| real_matbench_phonons -> real_matbench_dielectric | fallback | +0.000 | +0.000 | [+0.000, +0.000] | +1.051 | non-negative |
| real_moleculenet_esol -> real_moleculenet_lipophilicity | fallback | +0.000 | +0.000 | [+0.000, +0.000] | -4.049 | non-negative |
| real_moleculenet_freesolv -> real_moleculenet_lipophilicity | transfer | +2.976 | +3.727 | [+5.471, +7.936] | +6.704 | significant positive |
| real_moleculenet_lipophilicity -> real_moleculenet_freesolv | fallback | +0.000 | +0.000 | [+0.000, +0.000] | -0.802 | non-negative |

- All pairs non-negative: **True**
- Statistically positive pairs: **4/7**
- Majority statistically positive: **True**
- Goal achieved: **True**
