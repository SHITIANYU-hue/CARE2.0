# Source-outcome permutation falsification

The LLM-compiled role map, frozen patches, target anchor, target seeds, and 
budget are matched. Only the assignment of measured source outcomes to source 
candidates is permuted. Sequential transfer mass is zero, so this audit isolates 
source-informed initial design.

| Source -> target | True vs target AUC | True vs permuted AUC | Empirical p |
| --- | ---: | ---: | ---: |
| real_chemlex_acidamine -> real_buchwald_hartwig | +9.672 [+7.405, +11.939] | +20.347 [+20.347, +20.347] | 0.1500 |
| real_matbench_dielectric -> real_matbench_expt_gap | +48.954 [+44.050, +53.857] | +47.766 [+47.766, +47.766] | 0.1000 |
| real_moleculenet_lipophilicity -> real_moleculenet_freesolv | -1.027 [-1.228, -0.826] | +0.451 [+0.356, +0.545] | 0.2000 |

The paired intervals quantify target-seed variation conditional on the sampled
permutation ensemble; they are not the source-outcome randomization test. The
empirical randomization p-values compare the true assignment with the 19
permuted assignments, and none is below 0.05. These results therefore do not yet
show that the correct source feature-outcome association is more informative
than randomized outcome assignments. This retrospective falsification also does
not substitute for a fresh-task or prospective experiment.
