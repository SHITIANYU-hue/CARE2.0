# Classical transfer-BO confirmation protocol

## Question

Do CARE's source-informed policies improve target search beyond classical
transfer Bayesian optimization when source and target share a declared feature
space?

## Frozen comparison

`configs/classical_transfer_benchmark_v1.json` freezes the protocol before
held-out execution. Every method receives:

- the same source history;
- the same target candidate features;
- the same target initial observations for each seed;
- the same reveal budget;
- disjoint calibration and held-out seed ranges.

The classical arms are:

1. `target_gp_ucb`: target-only mixed-kernel GP-UCB;
2. `rgpe`: a ranking-weighted source/target GP ensemble;
3. `multitask_gp_icm`: a two-task intrinsic-coregionalization GP whose task
   correlation is selected online by target marginal likelihood. Its frozen
   grid includes zero and both positive and negative correlations.

The source GP uses a deterministic farthest-point sparse approximation capped
at 256 inducing observations. The cap, kernel, acquisition parameter, RGPE
sampling budget, and multi-task correlation grid are frozen in the config.

## Applicability boundary

These baselines are run only when source and target have identical ordered
decision columns. This includes the declared FreeSolv/Lipophilicity and
Matbench material-property pairs. ESOL and heterogeneous reaction pairs are
excluded because applying a same-space GP without first declaring a feature
mapping would not be a faithful baseline.

## Interpretation

Calibration is an offline benchmark split, not a wet-lab deployment gate. No
method or hyperparameter is selected from held-out results. A transfer method
is credited only when its paired held-out confidence interval against
`target_gp_ucb` excludes zero. CARE must additionally be compared on the same
seeds before claiming superiority over classical transfer BO.
