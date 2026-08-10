# Baumgartner warm-start v2 external confirmation

V2 adds an explicit source-quality floor to the three-candidate initial design.
It was created after v1 failed on new-substrate C-N campaigns, so those v1
targets are retained only as retrospective stress tests.

## Frozen development selection

Eight candidate policies were compared on nine real C-N development campaigns.
Each campaign used 100 frozen random-initial GP-UCB runs plus deterministic
space filling. A candidate first had to clear the task-level confidence and
non-loss gate; the eligible candidate with the largest expected AUC gain was
then selected.

The frozen v2 route is:

- source scope: same substrate first, then same precatalyst;
- first candidate: maximum median source rank;
- remaining two candidates: maximin distance inside the source top 50%;
- diversity weight inside the admissible region: `1.00`;
- mean development AUC delta: `+1.389941`;
- task-level 95% CI: `[+0.004589, +2.775293]`;
- task win/non-loss rates: `7/9` and `8/9`.

The server and local selection records are byte-identical (`SHA-256
d4123553facf0caf4f68fb15ce900d9a37a7eafd7255c92e24a7b1db9521cecd`).

## External target

The external confirmation uses the public Baumgartner Suzuki MINLP1 campaign as
the completed source and MINLP2 as the target. No MINLP2 outcome is loaded by
development selection. The target receives three initial observations and ten
target-only GP-UCB reveals, matched to 100 random-initial runs and deterministic
space filling.

## External confirmation result

The code, data version, configuration, selection record, and skill were committed
before MINLP2 was executed. The frozen v2 route produced:

- best-so-far AUC: `98.061210`;
- stronger non-transfer AUC: `89.400090` (space filling);
- AUC delta: `+8.661120`;
- final best yield: `100.000000`;
- mean random-initial final best: `94.767349`;
- final-best delta: `+5.232651`.

Across 100 random-initial runs, mean AUC was `88.840191` with a normal 95% CI
of `[87.344314, 90.336069]`. Fifteen random runs matched or exceeded the frozen
route's AUC, so this single-target result is a large practical gain but not a
task-level statistical confirmation.

The frozen route reached 100% yield after five total observations (three initial
plus two GP-UCB reveals). Only 46% of random-initial runs reached 100% within the
13-observation budget; among successful random runs, the median was seven total
observations. Space filling did not reach 95% yield.

## Post-hoc stress test

V2 was also rerun on the four already observed v1 targets. This is not held-out
evidence. It changed mean AUC delta from `-1.204666` to `+0.381563`, with three
task wins and one loss (`-0.722288`). Final best was non-inferior on all four
tasks. The result supports the diagnosed failure-mode fix but is kept separate
from the external confirmation.

Raw external and post-hoc metrics, stdout, and audits are under `confirmation/`
and `posthoc_stress/`.

## Claim boundary

The benchmark tests transfer of an initial-design procedure across two real
reaction-optimization datasets. It does not establish arbitrary cross-domain
transfer, LLM superiority, or continued source use after initialization.
