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

## Status

The code, data version, configuration, selection record, and skill are committed
before the external target is executed. Confirmation results are added under
`confirmation/` without changing the frozen route.

## Claim boundary

The benchmark tests transfer of an initial-design procedure across two real
reaction-optimization datasets. It does not establish arbitrary cross-domain
transfer, LLM superiority, or continued source use after initialization.
