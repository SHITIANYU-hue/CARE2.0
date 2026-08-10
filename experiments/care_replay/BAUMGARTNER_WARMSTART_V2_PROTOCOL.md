# Baumgartner warm-start v2 protocol

## Motivation

The v1 task-disjoint confirmation exposed a concrete failure: diversity could
move the second and third initial experiments into regions that every source
expert ranked poorly. V2 keeps the source-guided first candidate but constrains
subsequent maximin design to a source-admissible quantile.

## Frozen candidate family

Each candidate route specifies:

- a public-descriptor source scope;
- a source-rank admissible quantile;
- a mixed-variable diversity weight.

For a `source_quantile` of `0.50`, all three initial candidates must remain in
the top half of the aggregated source rank. The first maximizes source rank;
the next two maximize minimum mixed-space distance inside that region. After
the initial three outcomes, source information is removed and every method uses
the same target-only GP-UCB for ten reveals.

## Development selection

The same nine C-N development campaigns are used. Random initialization is
estimated with 100 frozen seeds per task. Deterministic space filling is the
second baseline. A route is eligible only when its campaign-level AUC delta has
a 95% confidence-interval lower bound above zero and at least 80% of campaigns
do not lose to the stronger per-campaign baseline. Among eligible routes, v2
selects the largest mean AUC gain, with confidence-bound gain as the tiebreaker.

This ordering reflects the purpose of the gate: first reject unstable routes,
then optimize expected experimental efficiency among the accepted routes.

## External confirmation

The policy is selected without loading the external target outcomes. The new
confirmation task is the public Baumgartner Suzuki `MINLP2 optimization`
campaign, with `MINLP1 optimization` as the completed source. Both expose
precatalyst identity, temperature, residence time, and loading. The target is
compared against 100 random-initial GP-UCB runs and deterministic space filling
under the same 3+10 reveal budget.

This tests whether an initial-design rule developed on C-N campaigns can be
reused on a different reaction-optimization dataset. It does not claim arbitrary
cross-domain transfer or continuous source use after initialization.

## Retrospective stress test

The four v1 Baumgartner C-N evaluation campaigns may be rerun with v2 only as a
post-hoc stress test. They cannot be relabeled as new held-out evidence because
their v1 outcomes were already observed before v2 was finalized.
