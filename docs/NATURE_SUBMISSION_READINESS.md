# CARE 2.0: Nature Submission Readiness

Updated 2026-08-23.

## Editorial position

CARE 2.0 should not be presented as a general-purpose agent that can already
discover new chemistry and materials across arbitrary domains. The defensible
paper is narrower and more distinctive:

> CARE 2.0 is an evidence-bounded controller for deciding when and how an LLM
> may reuse scientific experience in a new sequential experiment, while
> preserving a matched target-only comparator, recording every decision, and
> exposing negative transfer rather than hiding it.

This position separates CARE 2.0 from three established directions:

- [Coscientist](https://www.nature.com/articles/s41586-023-06792-0) demonstrates
  LLM-driven planning, tool use, and laboratory execution across chemical tasks.
  CARE 2.0 is not currently stronger on physical autonomy. Its intended
  contribution is controlled source-to-target learning under an explicit
  experimental budget.
- [ChemCrow](https://www.nature.com/articles/s42256-024-00832-8) integrates an
  LLM with 18 chemistry tools and evaluates task completion and expert-rated
  reasoning. CARE 2.0 instead evaluates whether transferred experience improves
  a sequential target experiment relative to a matched target-only policy.
- [A-Lab](https://www.nature.com/articles/s41586-023-06734-w) closes a physical
  materials-synthesis loop and reports 36 successful syntheses among 57 targets.
  CARE 2.0 currently uses finite-pool replay and cannot claim equivalent
  experimental discovery until a prospective wet-lab campaign is completed.
- [The AI Scientist](https://www.nature.com/articles/s41586-026-10265-5)
  automates an end-to-end machine-learning research cycle. CARE 2.0 addresses a
  different question: whether an LLM can make auditable, evidence-conditioned
  transfer decisions inside costly scientific optimization.

The novelty claim must therefore rest on the combination of transferable
hypotheses, matched causal evaluation of online LLM decisions, explicit
negative-transfer control, and complete decision traces. Any one of these in
isolation is not enough.

## What can be claimed now

The current Opus suite supports the following statements:

1. The LLM receives a real choice among executable experiments on every online
   round and can override the target-only GP default.
2. The online-controller increment is isolated from initial design by comparing
   against the same LLM initial observations followed by target-only GP-UCB.
3. Six of eleven source-target routes improve best-so-far AUC; two tie and three
   lose.
4. A five-route post-freeze chemistry extension is majority positive at the
   route-count level: three wins, one tie, and one loss.
5. Every proposal, critic response, selected candidate, reveal, prediction
   error, hypothesis status, and fallback is retained in the trace.

The current evidence does not support these statements:

1. Cross-domain transfer is statistically significant in general.
2. CARE 2.0 eliminates negative transfer.
3. The full LLM system consistently beats the fixed initializer.
4. The LLM updates its model parameters or has already implemented automatic
   persistent skill evolution.
5. The system has accelerated a new physical wet-lab discovery.

## Current statistical result

The primary comparison is online Opus versus the same LLM initial design plus
target-only GP-UCB, under an identical target-reveal budget.

| Evidence tier | Routes | Mean AUC delta | Bootstrap 95% interval | Win / tie / loss | Two-sided sign test |
|---|---:|---:|---:|---:|---:|
| All current routes | 11 | +1.0135 | [-0.4433, +2.7821] | 6 / 2 / 3 | p=0.5078 |
| Retrospective development | 6 | +1.8403 | [-0.8465, +4.6151] | 3 / 1 / 2 | p=1.0000 |
| Post-freeze chemistry extension | 5 | +0.0212 | [-0.5171, +0.4194] | 3 / 1 / 1 | p=0.6250 |

These intervals resample source-target routes. Several routes share task
families, and each route currently contains one online Opus trajectory, so the
analysis is a sensitivity check rather than a population-level hierarchical
estimate.

## Frozen repeated-trajectory protocol

The first confirmatory gap now has an executable frozen protocol in
`experiments/care_replay/configs/online_llm_repeated_confirmation_v1.json`.
It includes all 11 routes from the evidence audit rather than selecting routes
by their observed sign, declares 30 independent API trajectories per route,
fixes the model, temperature, prompt/controller path, target budget, menu, and
same-initial GP-UCB comparator, and disables confirmatory inference until all
330 trajectories are complete.

The primary estimand is the equal-route-weighted mean of the within-route mean
best-so-far AUC deltas. The predeclared success rule requires the two-sided
hierarchical-bootstrap 95% interval to lie entirely above zero. Route-level
sign tests are secondary and use Benjamini-Hochberg correction. API failures,
negative routes, token use, GP overrides, critic revisions, and source-active
rates are retained. The runner writes a hash lock over the protocol, analysis
script, route configs, and initial records before the first model call.

This protocol addresses stochastic repeatability, but it does not turn the six
development routes into independent confirmation and it does not replace a new
task family or prospective experiment.

### Repetition pilot and operational amendment

One independent trajectory has completed for each of the 11 routes under the
v1 lock (11/330 total). The equal-route mean AUC delta is +1.7532, with six
positive, two tied, and three negative routes. This is an incomplete pilot, so
no confidence interval or confirmatory decision is reported. More importantly,
two route signs changed relative to the original one-trajectory audit. The
experimental-gap to Materials-Project-gap route changed from -2.4464 to
+2.4464 AUC, while the aniline to phenethylamine/AlPhos route changed from a
small positive effect to -0.7101. The repeated execution therefore confirms the
methodological concern that a single stochastic LLM trajectory cannot establish
route-level transferability.

The next API batch stopped at an account-level HTTP 429 cost cap. To separate
such infrastructure events from model or scientific failures, v2 freezes the
same controller, route panel, target budget, comparator, and analysis, while
adding an auditable retry policy. Rate limits, cost caps, timeouts, and network
failures may be retried up to three times; every failed attempt is retained.
Malformed model outputs are terminal, and a poor scientific outcome never
triggers a retry. Confirmatory inference remains disabled unless all 330
declared trajectories finish successfully. This operational amendment is in
`online_llm_repeated_confirmation_v2.json` and does not alter the scientific
decision policy.

## What a reviewer is likely to challenge

### 1. The strongest gains are development results

The largest positive effects occur on FreeSolv to Lipophilicity and experimental
band gap to dielectric, both used during controller development. They are useful
for mechanism discovery but cannot carry the headline generalization claim.

Required response: freeze the code, prompt, model configuration, candidate-menu
policy, and analysis before revealing a new task family.

### 2. The frozen extension is too small and uncertain

Three of five chemistry routes are positive, but the average gain is close to
zero and the uncertainty interval crosses zero.

Required response: run repeated online LLM trajectories on predeclared routes,
then add a genuinely new task family. Report hierarchical or clustered
uncertainty rather than treating every route as fully independent.

### 3. Initial design weakens the complete system

The online controller can add value after a shared initial design, while the
complete LLM scientist can still lose because the outcome-blind initial design
is unstable.

Required response: treat initial design and online control as separate modules;
either improve the initial design prospectively or present the online controller
as the paper's primary intervention.

### 4. Replay is not a new discovery

Public measured datasets make the comparison reproducible, but the target
outcomes already exist in the replay table.

Required response: add a paired prospective campaign. A minimal credible design
uses the same starting observations and experiment budget for CARE 2.0 and a
target-only GP baseline, with the next experiment chosen before each result is
measured.

### 5. LLM self-improvement is currently overstated

The online loop updates context, posterior state, hypothesis, and the next
decision. It does not train model weights, and the online trajectory is not yet
automatically distilled into a persistent reusable skill.

Required response: use the terms "evidence-conditioned hypothesis revision" and
"external trace memory" for the current system. Reserve "self-evolving skill
library" for a separately implemented and evaluated module.

## Minimum Nature-level evidence package

### P0: required before a strong submission

1. **Frozen confirmation protocol.** Publish the route list, data split, model,
   prompts, stopping rule, candidate-menu policy, metric, and analysis script
   before running the final evaluation.
2. **Repeated online trajectories.** Use predeclared LLM seeds or independent
   API calls per route, preserving all failures. Pair each run with the same
   initial observations and target-only GP comparator.
3. **Fresh task family.** Add at least one domain not used during prompt or
   controller development. A new route inside the same C-N dataset is useful but
   insufficient for a broad cross-domain claim.
4. **Prospective experimental validation.** Run at least one wet-lab or genuinely
   prospective closed-loop campaign with matched budget and starting state.
5. **Complete-system accounting.** Report the online increment, initial-design
   effect, and complete system separately. Do not let a strong online result hide
   a weak initial design.

### P1: needed to make the paper distinctive

1. Distill successful and failed trajectories into versioned skill artifacts,
   then evaluate whether those artifacts transfer without target-outcome access.
2. Add an abstention endpoint: the controller should be rewarded for declining
   unsupported transfer, not only for choosing a candidate.
3. Have domain experts score the scientific plausibility, falsifiability, and
   post-reveal revision quality of blinded hypothesis traces.
4. Report experiment count, token cost, latency, model/API version, parse errors,
   fallback rate, and negative-transfer rate.
5. Test model robustness by rerunning a frozen subset with at least one different
   strong LLM.

### P2: useful supporting evidence

1. Candidate-menu size and GP-rank bound sensitivity.
2. Proposer-only versus proposer-plus-critic ablation.
3. Source evidence, target observations, and generic scientific prior ablations.
4. Skill retrieval and knowledge-base update ablations.
5. Calibration curves for predicted improvement probability versus realized
   improvement.

## Nature Portfolio reproducibility requirements

[Nature Portfolio reporting policy](https://www.nature.com/nphys/editorial-policies/reporting-standards)
requires central custom code to be available to editors and reviewers and
expects code, data, materials, and protocols to be made available for
replication. The manuscript package should therefore include:

- a DOI-minted release of the exact code and configuration used for final
  figures, preferably through Zenodo or Code Ocean;
- a data-availability statement identifying every public dataset and every
  restriction on private or wet-lab data;
- a code-availability statement with commit, release tag, environment, model
  endpoint, and reproduction commands;
- source data for every figure, including route-level and trajectory-level
  records;
- complete prompts, normalized LLM responses, parser failures, fallbacks, and
  SHA-256 fingerprints;
- exact sample sizes, the unit of analysis, whether observations are repeated,
  the statistical tests and sidedness, effect sizes, and uncertainty intervals;
- an explicit statement that human authors remain responsible for scientific
  conclusions and that LLM outputs were audited.

The [Nature Portfolio AI policy](https://www.nature.com/nature/editorial-policies/ai)
states that scholarly judgement and accountability remain human. CARE 2.0's
audit trail and calibrated decision boundary can be framed as technical support
for that requirement, but they do not transfer accountability to the model.

## Recommended paper structure

1. **Problem:** scientific experience can help a new optimization campaign, but
   uncontrolled transfer causes negative experiments and invalid evidence reuse.
2. **Method:** source evidence is translated into a falsifiable hypothesis; a
   matched target-only model defines the default; the LLM chooses among bounded
   executable candidates; each reveal updates the hypothesis and trace.
3. **Causal evaluation:** same initial observations, same target budget, one
   intervention difference: online LLM control.
4. **Development evidence:** show where the mechanism was discovered, without
   presenting these tasks as confirmation.
5. **Frozen confirmation:** repeated trajectories on untouched task families and
   a prospective campaign.
6. **Failure analysis:** negative routes, initial-design failures, unbounded
   overrides, and abstention behavior.
7. **Reusable scientific memory:** skill artifacts derived from both successful
   and failed traces, evaluated on a later frozen task.

## Submission decision

The current package is not ready for a strong claim at Nature or Nature
Communications. The method and audit architecture are promising, but the
confirmatory effect and prospective evidence are not yet strong enough. The
highest-value next move is not another development sweep. It is a frozen,
repeated, cross-family confirmation followed by one paired prospective campaign.
