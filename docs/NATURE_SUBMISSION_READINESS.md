# CARE 2.0: Nature Submission Readiness

Updated 2026-08-25.

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
3. In the frozen 11-route first-trajectory portfolio, six routes improve
   best-so-far AUC, two tie, and three lose relative to same-start target GP.
4. The round-three threshold-5 gate improves the original 11-route development
   panel but fails on six route-disjoint retrospective routes. A training-only
   comparison of 115 controller candidates instead selects a one-round bounded-
   authority policy; on the six disjoint routes it changes the mean from -1.056
   for full online LLM to +0.612 versus target GP and reduces losses from two to
   one. The interval crosses zero, so this is a mechanism and robustness signal.
5. The complete 17-route retrospective inventory, including six earlier
   negative or null development routes, has a smaller and uncertain mean effect
   of +0.762 versus target GP, preventing selective route reporting.
6. Every proposal, critic response, selected candidate, reveal, prediction
   error, hypothesis status, and fallback is retained in the trace.
7. A frozen source-outcome assignment falsification was completed on three
   routes with 100 target seeds and 19 matched outcome permutations per route.
   Reaction and materials warm starts beat target-only, but none of the true
   source-outcome assignments rejected the permutation null at 0.05.
8. A production-path hidden-label non-interference audit passed on all 170
   archived request states across the complete 17-route portfolio. The audit
   permuted 2,725,689 state-level unrevealed label positions while preserving
   revealed outcomes and public candidate attributes; source priors, executable
   menus, diagnostics, and current-code LLM prompts remained exactly invariant.
   Archived prompt SHA-256 values were also verified. This supports the
   implementation information boundary, not LLM efficacy.
9. The final LLM probability of improving the current best was scored on every
   archived decision. Across 170 decisions and 17 routes, 27 improved the
   pre-round best. The LLM Brier score was 0.133 versus 0.134 for the production
   GP probability on the same executed candidate. The equal-route difference
   was -0.0016 with a route-bootstrap 95% interval [-0.0132, +0.0090], so there
   is no stable calibration advantage. The LLM was modestly optimistic: mean
   probability 0.219 versus observed improvement rate 0.159.
10. A frozen external task-family stress test was completed on the official
    FLIP2 Hydrophobic Core `to-P06241` wild-type split. Neither this protein-
    engineering family nor its target was used for controller development.
    Across 100 paired target seeds, multisource RGPE, ICM-BMA, and a fixed skill
    prior all transferred negatively relative to target-only GP-UCB, with mean
    AUC deltas of -16.959, -8.133, and -23.008. In one live Opus trajectory,
    the LLM downgraded the source hypothesis, selected GP rank one, and set
    `continue_source_transfer=false`; the online increment was zero, while the
    full LLM trajectory was 7.186 AUC below the fixed-design comparator.
11. A source-only family gate was audited without using P06241 outcomes. On two
    50-seed pseudo-target rotations between the completed P01053 and P0A9X9
    backbones, RGPE, ICM-BMA, and the fixed skill prior all had paired 95%
    intervals below zero. The frozen eligibility rule therefore rejected every
    transfer policy and deployed target-only GP-UCB on P06241. The otherwise
    selected ICM-BMA policy lost 8.133 AUC, so the fallback avoided that loss.
    The gate was designed after the external result was observed and is
    therefore a retrospective mechanism audit, not prospective confirmation.
12. The unchanged source-only gate was then preregistered on the official FLIP2
    Rhodopsin by-wild-type benchmark before the raw file was downloaded. The
    config, code hashes, eligibility rule, fallback, two 50-seed source-only
    calibration routes, and 100-seed official-test deployment were committed as
    `4130dd4`. No method cleared the requirement that its paired AUC 95% interval
    lower bound exceed zero on both calibration routes. The ungated selector
    would have deployed the fixed skill prior, which lost 1.548 AUC on the
    untouched official test [95% CI -1.996, -1.099]. The gate deployed target-
    only GP-UCB and avoided that loss. This is prospective negative-transfer
    prevention on one external protein family, not positive-transfer efficacy.
13. A separate positive-transfer protocol was preregistered on the official
    FLIP2 Imine Reductase `two-to-many` split before the raw file was downloaded.
    Claude Opus 4.8 received only public task metadata and chose a bounded
    additive-mutation skill over abstention. The config, implementation hashes,
    LLM request/response hash, eligibility rule, and 200 evaluation seeds were
    committed as `5389a9c`. The skill passed train-to-validation calibration
    and improved best-so-far AUC on the untouched official test by +0.389
    [95% paired-bootstrap interval +0.274, +0.503], with 72 wins, nine ties,
    and 19 losses. A predeclared fixed skill prior improved more (+0.864), so
    this is positive evidence for one LLM-generated skill, not LLM superiority.
14. The IRED result was subjected to a source-outcome assignment falsification.
    The target seeds, initial observations, budget, GP kernel, additive executor,
    source rows, outcome marginal distribution, and decaying authority were held
    fixed; only the binding between 3,500 measured source outcomes and mutation
    features was changed across 99 deterministic permutations. The true binding
    exceeded the mean permuted skill by +0.340 AUC [95% target-seed bootstrap
    interval +0.259, +0.422], but six permuted bindings achieved an equal or
    larger mean AUC effect, giving a one-sided assignment-level randomization
    value of p=0.07. This is suggestive mechanism evidence, not rejection of the
    source-assignment null.

The current evidence does not support these statements:

1. Cross-domain transfer is statistically significant in general.
2. CARE 2.0 eliminates negative transfer.
3. The full LLM system consistently beats the fixed initializer.
4. The LLM updates its model parameters or has already implemented automatic
   persistent skill evolution.
5. The system has accelerated a new physical wet-lab discovery.
6. The correct source feature-outcome association has been shown to carry more
   transferable information than randomized source-outcome assignments.
7. The LLM's self-reported improvement probability is prospectively calibrated
   or can already serve as a validated abstention gate.
8. The Hydrophobic Core task establishes positive cross-domain efficacy. It
   establishes a negative-transfer boundary and one auditable abstention case
   only. The later IRED result is positive within one enzyme family and does not
   establish universal cross-domain efficacy.
9. The source-only family gate eliminates negative transfer in general. It now
   has one prospective external-family confirmation, but that single successful
   veto does not establish universal safety or positive transfer efficacy.
10. The prospective IRED gain proves that the LLM identified the uniquely
    correct source feature-outcome mechanism. The later assignment audit was
    post-hoc and did not reject its 99-permutation null at 0.05.

## Current statistical result

The primary comparison is online Opus versus the same LLM initial design plus
target-only GP-UCB, under an identical target-reveal budget.

| Evidence tier or method | Routes | Mean AUC delta | Route-bootstrap 95% interval | Win / tie / loss | Two-sided sign test |
|---|---:|---:|---:|---:|---:|
| Frozen 11-route online LLM versus target GP | 11 | +1.7532 | [+0.4147, +3.3149] | 6 / 2 / 3 | p=0.5078 |
| Frozen 11-route cross-validated calibration gate versus target GP | 11 | +2.0357 | [+0.4245, +3.9006] | 5 / 4 / 2 | p=0.4531 |
| Frozen 11-route fixed-threshold-5 gate replay versus target GP | 11 | +2.1346 | [+0.5146, +4.0010] | 6 / 3 / 2 | p=0.2891 |
| Complete 17-route retrospective inventory versus target GP | 17 | +0.7618 | [-0.5008, +2.0907] | 8 / 4 / 5 | p=0.5811 |
| Route-disjoint full online LLM versus target GP | 6 | -1.0558 | [-3.0103, +0.1539] | 2 / 2 / 2 | p=1.0000 |
| Route-disjoint fixed round-three threshold-5 gate versus target GP | 6 | -1.9677 | [-6.4382, +0.8731] | 2 / 2 / 2 | p=1.0000 |
| Route-disjoint selected one-round bounded authority versus target GP | 6 | +0.6124 | [-0.4422, +1.7934] | 2 / 3 / 1 | p=1.0000 |

These intervals resample source-target routes, not independent LLM calls.
Several routes share task families, and each route currently contains one
online Opus trajectory, so they quantify benchmark-route variation rather than
within-route stochastic uncertainty. Every gate/controller row is a
retrospective replay. The six evaluation routes do not enter controller
selection, but their outcomes were previously known, so they are not an
external or prospective test.

## External task-family stress test

The FLIP2 Hydrophobic Core evaluation is the first genuinely new task family
added after the controller-development portfolio. The official `to-P06241`
split contains 24,935 measured sequences: P01053 and P0A9X9 are completed
sources and P06241 is the 9,972-candidate target. The config, 100 seeds, target
budget, kernel, source counts, and comparator identities were frozen before
execution. `protocol_lock.json` records the canonical config, raw-file, and
implementation hashes. No target outcome entered parameter or controller
selection.

| Policy | Mean AUC | Delta vs target GP [95% paired interval] | AUC win rate |
|---|---:|---:|---:|
| Target-only GP-UCB | 81.949 | reference | reference |
| Multisource RGPE | 64.990 | -16.959 [-18.407, -15.511] | 0.00 |
| Multisource ICM-BMA | 73.816 | -8.133 [-9.872, -6.394] | 0.12 |
| Fixed multisource skill prior | 58.941 | -23.008 [-25.657, -20.359] | 0.00 |

The live Opus hypothesis expected low aromatic content and high beta-branched
Val/Ile packing to transfer across backbones. After three initial P06241
measurements, the best target observation instead supported a low-diversity,
methionine-rich region. At the first online round, proposer and critic both
abandoned direct source transfer and chose target-GP rank one. The one-round
authority limit then avoided 11 later LLM rounds, or 22 nominal proposer/critic
calls. This is a useful case because the reasoning trace explains why transfer
was rejected and the executed action matches the safe target-only default.
It is not a positive-performance result: online delta was 0.000, and the LLM-
generated initial design was worse than the fixed initializer in this single
trajectory.

### Source-only family gate

The follow-up audit asks a more operational question: can the completed source
campaigns veto unsafe transfer without reading any deployment-target outcome?
P01053 and P0A9X9 were alternately treated as pseudo-targets for 50 paired seeds
per direction. A transfer method was eligible only when its paired 95%
confidence-interval lower bound exceeded zero on both source-only routes.
P06241 was prohibited from calibration and method selection.

| Method | P01053 to P0A9X9 delta [95% CI] | P0A9X9 to P01053 delta [95% CI] | Eligible? |
|---|---:|---:|---:|
| RGPE | -6.743 [-8.466, -5.020] | -5.573 [-11.073, -0.073] | No |
| ICM-BMA | -3.361 [-4.739, -1.984] | -8.158 [-13.936, -2.380] | No |
| Fixed skill prior | -12.702 [-14.327, -11.078] | -27.288 [-32.522, -22.054] | No |

All three methods were rejected. An ungated source-only selector would have
chosen ICM-BMA as the least harmful calibration method; it lost 8.133 AUC on
P06241. The frozen fallback instead selected target-only GP-UCB and avoided
that loss. This is useful evidence that target labels are not technically
required for the veto decision. It is not evidence of prospective safety,
because the gate was designed after the P06241 stress-test result was known.
This requirement defined the next test: preregister the exact rule, route
construction, and fallback on a second external family before any target
outcome is inspected.

### Prospective Rhodopsin family-gate confirmation

That confirmatory test has now been completed on a second external family. The
official FLIP2 Rhodopsin by-wild-type split contains 884 measured variants from
75 wild types. The official train, validation, and test partitions contain 584,
116, and 184 variants and correspond to 5, 34, and 36 wild types. The raw file
was absent when the protocol was frozen. Commit `4130dd4` records the config,
implementation hashes, source-only routes, eligibility rule, fallback, and 200
evaluation seeds before download.

| Method | Train to validation delta [95% CI] | Validation to train delta [95% CI] | Official test delta [95% CI] | Eligible? |
|---|---:|---:|---:|---:|
| RGPE | -0.298 [-0.922, +0.325] | -0.263 [-1.252, +0.727] | -3.819 [-4.498, -3.140] | No |
| ICM-BMA | -0.791 [-1.397, -0.185] | -0.965 [-1.660, -0.270] | -1.539 [-2.066, -1.011] | No |
| Fixed skill prior | +0.384 [-0.301, +1.069] | +1.192 [+0.271, +2.112] | -1.548 [-1.996, -1.099] | No |

The fixed skill prior had the best route-equal calibration mean (+0.788), but
its first-route confidence interval crossed zero, so the frozen gate rejected
it and fell back to target-only GP-UCB. The official test was read only after
that decision and showed that the ungated choice would have caused a stable
1.548 AUC loss. This converts the earlier retrospective mechanism into one
prospective external safety result. It does not show positive external efficacy,
universal negative-transfer control, or prospective wet-lab discovery.

### Prospective IRED positive-transfer confirmation

The official FLIP2 IRED `two-to-many` split asks whether completed zero-, one-,
and two-mutation variants can guide a budgeted search among variants with three
to fifteen mutations. Before downloading the file, Claude Opus 4.8 received
only public task metadata and an executable menu containing a shrunk additive-
mutation skill or abstention. It selected the additive skill and recorded
epistasis, unseen positions, distribution shift, and excessive early source
authority as failure conditions. The prompt, raw response, parsed hypothesis,
usage, and hashes are archived.

| Method | Train to validation delta [95% paired-bootstrap interval] | Official test delta [95% paired-bootstrap interval] | Test win / tie / loss |
|---|---:|---:|---:|
| RGPE | +0.137 [+0.068, +0.209] | -0.085 [-0.207, +0.032] | 44 / 13 / 43 |
| ICM-BMA | +0.032 [-0.032, +0.098] | -0.194 [-0.302, -0.091] | 23 / 17 / 60 |
| Fixed skill prior | +0.489 [+0.385, +0.595] | +0.864 [+0.709, +1.019] | 78 / 8 / 14 |
| LLM additive skill | +0.468 [+0.366, +0.574] | +0.389 [+0.274, +0.503] | 72 / 9 / 19 |

The preregistered success rule is met because the LLM skill was deployed and
its official-test paired AUC interval is entirely above zero. Its final-best
interval crosses zero and its top-10 hit rate is lower than target GP, so the
claim is specifically earlier discovery under the AUC endpoint. The fixed prior
is stronger. This closes one external positive-efficacy gap but does not close
the wet-lab or broad cross-domain generalization gaps.

### IRED source-outcome mechanism falsification

The positive IRED result leaves a harder attribution question: did the additive
skill benefit from the measured mutation-activity binding, or could an arbitrary
source ranking have produced a similar early-search gain? A post-hoc mechanism
audit retained the official test target, all 100 target seeds, the three initial
observations, twelve-reveal budget, GP kernel, source rows, source outcome
marginal distribution, additive executor, and 0.50-to-0.10 authority schedule.
It then compared the true binding with 99 deterministic outcome permutations
over the same 3,500 source variants.

| Mechanism comparison | Mean paired AUC effect | 95% interval | Assignment-level result |
|---|---:|---:|---:|
| True source binding versus target GP-UCB | +0.389 | [+0.274, +0.504] | 72 / 9 / 19 target-seed W/T/L |
| Mean outcome-permuted skill versus target GP-UCB | +0.049 | [-0.037, +0.132] | 46 of 99 assignment means positive |
| True binding versus the per-seed permutation mean | +0.340 | [+0.259, +0.422] | six null assignments at least as large |

The paired target-seed interval for true minus permutation mean is entirely
above zero, so the true binding is more useful than the average false binding.
However, the finite randomization test asks the stricter question of where the
true assignment ranks among independently permuted assignments. It ranks seventh
of 100, corresponding to a one-sided plus-one-corrected p-value of 0.07. Thirty
permuted assignments also had target-seed intervals above zero versus target GP.
The defensible conclusion is therefore that the measured binding carries a
suggestive, non-unique signal. The audit does not establish that the LLM found
the uniquely correct causal mechanism, and it reinforces the need for an
independent prospectively frozen source-assignment test.

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
development routes into independent confirmation. The FLIP2 Hydrophobic Core
stress test supplies a negative external efficacy result, Rhodopsin supplies a
prospective safety result, and IRED supplies one prospective positive finite-
pool result. Replication across additional external families and a prospective
laboratory experiment remain necessary.

The executable-gate protocol is separately frozen in
`experiments/care_replay/configs/online_llm_gated_repeated_confirmation_v1.json`.
It keeps the same 11 routes and 30 trajectories per route, fixes evaluation
after reveal three and one global MAE threshold of five, and predeclares that a
triggered trajectory makes no later LLM request. Target-only GP-UCB continues
from all accumulated observations, so the target budget is unchanged while LLM
rounds and token use can fall. This 330-trajectory protocol has not completed;
no prospective gate claim is currently evaluated.

This first gate protocol is retained unchanged as a frozen audit artifact, but
the route-disjoint failure means it is no longer the primary protocol proposed
for execution. The current primary controller protocol is
`online_llm_bounded_authority_route_disjoint_confirmation_v1.json`. The policy
was selected on 11 routes without using the six evaluation-route metrics. It
permits one online proposer/critic decision, then makes no later LLM request and
continues target-only GP-UCB from every accumulated observation. The protocol
declares six routes, 30 independent trajectories per route, no optional
stopping, and a hierarchical-bootstrap success rule that is disabled until all
180 trajectories complete. Because route identities and one earlier outcome
trajectory per route were already known, completion will establish internal
stochastic replication, not external-task or wet-lab validation.

### Repetition pilot and operational amendment

One independent trajectory has completed for each of the 11 routes under the
v1 lock (11/330 total). The equal-route mean AUC delta is +1.7532, with six
positive, two tied, and three negative routes. This is an incomplete pilot, so
no confidence interval or confirmatory decision is reported. More importantly,
four of the eleven paired routes strictly reversed sign relative to the early
one-trajectory audit: experimental gap to Materials Project gap, aniline to
phenethylamine/AlPhos, aniline to benzamide/tBuXPhos, and aniline to
phenethylamine/tBuBrettPhos. Both batches still have the same aggregate count
of six wins, two ties, and three losses, showing that an unchanged portfolio
count can hide route-level instability. The mean absolute paired AUC change is
1.0479. Because the early calls predate the confirmation lock, this is a
descriptive stochasticity audit rather than a confirmatory repeated-measures
test.

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

The first v2 trajectory, Reizman cases 1--3 to case 4, completed without a
retry. Its AUC delta was +0.3200 and its final-best delta was +0.4000, compared
with +1.3800 and +5.7000 in the v1 trajectory. The direction remained positive,
but the smaller magnitude reinforces the need for repeated calls. Three
additional v2 trajectories reached an account-level insufficient-balance error;
each retained all three predeclared infrastructure attempts and produced no
scientific result. They are reported as infrastructure failures, not negative
transfer or model failures.

### Cross-model robustness protocol

A separate GLM-5.3 protocol now freezes the same eleven routes, initial target
observations, reveal budget, candidate-menu policy, and same-initial GP-UCB
comparator. It declares 30 trajectories per route and retains the same retry
and incomplete-protocol rules. GLM-5.3 is analyzed independently rather than
pooled with Opus after outcomes are observed. The primary analysis again asks
whether the equal-route mean AUC interval lies above zero; agreement with Opus
route directions is secondary. This experiment addresses model dependence, but
does not replace a new task family or prospective campaign.

The first GLM-5.3 pilot did not produce a complete scientific trajectory. The
standard client initially returned valid structured decisions, but account
rate limits exhausted the predeclared infrastructure retries. A second frozen
version added provider-aware request pacing, `reasoning_effort=low`, and
in-place retry of the identical failed HTTP request. It passed the rate limit
that interrupted the first pilot, but a later critic returned an otherwise
valid decision inside a persistent top-level `answer` object. The strict schema
validator terminated that trajectory as a model-format failure.

A third version adds one transparent transport adapter: when and only when
`answer` is the sole top-level key and its value is an object, the envelope is
removed before the unchanged scientific validator runs. The adapter cannot
rename fields, substitute candidates, coerce values, fill missing fields, or
repair scientific content. Its first execution exhausted eight in-place HTTP
retries over approximately twelve minutes without receiving one usable
response, and the interrupted outer retry is retained as a second
infrastructure attempt. The earlier infrastructure and model-format failures
remain in their original audit trees and are not reclassified. No GLM
trajectory is complete, so no GLM effect estimate or cross-model agreement
claim is reported. The version history and precise invariants are recorded in
`docs/GLM53_CROSS_MODEL_PROTOCOL.md`.

This design follows two relevant evaluation precedents. A recent
[Communications Materials agent study](https://www.nature.com/articles/s43246-025-00994-x)
repeated identical scientific-agent prompts five to ten times to assess
reliability, while the
[AILA study in Nature Communications](https://www.nature.com/articles/s41467-025-64105-7)
compared four foundation models on the same scientific automation benchmark.
More generally, Blackwell, Barry, and Cohn's
[LLM uncertainty analysis](https://arxiv.org/abs/2410.03492) shows why repeated
calls and uncertainty estimates are needed even under nominally fixed model
settings. CARE 2.0 combines both controls: within-model repetition and a
separately frozen cross-model replication.

### Context-preserving online decisions and outcome-scale contract

The transport-robust development protocol has now completed one trajectory in
each of three task families. Relative to the same-initial target-only GP-UCB,
FreeSolv to Lipophilicity improved best-so-far AUC by 6.325 and final best by
4.875; phonon peak to bulk modulus improved AUC by 3.240 and final best by
4.324; Reizman Suzuki cases 1--3 to case 4 improved AUC by 0.320 and final best
by 0.400. LLM decision authority was 100% in all three trajectories, and the
controller overrode the GP rank-one candidate in 40%, 40%, and 70% of online
rounds, respectively. These are single development trajectories, not
confirmatory estimates.

The mechanisms differ. In the molecular route, source transfer remained active
in five of ten rounds, providing a positive source-outcome transfer example. In
the materials route, the controller disabled direct phonon-to-modulus transfer
in every round after judging the source relation unsupported, but still used
the revealed target history to reroute four GP choices. That result is evidence
for negative-transfer abstention and semantic target adaptation, not evidence
that phonon outcomes directly predict bulk modulus. The Suzuki trajectory kept
source transfer active in five rounds and produced only a small increment over
the matched GP; it remained 4.810 AUC points below the earlier fixed-v2 CARE
trajectory. This distinction prevents a positive online increment from being
misreported as superiority over every baseline.

The context-preserving repair audit also exposed an interpretation error: the
LLM labelled a normalized bulk-modulus score as GPa because the prompt named
the objective but did not state its transformation. A new frozen development
amendment now supplies, for every source and target dataset, the raw quantity,
raw unit, replay-score definition, optimization direction, and the instruction
never to attach a raw unit to a transformed score. It does not reveal hidden
candidate values or change the candidate menu, budget, comparator, or validator.

The first trajectory under this amendment improved AUC by 3.0825 and final best
by 2.7458 relative to the same-initial target-only GP-UCB. More importantly for
the stated repair, a deterministic audit over parsed proposer and critic
responses found five numeric score--GPa couplings in the pre-amendment trace
and zero among 20 response events after the contract was added. Both trace
hashes and every matched pre-amendment sentence are retained in
`audit/outcome_semantics_audit.json`. Because these are different stochastic
development trajectories, the performance values are not a causal comparison
of prompt versions. The valid conclusion is narrower: the declared response-
level unit failure disappeared in this post-amendment material trajectory.

### Direct classical transfer-BO comparison

CARE should be compared with transfer Bayesian optimization, not only with a
target-only GP. The repository already contains a calibration-only model
selection benchmark over target GP-UCB, RGPE, and multitask GP-ICM, followed by
100 disjoint held-out seeds. The older fixed CARE source-outcome router beats
the calibration-selected classical method on three routes with intervals above
zero, loses decisively on dielectric to experimental band gap, and is
indistinguishable on phonons to dielectric AUC:

| Route | Calibration-selected baseline | CARE mean AUC delta [95% interval] |
|---|---|---:|
| FreeSolv to Lipophilicity | target GP-UCB | +1.578 [0.621, 2.534] |
| Lipophilicity to FreeSolv | target GP-UCB | +42.995 [40.287, 45.703] |
| Experimental gap to dielectric | multitask GP-ICM | +2.884 [0.321, 5.447] |
| Dielectric to experimental gap | RGPE | -44.838 [-49.274, -40.403] |
| Phonons to dielectric | multitask GP-ICM | +0.045 [-1.911, 2.002] |

This benchmark is consistent with the design principle in
[networked autonomous materials exploration](https://www.nature.com/articles/s41524-025-01851-8):
when external knowledge is irrelevant, model selection should fall back to a
non-transfer model. CARE's distinctive claim is not that an LLM universally
beats transfer GP methods; it is that an LLM can express a falsifiable
source-target hypothesis, revise or abstain online, and leave an auditable
scientific trace. A Nature-level comparison still requires the online LLM
controller, RGPE, multitask GP, and target-only GP to be run under one newly
frozen repeated protocol.

As a bridge to that protocol, the three context-preserving online trajectories
have now been replayed against target-only GP-UCB, RGPE, and multitask GP from
the exact LLM initial candidate IDs and the same ten-reveal budget. The rebuilt
target-only GP metrics match the values stored by the online runner, providing
an executable equality check on the starting state and budget. Relative to the
strongest realized baseline by AUC, the online LLM gains were +6.325 for
FreeSolv to Lipophilicity, +1.922 for phonons to bulk modulus, and -1.960 for
Reizman cases 1--3 to case 4. The corresponding equal-route mean was +2.0956,
and the LLM beat every named realized baseline on two of three routes. In the
failed route, multisource ICM-BMA reached 99.8 final yield and 96.38 AUC, versus
94.5 and 94.42 for the online LLM. This is a post-hoc, single-trajectory stress
test: the strongest baseline is identified after observing each route, no
confidence interval is estimable, and the result does not replace calibration-
selected repeated confirmation. Its value is to show that the method survives
stronger comparators on the molecular and material examples while exposing a
specific classical method that remains better on Suzuki.

### Full-portfolio baseline audit and prediction-error gate

The three-route stress test was broadened in two directions. First, every one
of the eleven completed trajectories in the frozen v1 route panel was replayed
from its exact initial candidate IDs against target-only GP-UCB, RGPE, and
multitask GP under the same target pool and ten-reveal budget. The primary
classical comparator rule was fixed at analysis time: RGPE for a single source
and multisource RGPE for multiple sources. This is not prospective
preregistration because the online trajectories already existed.

Across the eleven routes, the online LLM gained +1.753 AUC over target GP
(route-bootstrap interval +0.415 to +3.315; 6 wins, 2 ties, 3 losses) and
+3.229 over the rule-fixed RGPE comparator (interval -1.844 to +10.141;
6 wins, 1 tie, 4 losses). Against the strongest realized baseline selected
post hoc within each route, however, the mean was -0.550 with only 3 wins,
1 tie, and 7 losses. The latter is an intentionally unfavorable oracle
diagnostic; it shows that the current LLM controller is not yet the strongest
optimizer on most routes.

Second, the inventory was expanded to all seventeen distinct online LLM routes
already present in the repository, including six earlier null or negative
development cases. In this complete retrospective inventory, the mean gain was
+0.762 versus target GP (interval -0.501 to +2.091; 8 wins, 4 ties, 5 losses)
and +0.987 versus rule-fixed RGPE (interval -2.901 to +5.960; 8 wins, 2 ties,
7 losses). Against the strongest realized baseline, the online LLM averaged
-1.821 AUC with 3 wins, 2 ties, and 12 losses. Reporting both the frozen
11-route panel and the complete 17-route inventory makes route selection
visible rather than allowing the stronger subset to stand in for all prior
experiments.

The new calibration-gate audit tests a narrower CARE claim: whether observed
LLM reliability can control decision authority. After three LLM-guided target
reveals, the gate computes the mean absolute error between the LLM's asserted
expected outcomes and the revealed values. Responses that explicitly defer to
GP are not scored as predictions. A threshold is selected for each held-out
route using only the other ten routes; if the error exceeds that threshold, or
the LLM explicitly falsifies or abandons the transfer hypothesis, target-only
GP continues from all observations accumulated so far. The GP is not restarted
and the result is not spliced from an independent trajectory.

This gate switched on eight of eleven held-out evaluations. Its mean AUC gain
over target GP was +2.036 (route-bootstrap interval +0.425 to +3.901; 5 wins,
4 ties, 2 losses), compared with +1.753 and 6/2/3 for the ungated online LLM.
Thus the negative-transfer rate fell from 27.3% to 18.2%. The gate changed the
mean by only +0.283 relative to the original LLM, with an interval crossing
zero, and it still averaged -0.267 against the strongest realized baseline.
The defensible conclusion is that prediction-error calibration improves risk
control in this retrospective portfolio, not that it establishes universal
superiority. Prospective repeated confirmation remains required.

The second audit applies one global threshold of five to every route, matching
the rule frozen in the executable protocol. It switches on ten of eleven
retrospective trajectories and changes the mean AUC delta versus target GP from
+1.753 to +2.135 (route-bootstrap interval +0.515 to +4.001; 6 wins, 3 ties,
2 losses). Relative to the ungated LLM, the mean change is +0.381 with an
interval from -0.371 to +1.432. Relative to the strongest realized baseline
selected post hoc, it remains -0.168 with 3 wins, 1 tie, and 7 losses. This
fixed-rule replay validates the intended runtime semantics and provides a
rationale for prospective testing; it is not itself prospective evidence. In
counterfactual execution it replaces 70 later LLM-guided rounds, equivalent to
140 nominal proposer/critic calls. These numbers describe calls the executable
Gate would avoid on the saved trajectories, not API savings already realized.

A stricter route-split audit then held six earlier trajectories out of all
controller selection. The imported round-three threshold-5 rule did not
generalize: it averaged -1.968 AUC versus target GP, compared with -1.056 for
the full online LLM, and retained two losses. The failure is concentrated in
`phonons -> perovskites`, where handing control back after three LLM rounds was
too late to recover the early search reward.

We next evaluated 115 candidate handoff policies on the 11 training routes
only. The declared lexicographic rule first minimizes negative routes, then
maximizes equal-route mean and worst-route AUC, and finally prefers the simpler
controller on ties. It selected a threshold-free one-round authority limit.
On the six route-disjoint trajectories this controller averaged +0.612 versus
target GP (95% interval -0.442 to +1.793; 2 wins, 3 ties, 1 loss), and +1.668
relative to the original full online LLM. The result identifies a plausible
mechanism: LLM semantic transfer is most useful as an early intervention,
whereas repeated stochastic policy control can erase that benefit. The
intervals cross zero and one loss remains, so the claim is improved observed
robustness, not general superiority.

The controller choice is also reasonably stable to individual development
routes. Leave-one-route-out re-selection chose `bounded_authority_r1` in 10 of
11 folds. Applying each fold's selected policy to its omitted route produced a
mean AUC delta of +1.141 with 2 wins, 8 ties, and 1 loss. This reduces concern
that one development route alone determines the handoff rule, but it does not
replace evaluation on a genuinely new prospective panel.

## Live bounded-authority pilot: participation did not change the action

The first live launch completed one successful Opus 4.8 trajectory on each of
the six route-disjoint tasks. Every proposer/critic response was structurally
valid, but all six selected the same target GP-UCB rank-one candidate as the
matched comparator. Consequently, the online LLM increment was exactly 0.0000
AUC on every route: 0 wins, 6 ties, 0 losses, with an override rate of 0/6. The
six calls consumed 99,243 tokens. The one-round authority limit then handed all
remaining optimization to GP-UCB, avoiding 108 nominal later proposer/critic
calls. The complete system, which additionally includes the stochastic LLM
initial design, averaged -0.0087 AUC versus the fixed initial-design system
(3 wins, 1 tie, 2 losses).

One pre-response 401 credential failure is retained in the attempt tree. Under
the frozen retry semantics it is terminal, so this launch cannot satisfy the
declared 180-trajectory completion rule. The six successful trajectories are
reported as an operational pilot, not as confirmatory inference. Their main
value is diagnostic: a model can participate, emit a hypothesis, and pass a
critic while having zero causal effect on the chosen experiments.

This result changes the paper's contribution hierarchy. The strongest current
LLM evidence is the earlier frozen semantic-skill compiler: source evidence and
the public target schema are converted into executable rule features, priors,
and an acquisition schedule, then evaluated without further LLM calls. On
paired held-out replay it improved best-so-far AUC over the strongest target-only
anchor by +1.384 for FreeSolv, +2.859 for Buchwald-Hartwig, and +5.930 for
Matbench experimental band gap. Same-seed schedule-only and no-prior ablations
showed that both the semantic representation and proposed direction contributed
on the reaction and materials tasks. ChemLex remained negative and was not
promoted to confirmation.

A stricter source-outcome assignment falsification has now been completed.
Across 100 target seeds and 19 matched outcome permutations per route, the true
reaction and materials assignments beat target-only, while the molecular route
did not. However, the empirical randomization p-values were 0.15, 0.10, and
0.20. Thus the audit does not establish that the correct source feature-outcome
binding is more informative than randomized assignments. A prospectively frozen
new task family remains mandatory; retrospective warm-start gains cannot carry
the causal source-outcome claim.

The hidden-target information boundary now has executable evidence. A frozen
audit reconstructed every archived request state in the 17-route online
portfolio, permuted all labels that were unrevealed at that state, and reran the
production source-prior, menu, diagnostics, and prompt builders. All 170 states
were invariant and every archived prompt matched its recorded digest. Six early
routes use a historical prompt/menu schema, so exact old-prompt reconstruction
is retained as a version-drift diagnostic rather than conflated with the
current-code non-interference endpoint. This closes a reproducibility gap but
does not close the larger efficacy or prospective-generalization gaps.

The defensible architecture is therefore:

`source evidence + target schema -> LLM-compiled skill -> strict compiler -> calibration -> frozen skill -> target optimizer`

The online proposer/critic is a secondary hypothesis-revision and abstention
module until it demonstrates nonzero beneficial decision impact on a development
panel and then repeats that result under a newly frozen disjoint protocol.

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

Current evidence update: the live pilot strengthens this concern. The online
controller itself made no action change, while the complete system's initial
design effect was mixed. The manuscript should not use the retrospective
bounded-authority result as its headline LLM contribution.

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

### 6. The LLM does not yet know when to trust itself

The complete 170-decision archive shows modest probability discrimination but
no route-level calibration advantage over the GP probability assigned to the
same candidate. The LLM is also more optimistic than the realized improvement
frequency. Raw model confidence is therefore not a defensible transfer gate.

Required response: fit probability calibration only on declared development
routes, combine the calibrated probability with action-change and evidence-
support features, and freeze the abstention rule before disjoint evaluation.

### 7. Positive efficacy is not yet cleanly attributable to the LLM mechanism

The prospective IRED additive skill improves AUC, but the fixed rank prior is
stronger and the later 99-assignment falsification gives p=0.07. Arbitrary
outcome bindings can occasionally induce useful rankings in a finite candidate
pool, so target-seed significance alone cannot establish semantic correctness.

Required response: preregister the source-assignment null together with the next
external-family efficacy protocol, use a mechanism-matched fixed prior and
learned transfer baselines, and keep the randomization test as a co-primary
attribution endpoint rather than adding it only after a positive result.

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
   insufficient for a broad cross-domain claim. Hydrophobic Core and Rhodopsin
   provide external negative-transfer and safety evidence, while IRED provides
   one preregistered positive finite-pool result. A more distant family and
   independent replication are still required for a broad claim.
4. **Prospective experimental validation.** Run at least one wet-lab or genuinely
   prospective closed-loop campaign with matched budget and starting state.
5. **Complete-system accounting.** Report the online increment, initial-design
   effect, and complete system separately. Do not let a strong online result hide
   a weak initial design.
6. **Matched transfer-BO baselines.** Compare the online LLM controller with
   calibration-selected RGPE and multitask GP under the same initial target
   observations, candidate pool, reveal budget, and held-out routes.
7. **Decision-impact gate before a new confirmation.** The first live pilot
   produced zero GP overrides, and one terminal credential failure prevents the
   original launch from satisfying its completion rule. Treat it as a failed
   operational pilot. Develop a challenger or compiled-skill policy only on the
   declared development routes; require nonzero beneficial action change there;
   then freeze a new disjoint repeated protocol before any evaluation calls.
8. **Prospectively calibrated abstention.** The retrospective reliability audit
   found no stable LLM calibration advantage and a +0.060 probability bias.
   Fit any calibration map on development routes only, then lock its parameters
   and abstention threshold before a new disjoint confirmation.
9. **Prospective mechanism attribution.** Freeze a source-outcome assignment
   randomization or another mechanism-matched negative control before the next
   external result. The current IRED audit is informative but post-hoc and has
   p=0.07.

Completed supporting control: the 17-route hidden-label non-interference audit
now provides production-path evidence that unrevealed target labels cannot
change the current source prior, candidate menu, diagnostics, or LLM prompt on
the 170 archived states. This control should accompany, not replace, the fresh
task and prospective confirmation requirements above.

Completed prospective safety control: the Rhodopsin protocol was committed
before dataset download, retained all declared seeds, and correctly fell back to
target-only GP-UCB before the official test was inspected. This satisfies one
external negative-transfer-prevention endpoint, but it does not satisfy the
positive-efficacy or wet-lab requirements above.

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
5. Extend the completed retrospective probability-calibration audit with a
   development-only calibration map and a newly frozen disjoint evaluation.

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

### Figure provenance rule

Every empirical figure must be generated by a checked-in script from versioned
CSV, JSON, or JSONL results. The source path, metric definition, unit of
analysis, and completed sample count must accompany the figure. Failed API
calls and interrupted trajectories remain in the audit record but are never
plotted as scientific outcomes. Conceptual framework diagrams are allowed, but
must be labeled as schematics rather than experimental evidence. In the current
deck, the route comparison is generated from
`2026-08-24-online-llm-predeclared-baseline-portfolio-v1/route_comparisons.csv`,
the paired stochasticity plot from
`2026-08-23-llm-trajectory-stability-audit/paired_route_stability.csv`, and the
fixed-gate plot from
`2026-08-24-online-llm-fixed-threshold5-gate-replay-v1/route_results.csv`.
Each route still has one completed online trajectory in these retrospective
figures; within-route uncertainty remains unestimated until the frozen repeated
protocol completes.

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

The current package is materially stronger but is not ready for a broad Nature
claim. The method now has commit-before-download external safety and positive-
efficacy confirmations, but the positive IRED skill is weaker than the fixed
prior and remains one-family finite-pool evidence. The IRED source-outcome audit
finds a stable +0.340 AUC advantage over the mean false binding, but its
assignment-level p=0.07 prevents a causal-mechanism claim. Repeated online LLM
confirmation, replication on a more distant external family, and a paired
prospective wet-lab campaign are still missing. The highest-value next move is
an independently preregistered external replication with its attribution null
frozen in advance, followed by one paired prospective laboratory campaign, not
another post-hoc development sweep.
