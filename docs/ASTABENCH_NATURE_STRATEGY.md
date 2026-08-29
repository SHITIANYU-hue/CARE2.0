# AstaBench and cognitive-map strategy for CARE 2.0

Updated 2026-08-29.

## Decision

CARE 2.0 should add AstaBench as an external, orthogonal evaluation rather than
reframe the paper around a new benchmark. The primary scientific contribution
remains evidence-bounded transfer between sequential experiments. AstaBench can
answer a different reviewer question: does the same agent show reproducible
scientific reasoning outside the internally constructed replay routes, under a
standard interface and a visible inference budget?

DiscoveryBench is the first target because it requires an agent to load data,
form a specific hypothesis, and explain an executable workflow. Literature QA
would test retrieval, not CARE's main mechanism. End-to-end discovery is useful
later but substantially more expensive and introduces software-engineering
confounds before the data-analysis mechanism is established.

## Implemented external-evaluation path

The solver in `experiments/astabench/care_astabench_solver.py` uses the original
benchmark tools and adds a frozen procedural map:

1. audit files, variables, units, missingness, and leakage;
2. sample two to four competing hypotheses;
3. map each hypothesis to an executable statistical test;
4. retain effect size, uncertainty, assumptions, and negative evidence;
5. falsify the leading path and reroute if an assumption fails;
6. submit the narrowest supported hypothesis and a reproducible workflow.

The map contains generic procedures only. It stores no DiscoveryBench answers,
gold workflows, or target-specific conclusions. Inspect logs preserve the model
messages, tool calls, usage, failures, and final answer.

Benchmark execution is intentionally read-only across tasks. Production CARE
may consolidate a completed, audited experiment into memory, but allowing one
validation or test item to update the skill map used by later items would make
the external score order-dependent and risk benchmark contamination. Candidate
improvements discovered during validation must therefore be reviewed, frozen as
a new protocol version, and evaluated in a fresh run; test outcomes never feed
back into the reported solver.

The frozen comparison is CARE versus official ReAct under the same model,
provided tools, task order, message limit, and token limit. Ablations remove the
rerouting step or the full procedural map. The primary endpoint is the paired
per-task official DiscoveryBench score. A cost-adjusted result is secondary;
the protocol does not permit declaring success from a cheaper but substantially
worse agent.

## What the cognitive-map paper contributes

Lin et al., *Neural sampling from cognitive maps enables goal-directed
imagination and planning*, provides a useful computational analogy, not direct
evidence for CARE. Its relevant principles are compositional state-action maps,
sampling several goal-directed paths, feasibility constraints, and rapid
rerouting around new barriers without relearning the entire map.

CARE adapts those principles conservatively:

| Cognitive-map concept | CARE implementation |
|---|---|
| State | task schema, current hypothesis, revealed evidence, uncertainty |
| Action | an executable analysis or transfer skill |
| Transition | tool result or target observation that updates support |
| Goal | a verified hypothesis or improved target experiment |
| Sampled paths | several candidate hypothesis-skill chains proposed before selection |
| Affordance constraint | available columns, allowed tools, budget, and leakage boundary |
| Barrier | failed assumption, prediction error, negative transfer, or missing support |
| Rerouting | preserve the failed path and move to another frozen candidate |

This is not an implementation of the paper's neural GCML, grid-cell code, or
Hebbian learning rule. CARE uses an LLM and explicit audit records. The paper
supports the design intuition that compositional maps and constrained path
sampling may generalize to unseen states; CARE must still establish that claim
empirically through AstaBench and prospective transfer tests.

## Current quality-cost result

The AstaBench-style internal audit uses the frozen 11-route first-trajectory
panel. It reports real model tokens and the best-so-far AUC difference against
the same-initial target-only GP. It deliberately does not convert historical
tokens to dollars because no provider price snapshot was frozen at execution.

This evidence is descriptive. There is one completed trajectory per route out
of the planned 30, and negative routes remain in the analysis. The separate
bounded-authority pilot shows zero online increment on six routes because the
LLM accepted GP rank one and the gate then fell back to target-only GP. That is
safety and efficiency evidence, not efficacy evidence.

## Nature-facing evidence package

The manuscript should organize evidence in four layers:

1. **Mechanism:** frozen skills, candidate paths, execution, falsification,
   gate decisions, and self-update boundaries are inspectable.
2. **Internal causal comparison:** CARE is compared with the same-initial
   target-only policy and matched transfer-BO baselines under equal target
   budgets.
3. **External generalization:** DiscoveryBench compares CARE with simple ReAct
   using official scoring and cost accounting. Positive and negative tasks are
   both retained.
4. **Prospective scientific value:** preregistered external-family transfer and,
   ultimately, a physical wet-lab campaign test whether useful skills survive
   outside finite-pool replay.

The strongest publishable position is not “CARE solves every domain.” It is:

> CARE makes an LLM's scientific reuse decisions executable, falsifiable, and
> auditable; it measures when prior experience helps, abstains when evidence is
> insufficient, and exposes the cost and failures of that process.

## Remaining blockers

1. Accept the AstaBench gated-data license and provide `HF_TOKEN` in the run
   environment. The current machine cannot access the official split without it.
2. Freeze one model ID and run validation for CARE, ReAct, and the two ablations.
3. Lock code and inspect the test split only after the validation decision.
4. Complete the existing 30-trajectory-per-route confirmation or reduce the
   claim; first trajectories cannot support stable route-level efficacy.
5. Add at least one prospective physical experiment. AstaBench cannot replace
   wet-lab evidence for a broad AI Scientist claim.

## Sources

- [AstaBench suite](https://allenai.org/asta/bench)
- [AstaBench technical report](https://allenai.org/papers/astabench)
- [AstaBench implementation](https://github.com/allenai/asta-bench)
- [Agent baseline implementations](https://github.com/allenai/agent-baselines)
- [Neural sampling from cognitive maps enables goal-directed imagination and planning](https://www.nature.com/articles/s42256-026-01254-4)
- [GCML reference implementation](https://github.com/LH-cbicr/GCML)
