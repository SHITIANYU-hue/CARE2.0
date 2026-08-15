# Claude Opus 5 online scientist development suite

Date: 2026-08-15

This archive tests whether a strong LLM can remain involved throughout finite-
pool experiment selection without sacrificing the gains of the calibrated CARE
controller. Claude Opus 5 proposes the initial design and participates in every
online reveal. The target outcomes are hidden until the selected experiment is
revealed.

## Controller

For each target, the controller follows this loop:

1. Build an outcome-blind initial design from source outcomes and public target
   conditions. Single-group C-N tasks use direct Opus selection. The multi-group
   Suzuki task compiles the Opus semantic anchor with a source anchor and a
   geometry probe.
2. Fit target-only GP-UCB on the target observations revealed so far.
3. Compute a source prior from feature-outcome relationships in the source
   tasks, then rank-fuse the GP and source views.
4. Give Opus the evidence, compact candidate menu, and previous observations.
   Opus returns a structured hypothesis, candidate choice, expected direction,
   uncertainty, and falsification condition.
5. Apply the calibrated eligibility check and reveal exactly one target outcome.
6. Save the complete prompt, raw response, normalized decision, executed
   candidate, revealed value, and token usage before the next round.

The first online reveal is a frozen safety anchor: the consensus rank-one
candidate is executed while the Opus hypothesis and recommendation are still
recorded. From the second reveal onward, Opus chooses among the calibrated
top-consensus candidates. Thus the LLM participates in all rounds but does not
have unconstrained access to the full target pool.

## Comparators

- `llm_initial_target_gp`: target-only GP-UCB starting from the exact same
  executed LLM initial design. This isolates the value of the online LLM
  controller after initialization.
- `fixed_v2_initial_design`: the frozen CARE v2 controller at the same total
  target-observation budget. This compares the complete LLM scientist against
  the established deterministic controller.

No comparator receives additional target observations. Source outcomes are
available as transfer evidence, but target outcomes are revealed only after a
candidate is selected.

## Results

| Target | Rounds | LLM participation | AUC | Delta vs same-initial GP | Delta vs fixed v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C-N morpholine / AlPhos | 10 | 100% | 66.6072 | -1.2536 | +0.4816 |
| C-N morpholine / tBuBrettPhos | 10 | 100% | 9.6218 | +7.0727 | +1.8005 |
| C-N morpholine / tBuBrettPhos preliminary | 8 | 100% | 11.9909 | 0.0000 | 0.0000 |
| C-N phenethylamine / AlPhos | 10 | 100% | 98.5826 | +0.7039 | 0.0000 |
| Suzuki MINLP2 | 10 | 100% | 100.0000 | 0.0000 | +1.9388 |

Aggregate observations:

- 48/48 online rounds contain an Opus decision and a target reveal.
- Selected runs contain zero API or structured-output errors.
- Against fixed CARE v2, mean AUC delta is `+0.8442`: three wins, two ties,
  zero losses.
- Against identical-initial-design GP-UCB, mean AUC delta is `+1.3046`: two
  wins, two ties, one loss. Mean final-best delta is `+1.9245` with no losses.
- Initial design used 28,148 tokens; online decisions used 287,394 tokens.

The one-sided exact sign-test p-value against fixed v2 is `0.125` after ties are
excluded. The result is therefore useful development evidence, but it is not a
confirmatory statistical claim. Four targets are related C-N campaigns and one
is Suzuki; independent frozen campaigns and more distant domains are still
required to establish generalization.

## Archive layout

Each task directory contains:

- `initial_record.json`: outcome-blind initial prompt, raw Opus response,
  normalized initial design, usage, and configuration fingerprint;
- `llm_trace.jsonl`: one structured audit event per online reveal;
- `summary.json`: metrics, comparator deltas, budget, controller policy, and
  trace location.

`aggregate/suite_summary.json` is the machine-readable suite report and
`aggregate/suite_metrics.csv` is its tabular form. `run_manifest.json` records
the frozen protocol. `SHA256SUMS` protects the archive contents from silent
changes.

The trace contains model-visible evidence and concise model-generated
rationales. It is an audit artifact, not hidden chain-of-thought.

## Reproduction

Use `configs/baumgartner_multisource_warmstart_v2.json` and provide an API key
through an environment variable. A task is run in two explicit stages:

```bash
python3 scripts/run_online_llm_scientist.py generate-initial \
  --config configs/baumgartner_multisource_warmstart_v2.json \
  --target-task TARGET_TASK \
  --source-tasks SOURCE_TASK_1 SOURCE_TASK_2 \
  --output /tmp/initial_record.json \
  --llm-api-mode anthropic \
  --llm-base-url https://api.anthropic.com \
  --llm-model claude-opus-5 \
  --llm-api-key-env ANTHROPIC_API_KEY

python3 scripts/run_online_llm_scientist.py run \
  --config configs/baumgartner_multisource_warmstart_v2.json \
  --initial-record /tmp/initial_record.json \
  --output-dir /tmp/online_run \
  --initial-design-mode auto \
  --rounds 10 \
  --force-first-consensus \
  --fail-on-llm-error \
  --llm-api-mode anthropic \
  --llm-base-url https://api.anthropic.com \
  --llm-model claude-opus-5 \
  --llm-api-key-env ANTHROPIC_API_KEY
```

Aggregate task directories with `scripts/build_online_llm_scientist_report.py`.
