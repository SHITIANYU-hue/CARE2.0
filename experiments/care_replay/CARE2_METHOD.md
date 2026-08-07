# CARE 2.0 canonical method and code map

This note fixes one implementation as the CARE 2.0 method described in the
slides. Other scripts in this directory remain useful experiments, but they are
not alternate definitions of the main method.

## What the method does

CARE 2.0 asks whether a fixed history of measured source outcomes can improve a
new finite-pool search task. The source history is frozen before target replay.
The system maps source variables to public target descriptors, compiles several
executable priors, and offers them as experts beside a target-only acquisition
policy. Revealed target observations can calibrate those experts before each
new selection. Hidden target outcomes are never used to compile the skill or
score an unrevealed candidate.

The deployed route is chosen on calibration seeds. Transfer is accepted only
when it clears every frozen gain, stability, non-loss, and confidence test
against both the matched target-only LLM policy and the strongest target-only
BO policy. Otherwise CARE executes the matched target-only policy exactly.

## The one public execution path

Use `scripts/run_care2.py`:

```bash
# One source-target pair
python3 experiments/care_replay/scripts/run_care2.py pair [pair arguments]

# The predeclared benchmark suite
python3 experiments/care_replay/scripts/run_care2.py suite \
  --config experiments/care_replay/configs/source_outcome_benchmark.json \
  --strategy full_source_outcome \
  --calibration-seed-start 40000 \
  --heldout-seed-start 41000 \
  --pair-workers 12 \
  --parallel-pairs 7 \
  --output-tag frozen_source_outcome_v1
```

`pair` delegates to `run_calibrated_source_outcome_transfer.py`, which is the
canonical single-pair algorithm. `suite` only repeats that algorithm over a
frozen list of pairs; it does not implement a second transfer method.

## What a TransferSkill contains

`scripts/transfer_skill.py` defines the versioned `TransferSkill` artifact. It
contains:

- a source-target hypothesis and its failure condition;
- the fixed source history summary and source-target role map;
- the LLM-generated, normalized kernel patches and their rationales;
- executable operators for initial design, neighbor/additive/interaction
  priors, kernel geometry, online routing, and exact fallback;
- every execution parameter and every confirmation-gate threshold;
- the disjoint calibration and held-out seed ranges used for confirmation;
- hashes of both frozen LLM records;
- an evidence boundary that forbids hidden target outcomes.

The artifact is executable rather than descriptive. The canonical runner reads
its router settings and confirmation thresholds from the compiled skill. A
SHA-256 fingerprint changes when any patch, threshold, role map, or source
evidence summary changes.

## Slide-to-code map

| Method block | Code | Concrete output |
| --- | --- | --- |
| Historical experiments | `run_transfer_ablation.source_observations` | fixed source candidate history |
| Evidence extraction | `run_transfer_ablation.compile_transfer_card` | role effects and value priors |
| Hypothesis generation | `cross_task_router.propose_route` plus frozen LLM record | route proposal and patch rationales |
| Skill compilation | `transfer_skill.compile_transfer_skill` | versioned executable `TransferSkill` |
| Transfer execution | `run_llm_transfer_router.run_router_policy` | source-informed acquisition scores |
| Online safety check | `run_llm_transfer_router.router_gate_decision` | use router candidate or target anchor each round |
| Confirmation gate | `run_calibrated_source_outcome_transfer.select_route` | transfer or exact target-only fallback |
| Target experiment | `run_llm_transfer_router.run_seed` | finite-pool prequential replay |
| Statistical evaluation | selector delta/fold/CI helpers | paired final, AUC, non-loss, fold and CI statistics |
| Audit and memory artifact | `transfer_skill.build_canonical_trace` | source-to-gate-to-outcome JSONL trace |

## Where transfer is visible

Transfer is not the LLM merely describing two tasks. It occurs when measured
source outcomes change one or more of these target decisions:

1. which candidates enter the initial target design;
2. the neighbor prior over unrevealed target candidates;
3. additive source value effects mapped to target fields;
4. pairwise source interaction residuals mapped to target fields;
5. categorical kernel geometry;
6. the final acquisition ranking after target-only and transfer experts are
   combined.

When transfer is deployed, every held-out acquisition event records the target
anchor candidate, transfer candidate, final selected candidate, transfer mass,
active experts, revealed outcome, and gate reason. An exact-fallback trace keeps
the same fields but leaves transfer-only values empty.

## Outputs from every canonical pair run

- `*_metrics.csv`: all target-only, raw-transfer, and deployed-selector rows;
- `*_summary.json`: protocol, statistics, selection, and embedded skill;
- `*_transfer_skill.json`: the standalone frozen skill;
- `*_canonical_trace.jsonl`: compact end-to-end method trace;
- `*_audit_<mode>_seed<seed>.jsonl`: full low-level diagnostics.

## Contribution controls

The canonical pair run already produces four matched comparisons under the
same target seeds and reveal budget:

| Question | Comparison |
| --- | --- |
| Does any historical transfer help? | deployed CARE vs strongest target-only BO |
| Does it beat an LLM that has no source evidence? | deployed CARE vs matched target-only LLM |
| What happens without the confirmation gate? | raw `llm_transfer_router` vs deployed `care_source_outcome_router` |
| Does the gate prevent a loss? | rejected pairs reproduce the matched target-only LLM exactly |

The random-rule controls live in `run_random_rule_control.py`. Source-evidence
causality is summarized by `build_source_evidence_ablation.py`. A single
same-budget similarity-only arm and an online-gate-off arm are not yet part of
the frozen confirmation suite; they must be reported as planned ablations, not
as completed evidence.

## Claim boundary

The current method supports evaluated source-target paths in reactions,
molecular properties, and materials. It does not by itself establish arbitrary
far-domain transfer. A positive result means that one frozen skill passed the
specified confirmation protocol on held-out seeds for that declared pair.
