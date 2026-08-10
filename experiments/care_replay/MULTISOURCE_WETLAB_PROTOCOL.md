# CARE 2.0 multi-source wet-lab protocol

## Question

Can completed reaction campaigns A, B, and C help optimize a genuinely new
campaign E under a fixed experimental budget, without using E's unrevealed
outcomes to choose the transfer method?

This protocol is separate from the older pairwise replay selector. It tests a
deployable new-task boundary: source data are complete, the target search space
is public, and target outcomes arrive only through sequential `ask()` / `tell()`
updates.

## Real data and search space

The benchmark uses the four Reizman Suzuki reaction tasks distributed with the
public code for Taylor et al., *Accelerated Chemical Reaction Optimization Using
Multi-Task Learning* (ACS Central Science, 2023). Each task contains 96 measured
conditions with one categorical variable and three continuous variables:

| Variable | Type | Declared range |
| --- | --- | --- |
| catalyst | categorical | 8 catalysts |
| residence time | continuous | 60-600 s |
| temperature | continuous | 30-110 C |
| catalyst loading | continuous | 0.498-2.515 mol% |

Cases 1-3 are development tasks. Case 4 is the untouched confirmation task.
The source files, pinned upstream commit, license, and hashes are kept under
`data/raw/reizman_suzuki/`.

## Frozen task split

- Development: cases 1, 2, and 3, evaluated by leave-one-task-out replay.
- Confirmation: case 4, loaded only after route selection is written.
- Development: 30 seeds per held-out task, 90 paired folds in total.
- Confirmation: 100 new seeds.
- Every policy receives the same three initial target observations and twelve
  additional reveals.
- Best-so-far AUC is the route-selection metric because it measures how quickly
  useful conditions are found under the same wet-lab budget.

The complete frozen configuration is
`configs/multisource_new_task_benchmark_v1.json`. The runner rejects overlapping
development/evaluation task lists and a mismatched selection-record hash.

## Compared policies

| Policy | Source information used |
| --- | --- |
| target GP-UCB | none; matched target-only control |
| multi-source RGPE | one source GP per completed task, weighted by online rank evidence |
| multi-source ICM-BMA | one source-conditioned multitask GP per task, combined by target marginal likelihood |
| multi-source skill prior | median source rank blended into target GP-UCB with a development-selected decay |
| development-selected route | the only deployable route; exact target-only fallback unless a source policy clears the frozen gate |

The admission rule is fixed before case 4: the paired 95% confidence-interval
lower bound for development AUC must be greater than zero. If no source policy
passes, the deployed route is exactly target GP-UCB.

## Persistent skill artifact

`skill_banks/care2-wetlab-transfer/SKILL.md` is the first persistent CARE 2.0
skill artifact. It is not a renamed pair-local `TransferSkill`. It consolidates
multi-task evidence into:

- a concise procedure that an agent can read;
- explicit applicability and abstention conditions;
- evidence IDs and source-task hashes;
- the frozen development selection decision;
- a manifest fingerprint and detailed evidence kept outside the prompt.

This version is produced by a deterministic compiler. LLM-based trace mining,
deduplication, and skill evolution remain future work and require a separate
task-disjoint validation step before any generated instruction is promoted.

## Wet-lab boundary

`scripts/wetlab_protocol.py` separates policy state from the offline replay
oracle. The policy can see public conditions and observations already returned
by the lab. It cannot request another condition while one is pending, and it
never stores unrevealed yields. In an actual campaign, `ReplayOracle` is replaced
by the laboratory result service without changing the policy interface.

## Confirmation result

All six source-informed candidates failed the development admission gate, so
the frozen deployment route was target GP-UCB. On case 4, the selected route
therefore matched target GP-UCB exactly across 100 seeds and introduced no
negative transfer.

The least aggressive skill prior had a case-4 AUC delta of `+0.337`, but its
95% CI was `[-0.225, +0.899]`; this is not reliable evidence of positive
transfer. RGPE improved top-10 hit rate by `+0.10` with CI
`[+0.029, +0.171]`, while reducing final best yield by `-2.707` and AUC by
`-0.905`. That trade-off is retained as a negative result rather than selected
after seeing the target.

The result supports calibrated abstention and a real multi-source execution
path. It does not yet support a claim that accumulated source knowledge improves
new wet-lab tasks.

## Reproduction

```bash
python3 experiments/care_replay/scripts/calibrate_multisource_skill.py \
  --config experiments/care_replay/configs/multisource_new_task_benchmark_v1.json \
  --output-dir /tmp/care2-multisource-development

python3 experiments/care_replay/scripts/run_multisource_transfer_baselines.py \
  --config experiments/care_replay/configs/multisource_new_task_benchmark_v1.json \
  --experiment-id reizman_cases_123_to_case4 \
  --selection-record /tmp/care2-multisource-development/selection_record.json \
  --output-dir /tmp/care2-multisource-confirmation
```
