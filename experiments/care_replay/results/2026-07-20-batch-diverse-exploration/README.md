# Batch-diverse exploration ablation

Date: 2026-07-20

This run closes the remaining batch-diversity item in the July 20 exploration
proposal. It is an isolated target-only ablation, not a new transfer claim.

## Coverage of the exploration proposal

| Proposal item | Current implementation | Evidence |
| --- | --- | --- |
| Dynamic exploration weight | Round-dependent GP-UCB beta schedule | `run_transfer_weighted_kernel.py` and the July 19 live report |
| Novelty in candidate ranking | Public-factor novelty bonus | `run_exploration_batch.py` |
| LLM explore/exploit/avoid decision | Structured intent, evidence, confidence, and counter-hypothesis | `run_synthetic_suzuki.py` and July 19 live traces |
| Probabilistic exploration gate | Seeded, bounded authorization with an exploration quota | `run_synthetic_suzuki.py` |
| Alternative mechanism generation | LLM-generated kernel patches and failure conditions | `run_llm_kernel_skill_evolution.py` and July 19 live traces |
| Batch diversity | Same-evidence batch selection plus a minimum public-factor distance | `run_exploration_batch.py` and this result directory |

All six items now have an executable implementation. The new runs below cover
the previously missing batch item. They do not make new LLM calls.

## Protocol

Four public real datasets were run on the GPU server.

| Dataset | Candidates | Public decision fields |
| --- | ---: | ---: |
| Buchwald-Hartwig HTE | 3,955 | 4 |
| ChemLex Acid-Amine wetlab | 11,669 | 4 |
| MoleculeNet FreeSolv | 642 | 7 |
| Matbench experimental band gap | 4,604 | 7 |

The frozen settings were 30 seeds (`50-79`), 8 initial observations, a total
reveal budget of 6, and batches of 2. A candidate's target value is hidden
until the entire batch has been selected. All candidates in a batch therefore
use exactly the same pre-batch evidence.

The public distance is the fraction of decision fields with different values.
`target_diverse_batch` adds novelty with weight `0.04` and requires a minimum
within-batch distance of `0.34`. `target_diverse_batch_gate` applies the same
proposal but falls back to the incumbent when public acquisition loss exceeds
`0.02`. A Matbench-only sensitivity run tightens that bound to `0.005`.

The comparator is `incumbent_batch`, using the same initial samples, seeds,
batch size, and reveal budget. Reported intervals are paired 95% bootstrap
intervals over 20,000 resamples.

## Results

### Unconditional diversity

| Dataset | Delta final best | Delta AUC (95% CI) | Delta novelty | Delta batch distance |
| --- | ---: | ---: | ---: | ---: |
| Buchwald-Hartwig | -0.845 | -0.888 [-3.050, 1.095] | +0.060 | +0.161 |
| ChemLex | -0.329 | -0.475 [-1.712, 0.431] | +0.037 | +0.053 |
| FreeSolv | +0.667 | -0.678 [-4.693, 3.386] | +0.192 | +0.314 |
| Matbench band gap | -1.675 | -1.903 [-3.631, -0.368] | +0.259 | +0.417 |

The mechanism does what it is supposed to do operationally: novelty and
within-batch distance increase on every dataset. It does not produce a general
utility gain. The Matbench AUC loss is the clearest negative result.

### Acquisition-loss gate (`0.02`)

| Dataset | Delta final best | Delta AUC (95% CI) | Delta novelty | Authorization rate |
| --- | ---: | ---: | ---: | ---: |
| Buchwald-Hartwig | -0.590 | -0.683 [-2.886, 1.342] | +0.056 | 23.3% |
| ChemLex | +0.397 | -0.256 [-1.631, 0.878] | +0.036 | 18.9% |
| FreeSolv | +1.826 | +0.394 [-3.395, 4.339] | +0.188 | 58.9% |
| Matbench band gap | -1.750 | -1.965 [-3.613, -0.517] | +0.210 | 68.9% |

The gate improves the FreeSolv mean while retaining exploration, but its
interval still crosses zero. It does not protect Matbench.

For Matbench, tightening the acquisition-loss bound to `0.005` reduces the
authorization rate to 17.2% and the novelty delta to `+0.040`. AUC remains
negative at `-1.257` with a paired interval of `[-2.704, -0.218]`. This rules
out the simple explanation that the original gate was only too permissive.

## Interpretation

Batch diversity is now implemented and verified, but it should not be treated
as a headline optimization result. On FreeSolv there is a useful positive
direction; on reaction tasks the differences are mostly tied; on Matbench the
current exploration policy is harmful.

The Matbench result points to a representation and calibration problem. A
small loss under the current public incumbent score is not a reliable measure
of low target risk. The next materials experiment should replace the binned
Hamming geometry with composition descriptors or a pretrained materials
representation, then calibrate exploration authorization from target-only
prequential error. Merely tightening the same scalar threshold is not enough.

The new `llm_explore_batch_gate` path is implemented but was not run here. A
fresh CommonStack probe returned HTTP 429 with zero available balance. Existing
July 19 traces remain the current live-LLM evidence; no saved response was
relabelled as a new call.

## Reproducibility

`raw/` contains 558 files: summaries, per-seed metric tables, and complete
per-seed JSONL audit logs for the two four-dataset runs and the Matbench
sensitivity run. Each audit event records the public observation count,
incumbent, challenger, selected candidate, novelty, within-batch distance,
gate certificate, revealed value, and the evidence boundary.

The implementation is covered by 17 tests, including distance calculation,
diversity filtering, novelty scoring, low-cost authorization, high-cost
rejection, and audit-safe incumbent fallback.
