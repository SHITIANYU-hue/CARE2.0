# Multi-domain transfer feasibility replay

Date: 2026-07-03

This snapshot records the first server-side CARE 2.0 transfer sweep that
explicitly spans reaction HTE, molecular-property, ChemLex, and proxy materials
tasks. The goal is not to claim that every cross-domain direction already works.
The goal is to show that reusable source-domain role evidence can improve a
real target replay when the source and target roles are aligned, while the same
harness also exposes negative transfer and gate conservatism.

## Experiment Design

The transfer harness uses source-domain observations to build role-level
transfer cards, then maps those source roles onto target roles before replaying
the target task. Target outcomes are still revealed only through the replay
loop; the transfer card only changes the candidate scoring policy.

The role-map expansion covers:

- Reaction HTE: Buchwald-Hartwig, Suzuki-Miyaura, and ChemLex Acid-Amine.
- Molecular properties: ESOL, FreeSolv, and Lipophilicity.
- Proxy checks: synthetic ChemLex to real ChemLex, synthetic Suzuki to real
  Suzuki, and synthetic materials to real Matbench experimental band gap.

The main modes are:

- `incumbent`: the current non-transfer replay policy.
- `no_care_random`: random candidate selection baseline.
- `transfer_no_gate`: transfer card applied without additional rejection.
- `transfer_gate_v1`: transfer card with the standard transfer gate.
- `transfer_strict_gate_v1`: stricter gate intended to reduce harmful
  interventions.
- `llm_audit_transfer_gate_v1`: transfer gate with real LLM audit calls.

## Main Positive Result

The strongest result is Suzuki-Miyaura to Buchwald-Hartwig over 30 seeds. This
is the cleanest current evidence that CARE 2.0 transfer can beat both the
incumbent replay policy and the random baseline in a real HTE-to-HTE setting.

| Mode | Final best | Delta vs incumbent | Best-so-far AUC | AUC delta | Bad interventions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_care_random` | 83.7345 | -3.1473 | 79.2249 | -1.6026 | 0.0000 |
| `incumbent` | 86.8818 | 0.0000 | 80.8275 | 0.0000 | 0.0000 |
| `transfer_gate_v1` | 89.8725 | +2.9907 | 81.6410 | +0.8135 | 0.9333 |
| `transfer_strict_gate_v1` | 86.7952 | -0.0866 | 80.8930 | +0.0655 | 0.1000 |

Interpretation: the ordinary transfer gate preserves the full benefit in this
direction, while the strict gate sharply reduces harmful interventions but loses
most of the final-best gain. This is a useful design signal: the next gate
should keep the useful Suzuki-to-BH transfer pressure while detecting only the
highest-risk interventions.

## Additional Support

These results are smaller or more conditional than the 30-seed Suzuki-to-BH
result, but they show that the same transfer interface is not specific to one
dataset pair.

| Source -> Target | Seeds | Best transfer mode | Final delta | AUC delta | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| Suzuki-Miyaura -> ChemLex | 10 | `transfer_strict_gate_v1` | +4.4380 | +0.4367 | Positive transfer, but random is unusually strong on this target. |
| ChemLex -> Buchwald-Hartwig | 10 | `transfer_no_gate` | +1.4691 | -0.2507 | Final-best gain without AUC gain. |
| FreeSolv -> Lipophilicity | 10 | `transfer_strict_gate_v1` | +0.7000 | +0.0462 | Small molecular-property transfer signal. |
| ESOL -> Lipophilicity | 20 | `transfer_gate_v1` | +0.4062 | +0.1193 | Reproduces a positive molecular-property signal at 20 seeds. |
| FreeSolv -> Lipophilicity | 20 | `transfer_strict_gate_v1` | +0.3499 | +0.0556 | Safer but smaller molecular-property gain. |

## Boundary Cases

The sweep also gives a clear negative-control direction. Buchwald-Hartwig to
Suzuki-Miyaura does not transfer well under the current role mapping.

| Mode | Final delta | AUC delta | Bad interventions |
| --- | ---: | ---: | ---: |
| `transfer_gate_v1` | -1.3260 | -0.8504 | 1.6000 |
| `transfer_strict_gate_v1` | -1.8840 | -0.7159 | 1.0000 |

This matters because it suggests the method is not merely injecting generic
optimism. Directionality and role alignment matter, which is exactly where a
CARE 2.0 transfer system should focus.

## LLM Audit Check

The LLM audit path was run with a real CommonStack-compatible chat endpoint and
parsed successfully, with zero parse errors in this sweep.

| Mode | Final delta | AUC delta | Bad interventions | Rejected good challengers | LLM calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transfer_gate_v1` | +3.4720 | +0.9290 | 1.1000 | 0.0000 | 0.0000 |
| `llm_audit_transfer_gate_v1` | 0.0000 | 0.0000 | 0.1000 | 1.4000 | 2.8000 |

Interpretation: real LLM audit is wired into the replay harness, but the current
prompt/policy is too conservative. It acts more like a safety filter than a
benefit-preserving transfer gate. The next experiment should calibrate the LLM
audit so it rejects obvious negative transfer while keeping positive
Suzuki-to-BH challengers.

## Gate Tuning Check

A stricter `min_positive_roles=1` gate was tested on Suzuki-to-BH over 30 seeds.

| Mode | Final delta | AUC delta | Bad interventions | Interventions |
| --- | ---: | ---: | ---: | ---: |
| `transfer_gate_v1` | +2.9907 | +0.8135 | 0.9333 | 2.4333 |
| `transfer_strict_gate_v1` | +0.0083 | +0.2646 | 0.1333 | 0.3667 |

This confirms the current tradeoff: stricter gating lowers risk, but it is not
yet calibrated enough to keep the main transfer gain.

## Files

- `all_transfer_summary.csv`: all aggregate rows from the multi-domain,
  molecular, targeted Suzuki-to-BH, LLM audit, and strict-gate runs.
- `pair_best_transfer_summary.csv`: best transfer mode per source-target pair.
- `targeted_suzuki_to_bh_summary.csv`: focused Suzuki-to-BH 30-seed, LLM audit,
  and strict-gate summaries.
- `*_metrics.csv`: copied per-run metric tables for reproducibility.

Canonical summary JSON files are also stored under
`../../outputs/runs/`, and canonical metric CSVs are stored under
`../../outputs/tables/`.
