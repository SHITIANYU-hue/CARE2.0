# Commonstack LLM Replay

This snapshot records the first replay pass with real LLM calls in the decision
loop. The LLM endpoint was Commonstack's OpenAI-compatible API, using
`openai/gpt-4o-mini`. The API key was supplied through `CARE_LLM_API_KEY` and is
not stored in the repository.

Run commands:

```bash
CARE_LLM_API_KEY=... python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset synthetic_materials_i --seeds 3 --rounds 6 --initial 5 --modes no_care_random,incumbent,llm_no_gate,llm_gate_v1 --llm-model openai/gpt-4o-mini --llm-max-tokens 700 --output-tag llm_commonstack
CARE_LLM_API_KEY=... python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_matbench_expt_gap --seeds 3 --rounds 6 --initial 5 --modes no_care_random,incumbent,llm_no_gate,llm_gate_v1 --llm-model openai/gpt-4o-mini --llm-max-tokens 700 --output-tag llm_commonstack
```

## Modes

| Mode | Meaning |
| --- | --- |
| `no_care_random` | No CARE components. Randomly samples unseen candidates. |
| `incumbent` | Public-observation greedy baseline. No skill or gate intervention. |
| `llm_no_gate` | LLM proposes bounded factor-level adjustments; challenger is accepted without gate checks. |
| `llm_gate_v1` | LLM proposes bounded factor-level adjustments; CARE gate v1 decides whether to accept the challenger. |

## Aggregate Results

| Dataset | Mode | Final Best | AUC | Regret | Top-10 Hit | LLM Calls | Interv. | Bad Interv. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_materials_i | no_care_random | 92.2338 | 91.1267 | 4.2401 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| synthetic_materials_i | incumbent | 94.4644 | 92.3017 | 2.0095 | 0.6667 | 0.0000 | 0.0000 | 0.0000 |
| synthetic_materials_i | llm_no_gate | 94.5891 | 92.3654 | 1.8848 | 0.6667 | 3.0000 | 1.3333 | 0.6667 |
| synthetic_materials_i | llm_gate_v1 | 94.8996 | 92.4172 | 1.5743 | 0.6667 | 3.0000 | 1.6667 | 0.6667 |
| real_matbench_expt_gap | no_care_random | 49.2917 | 48.4444 | 50.7083 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| real_matbench_expt_gap | incumbent | 52.2500 | 49.3403 | 47.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| real_matbench_expt_gap | llm_no_gate | 52.2500 | 49.3403 | 47.7500 | 0.0000 | 3.0000 | 1.3333 | 0.3333 |
| real_matbench_expt_gap | llm_gate_v1 | 52.2500 | 49.3403 | 47.7500 | 0.0000 | 3.0000 | 1.0000 | 0.0000 |

## Audit Evidence

The per-seed audit logs under `outputs/runs/*_llm_commonstack_audit_*.jsonl`
include the raw LLM response, parsed JSON policy, applied factor adjustments,
model name, and API usage metadata in `hypothesis_snapshot.llm_policy`.

Example LLM adjustment on synthetic materials:

```json
[
  {"field": "dopant", "value": "D4", "direction": "prefer", "weight": 0.05},
  {"field": "anneal_temperature", "value": "750.0", "direction": "prefer", "weight": 0.05},
  {"field": "dwell_time", "value": "10.0", "direction": "prefer", "weight": 0.05},
  {"field": "dwell_time", "value": "30.0", "direction": "penalize", "weight": -0.05}
]
```

Example LLM adjustment on real Matbench:

```json
[
  {"field": "element_count_bin", "value": "binary", "direction": "prefer", "weight": 0.05},
  {"field": "anion_family", "value": "halide", "direction": "prefer", "weight": 0.05},
  {"field": "max_element_fraction_bin", "value": "moderately_concentrated", "direction": "penalize", "weight": -0.05},
  {"field": "transition_metal_flag", "value": "no_transition_metal", "direction": "penalize", "weight": -0.05}
]
```

## Readout

On synthetic materials, LLM-in-the-loop slightly improves over the public greedy
incumbent in this small smoke run. On real Matbench, the LLM proposes meaningful
factor-level adjustments and the gate removes bad interventions, but final
best does not improve over the incumbent. This is useful evidence that the LLM
path is wired and auditable; it is not yet a strong materials result.
