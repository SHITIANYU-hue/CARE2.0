# CARE Replay Experiment

This is a lightweight CARE 2.0 replay harness for the first experiment pass.

Current status:

- Uses synthetic finite candidate pools for Suzuki-style reaction optimization,
  ChemLex-style acid-amine optimization, and materials formulation optimization.
- Includes public real HTE adapters for Dreher-Doyle Buchwald-Hartwig and
  Perera Suzuki-Miyaura data from `rxn4chemistry/rxn_yields`.
- Includes a MoleculeNet ESOL adapter for molecular property finite-pool replay.
- Includes MoleculeNet FreeSolv and Lipophilicity adapters for additional
  molecular-property generalization checks.
- Includes a Matbench experimental band-gap adapter for real materials
  composition-property replay.
- Supports optional LLM-in-the-loop modes through an OpenAI-compatible endpoint.
- Implements the CARE 2.0 minimum loop:
  `TaskSpec -> SkillCard -> HypothesisEntry -> GateCertificate -> AuditLog -> Metrics`.
- Does not claim to reproduce CARE 1.0 paper numbers. It is a smoke test for the experiment interface while the original CARE 1.0 repo / public candidate tables are being confirmed.
- Tracks upcoming dataset intake requirements in `datasets/intake.md`,
  including Kimi-Lex and Materials Project blockers.

Run:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset synthetic_suzuki_i --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset synthetic_chemlex_i --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset synthetic_materials_i --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_buchwald_hartwig --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_suzuki_miyaura --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_moleculenet_esol --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_moleculenet_freesolv --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_moleculenet_lipophilicity --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_matbench_expt_gap --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset all --seeds 30 --rounds 10
```

Run a real LLM-in-the-loop smoke test:

```bash
export CARE_LLM_API_KEY="..."
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset synthetic_materials_i --seeds 3 --rounds 6 --initial 5 --modes no_care_random,incumbent,llm_no_gate,llm_gate_v1 --llm-model openai/gpt-4o-mini --output-tag llm_commonstack
```

Run the nine-dataset LLM generalization sweep:

```bash
export CARE_LLM_API_KEY="..."
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset all --seeds 5 --rounds 6 --initial 5 --modes no_care_random,incumbent,llm_no_gate,llm_gate_v1 --llm-model openai/gpt-4o-mini --output-tag llm_commonstack_5seed
```

The default LLM base URL is `https://api.commonstack.ai/v1`. The API key is read
from `CARE_LLM_API_KEY` or `COMMONSTACK_API_KEY`; it is never stored in the
repository.

Outputs:

- `outputs/tables/<dataset_id>_metrics.csv`
- `outputs/runs/<dataset_id>_summary.json`
- `outputs/runs/<dataset_id>_audit_<mode>_seed<seed>.jsonl`
- `outputs/runs/<dataset_id>_knowledge_<mode>_seed<seed>.json`
- `outputs/runs/<dataset_id>_audit_seed0.jsonl` and
  `outputs/runs/<dataset_id>_knowledge_seed0.json` as short compatibility
  handles for `gate_v2` seed 0.

Tracked result snapshots:

- `results/2026-06-28-doc-guided-sweep/`: aggregate summaries and metric tables
  for a six-adapter sweep guided by the CARE 2.0 / AI4Science notes.
- `results/2026-06-29-materials-baselines/`: synthetic materials and real
  Matbench experimental band-gap replay with an explicit `no_care_random`
  baseline and no-gate ablation.
- `results/2026-06-29-commonstack-llm-replay/`: synthetic materials and real
  Matbench replay with real Commonstack LLM calls in `llm_no_gate` and
  `llm_gate_v1` modes.
- `results/2026-06-30-generalization-sweep/`: nine-dataset baseline and CARE2
  generalization sweep after adding FreeSolv and Lipophilicity.
- `results/2026-06-30-llm-commonstack-5seed/`: nine-dataset LLM replay using
  real CommonStack calls with `llm_no_gate` and `llm_gate_v1`.

The synthetic adapters let us test whether the same CARE gate and audit protocol
behaves consistently across task shapes. The real HTE adapters download public
Excel files into `data/raw/`. For the real adapters, no
fixed high-performing group prior is encoded; the replay policy only uses
revealed observations. The current public observation model uses smoothed means
over all revealed decision factors, and the gate only applies bounded
factor-evidence adjustments when public observations support them.

The MoleculeNet ESOL adapter is not a reaction dataset. It frames measured
solubility as a finite-pool molecular property search task.

The MoleculeNet FreeSolv adapter frames experimental hydration free energy as a
finite-pool molecular property search task. The hidden objective is a fixed-scale
hydration-affinity score where more negative experimental hydration free energy
is better.

The MoleculeNet Lipophilicity adapter frames experimental lipophilicity as a
finite-pool molecular property search task. The hidden objective is a fixed-scale
normalized lipophilicity score.

The Matbench experimental band-gap adapter uses composition-only public
features and revealed experimental band gaps. It is a real materials replay
dataset, but the first-pass observation model is intentionally lightweight; the
2026-06-29 result snapshot should be read as a diagnostic baseline rather than
as a positive CARE result.

The LLM modes ask the model to propose bounded factor-level adjustments from
revealed observations only. Audit logs store the raw model response, parsed JSON
policy, applied factor adjustments, model name, and usage metadata.
