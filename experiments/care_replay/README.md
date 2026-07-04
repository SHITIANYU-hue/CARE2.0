# CARE Replay Experiment

This is a lightweight CARE 2.0 replay harness for the first experiment pass.

Current status:

- Uses synthetic finite candidate pools for Suzuki-style reaction optimization,
  ChemLex-style acid-amine optimization, and materials formulation optimization.
- Includes public real HTE adapters for Dreher-Doyle Buchwald-Hartwig and
  Perera Suzuki-Miyaura data from `rxn4chemistry/rxn_yields`.
- Includes the updated public ChemLex Acid-Amine wetlab adapter from Zenodo.
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
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_chemlex_acidamine --seeds 30 --rounds 10
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

Run the target-only incumbent rule ablation:

```bash
python3 experiments/care_replay/scripts/run_incumbent_ablation.py --dataset real_buchwald_hartwig --seeds 50 --rounds 10 --initial 5 --output-tag 50seed
python3 experiments/care_replay/scripts/run_incumbent_ablation.py --dataset real_moleculenet_lipophilicity --seeds 50 --rounds 10 --initial 5 --output-tag 50seed
```

Run the dependency-free surrogate baseline comparison:

```bash
python3 experiments/care_replay/scripts/run_surrogate_baselines.py --dataset real_buchwald_hartwig --seeds 50 --rounds 10 --initial 5 --output-tag 50seed
python3 experiments/care_replay/scripts/run_surrogate_baselines.py --dataset real_moleculenet_lipophilicity --seeds 50 --rounds 10 --initial 5 --output-tag 50seed
```

Run hybrid CARE transfer over a GP-UCB incumbent:

```bash
python3 experiments/care_replay/scripts/run_hybrid_surrogate_transfer.py --source-dataset real_moleculenet_freesolv --target-dataset real_moleculenet_lipophilicity --seeds 50 --rounds 10 --initial 5 --source-observations 192 --output-tag 50seed
python3 experiments/care_replay/scripts/run_hybrid_surrogate_transfer.py --source-dataset real_suzuki_miyaura --target-dataset real_buchwald_hartwig --seeds 50 --rounds 10 --initial 5 --source-observations 96 --output-tag 50seed
```

Run transfer-weighted GP-kernel skill optimization:

```bash
python3 experiments/care_replay/scripts/run_transfer_weighted_kernel.py --source-dataset real_suzuki_miyaura --target-dataset real_buchwald_hartwig --seeds 50 --rounds 10 --initial 5 --source-observations 96 --scales 0,0.5,1,1.5,2,4 --output-tag 50seed
python3 experiments/care_replay/scripts/run_transfer_weighted_kernel.py --source-dataset real_moleculenet_freesolv --target-dataset real_moleculenet_lipophilicity --seeds 50 --rounds 10 --initial 5 --source-observations 192 --scales 0,1.5,4 --output-tag 50seed
```

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
- `results/2026-06-30-real-chemlex/`: real ChemLex Acid-Amine wetlab replay
  using the updated Zenodo v3 record.
- `results/2026-06-30-bh-to-suzuki-transfer/`: first explicit BH-to-Suzuki
  transfer-card ablation.
- `results/2026-07-03-multidomain-transfer-feasibility/`: multi-domain
  transfer role-map expansion and server-side feasibility sweep across
  reaction HTE, molecular-property, ChemLex, and proxy materials directions.
- `results/2026-07-03-transfer-advantage-sweep/`: 50-seed server sweep showing
  a larger transfer advantage with shared-vocabulary molecular value priors and
  a stabilized Suzuki-to-Buchwald-Hartwig HTE transfer result.
- `results/2026-07-03-llm-transfer-followup/`: 10-seed real CommonStack LLM
  transfer follow-up on FreeSolv-to-Lipophilicity and
  Suzuki-to-Buchwald-Hartwig, including LLM proposer and LLM auditor modes.
- `results/2026-07-03-incumbent-rule-ablation/`: 50-seed target-only ablation
  explaining the current incumbent rule strength on Buchwald-Hartwig and
  Lipophilicity.
- `results/2026-07-03-surrogate-baselines/`: dependency-free GP-UCB, GP-EI,
  and kNN-UCB target-only baselines compared with the 50-seed transfer results.
- `results/2026-07-03-hybrid-surrogate-transfer/`: CARE transfer cards applied
  as bounded adjustments over a mixed-kernel GP-UCB incumbent.
- `results/2026-07-04-transfer-weighted-kernel/`: CARE transfer-card role
  confidence used to reweight the GP-UCB categorical kernel, giving a small
  positive acquisition-level transfer result over GP-UCB on real HTE and
  molecular-property replay.

The synthetic adapters let us test whether the same CARE gate and audit protocol
behaves consistently across task shapes. The real HTE adapters download public
Excel files into `data/raw/`. For the real adapters, no
fixed high-performing group prior is encoded; the replay policy only uses
revealed observations. The current public observation model uses smoothed means
over all revealed decision factors, and the gate only applies bounded
factor-evidence adjustments when public observations support them.

Rule provenance is tracked in `rule_provenance.md`. The current incumbent and
transfer rules are transparent engineering controls for the CARE 2.0 replay
harness; they are not CARE 1.0 paper-number reproductions or named community
baselines.

The current stronger baseline set includes random search, the public incumbent,
incumbent ablations, mixed-kernel GP-UCB, mixed-kernel GP-EI, and kNN-UCB. The
GP-style baselines are implemented without external numerical dependencies.
The current hybrid setting uses GP-UCB as the incumbent acquisition and tests
whether transfer cards can improve that stronger optimizer.
The transfer-weighted kernel setting moves one step deeper: the transfer card
changes the GP kernel field weights directly, so the reusable skill optimizes
the acquisition geometry instead of only adding a post-hoc candidate bonus.

The MoleculeNet ESOL adapter is not a reaction dataset. It frames measured
solubility as a finite-pool molecular property search task.

The real ChemLex Acid-Amine adapter uses the updated Zenodo wetlab table. The
synthetic ChemLex adapter remains useful as a smoke-test control, but real
ChemLex should be used for claims about acid-amine wetlab replay.

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
