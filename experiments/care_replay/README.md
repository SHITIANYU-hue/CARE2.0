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
python3 experiments/care_replay/scripts/run_hybrid_surrogate_transfer.py --source-dataset real_moleculenet_freesolv --target-dataset real_moleculenet_lipophilicity --seeds 100 --rounds 5 --initial 5 --source-observations 192 --modes gp_ucb,hybrid_value_prior_gp_ucb_gate_v1,hybrid_value_prior_gp_ucb_no_gate --output-tag lowbudget5_100seed
```

Run transfer-weighted GP-kernel skill optimization:

```bash
python3 experiments/care_replay/scripts/run_transfer_weighted_kernel.py --source-dataset real_suzuki_miyaura --target-dataset real_buchwald_hartwig --seeds 50 --rounds 10 --initial 5 --source-observations 96 --scales 0,0.5,1,1.5,2,4 --output-tag 50seed
python3 experiments/care_replay/scripts/run_transfer_weighted_kernel.py --source-dataset real_moleculenet_freesolv --target-dataset real_moleculenet_lipophilicity --seeds 50 --rounds 10 --initial 5 --source-observations 192 --scales 0,1.5,4 --output-tag 50seed
```

Build a calibration/held-out summary for transfer-weighted kernel grids:

```bash
python3 experiments/care_replay/scripts/build_transfer_weighted_calibration_summary.py --metrics experiments/care_replay/outputs/tables/transfer_weighted_kernel_real_suzuki_miyaura_to_real_buchwald_hartwig_calibrated_grid_100seed_metrics.csv --out-dir experiments/care_replay/results/2026-07-05-calibrated-transfer-weighted-kernel --label suzuki_to_bh --calibration-seed-count 50
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
- `results/2026-07-04-hybrid-transfer-budget-sweep/`: 100-seed
  FreeSolv-to-Lipophilicity hybrid transfer sweep over GP-UCB, including 3/5/10
  round budgets and warm-start ablations. The main signal is early-discovery
  gain: higher AUC and top-10 hit rate under a strong target-only GP baseline.
- `results/2026-07-04-reaction-transfer-descriptors/`: reaction component
  descriptor infrastructure and matched-cap descriptor-value-prior rerun for
  BH/Suzuki transfer. It shows that descriptor-level transfer is active, but raw
  source descriptor value direction can cause negative transfer.
- `results/2026-07-05-gpt-transfer-descriptor-followup/`: GPT structured-output
  follow-up for descriptor-aware LLM transfer. GPT fixes the previous JSON
  reliability issue, but the current LLM proposer/auditor still produces almost
  no effective interventions under the constrained adjustment schema.
- `results/2026-07-05-target-calibrated-descriptor-transfer/`: target-calibrated
  descriptor prior experiment. Source descriptors only whitelist values, while
  target observations decide direction. This repairs Suzuki-to-BH raw descriptor
  negative transfer and beats incumbent in the strict setting, but does not yet
  beat the strongest role-level transfer baseline.
- `results/2026-07-05-molprop-value-prior-budget-sweep/`: 100-seed
  FreeSolv-to-Lipophilicity value-prior transfer sweep over 3/5/10-round
  budgets. This is currently the cleanest positive transfer case: the shared
  descriptor value-prior gate improves final best, AUC, and top-10 hit rate in
  all three budget settings.
- `results/2026-07-05-calibrated-transfer-weighted-kernel/`: 100-seed
  calibration/held-out follow-up for transfer-weighted GP kernels. On
  Suzuki-to-Buchwald-Hartwig, calibration selects `scale=1.5`, which still
  beats GP-UCB on held-out seeds by +0.3966 final best and +0.3162 AUC. The
  same GP-kernel idea does not hold up on FreeSolv-to-Lipophilicity, while the
  separate shared descriptor value-prior result remains positive under the
  same 50/50 held-out check.

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

Layered skill-transfer design is tracked in `skill_transfer_layers.md`. The
current transfer-weighted kernel experiment is treated as one concrete
model/acquisition binding inside a broader transferable skill artifact that also
needs representation mapping, mechanism hypotheses, gate/risk checks, and
workflow boundaries.

The current stronger baseline set includes random search, the public incumbent,
incumbent ablations, mixed-kernel GP-UCB, mixed-kernel GP-EI, and kNN-UCB. The
GP-style baselines are implemented without external numerical dependencies.
The current hybrid setting uses GP-UCB as the incumbent acquisition and tests
whether transfer cards can improve that stronger optimizer.
The latest budget sweep shows that shared-descriptor value-prior transfer over
GP-UCB mostly helps early discovery. In the 10-round 100-seed setting,
top-10 hit increases from 0.11 to 0.21 and AUC increases from 86.5433 to
86.7037, while final best is nearly tied. In 3- and 5-round low-budget
settings, the final-best and AUC deltas are larger.
The transfer-weighted kernel setting moves one step deeper: the transfer card
changes the GP kernel field weights directly, so the reusable skill optimizes
the acquisition geometry instead of only adding a post-hoc candidate bonus.
The target-calibrated descriptor setting adds a safety lesson for reaction
transfer: descriptor infrastructure is useful, but direct source value direction
is too risky. Source descriptor knowledge should narrow the search attention;
target observations should decide the sign and strength before the gate can
authorize an intervention.
The MoleculeNet value-prior budget sweep is the clearest current positive
transfer result because the source and target share descriptor vocabulary. In
3/5/10-round budgets, `transfer_value_prior_gate_v1` improves final best by
1.1662/0.9725/0.8550 and top-10 hit by 0.06/0.07/0.06 over incumbent.

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
