# CARE Replay Experiment

This is a lightweight CARE 2.0 replay harness for the first experiment pass.

Current status:

- Uses synthetic finite candidate pools for Suzuki-style reaction optimization,
  ChemLex-style acid-amine optimization, and materials formulation optimization.
- Includes public real HTE adapters for Dreher-Doyle Buchwald-Hartwig and
  Perera Suzuki-Miyaura data from `rxn4chemistry/rxn_yields`.
- Implements the CARE 2.0 minimum loop:
  `TaskSpec -> SkillCard -> HypothesisEntry -> GateCertificate -> AuditLog -> Metrics`.
- Does not claim to reproduce CARE 1.0 paper numbers. It is a smoke test for the experiment interface while the original CARE 1.0 repo / public candidate tables are being confirmed.

Run:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset synthetic_suzuki_i --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset synthetic_chemlex_i --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset synthetic_materials_i --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_buchwald_hartwig --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_suzuki_miyaura --seeds 30 --rounds 10
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset all --seeds 30 --rounds 10
```

Outputs:

- `outputs/tables/<dataset_id>_metrics.csv`
- `outputs/runs/<dataset_id>_summary.json`
- `outputs/runs/<dataset_id>_audit_seed0.jsonl`
- `outputs/runs/<dataset_id>_knowledge_seed0.json`

The synthetic adapters let us test whether the same CARE gate and audit protocol
behaves consistently across task shapes. The real HTE adapters download public
Excel files into `data/raw/`, which is ignored by git. For the real adapters, no
fixed high-performing group prior is encoded; the replay policy only uses
revealed observations plus generic diversity/risk checks.
