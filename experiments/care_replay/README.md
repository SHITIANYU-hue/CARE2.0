# CARE Replay Experiment

This is a lightweight CARE 2.0 replay harness for the first experiment pass.

Current status:

- Uses a synthetic Suzuki-like finite candidate pool.
- Implements the CARE 2.0 minimum loop:
  `TaskSpec -> SkillCard -> HypothesisEntry -> GateCertificate -> AuditLog -> Metrics`.
- Does not claim to reproduce CARE 1.0 paper numbers. It is a smoke test for the experiment interface while the original CARE 1.0 repo / public candidate tables are being confirmed.

Run:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --seeds 30 --rounds 10
```

Outputs:

- `outputs/tables/synthetic_suzuki_metrics.csv`
- `outputs/runs/synthetic_suzuki_summary.json`
- `outputs/runs/synthetic_suzuki_audit_seed0.jsonl`
- `outputs/runs/synthetic_suzuki_knowledge_seed0.json`

