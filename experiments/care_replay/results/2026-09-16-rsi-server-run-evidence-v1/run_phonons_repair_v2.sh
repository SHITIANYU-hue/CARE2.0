#!/bin/bash
set -eu
cd /work/zeyuwang/care-rsi/repo
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/work/zeyuwang/care-rsi/venv/bin/python
for CONDITION in base adapter; do
 OUT=experiments/care_replay/results/2026-09-16-rsi-local-${CONDITION}-phonons-repair-v2
 "$PY" experiments/care_replay/scripts/run_rsi_live_feedback.py --config experiments/care_replay/configs/rsi_local_posttraining_${CONDITION}_phonons_repair_v2.json --output-dir "$OUT" --workers 1
 mkdir -p "$OUT/code_snapshot"; cp experiments/care_replay/scripts/*.py "$OUT/code_snapshot/"
 "$PY" experiments/care_replay/scripts/audit_rsi_execution.py --result-dir "$OUT" > "$OUT/execution_audit_stdout.json"
done
printf 'REPAIR_PAIR_COMPLETE\n'
