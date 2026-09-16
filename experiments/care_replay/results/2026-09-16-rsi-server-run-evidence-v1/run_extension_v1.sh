#!/bin/bash
set -eu
cd /work/zeyuwang/care-rsi/repo
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/work/zeyuwang/care-rsi/venv/bin/python
for CONDITION in base adapter; do
  OUT=experiments/care_replay/results/2026-09-16-rsi-local-${CONDITION}-extension-v1
  "$PY" experiments/care_replay/scripts/run_rsi_live_feedback.py --config experiments/care_replay/configs/rsi_local_posttraining_${CONDITION}_extension_v1.json --output-dir "$OUT" --workers 1
  mkdir -p "$OUT/code_snapshot"
  cp experiments/care_replay/scripts/*.py "$OUT/code_snapshot/"
  cp /work/zeyuwang/care-rsi/extension_preflight.json "$OUT/preflight.json"
  "$PY" experiments/care_replay/scripts/audit_rsi_execution.py --result-dir "$OUT" > "$OUT/execution_audit_stdout.json"
done
printf 'PAIR_COMPLETE\n'
