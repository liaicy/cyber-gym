#!/bin/bash
# Train 8 LoRA DPO adapters for reachable-slice k-fold (run on a compute node
# after salloc). Uses the FIXED dpo_lora.py (new _fit_messages window).
# 4 folds x {strict, cross}. Tiny data (18-34 pairs) -> each run is a few min.
set -u
cd ~/WORK/exam-society
mkdir -p runs/kfold
for i in 0 1 2 3; do
  for name in strict cross; do
    DATA="kfold_2026_10_12/fold${i}/${name}_train.jsonl"
    OUT="runs/kfold/${name}_fold${i}"
    echo "=== TRAIN fold${i} ${name}  data=${DATA} ==="
    venv/bin/python scripts/dpo_lora.py \
      --model models/Qwen3-8B \
      --data "${DATA}" \
      --out "${OUT}" \
      --epochs 2 --lr 5e-6 --beta 0.1 --rank 32 --seed 0 2>&1 | tail -4
    echo "--- done ${name}_fold${i} (adapter at ${OUT}/final) ---"
  done
done
echo "KFOLD_TRAIN_DONE"
