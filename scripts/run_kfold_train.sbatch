#!/bin/bash
#SBATCH -J es_kfold
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/kfold_%j.out
#SBATCH -e logs/kfold_%j.err
#SBATCH --no-requeue
#SBATCH -t 02:00:00
cd $HOME/WORK/exam-society
. venv/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/kfold
nvidia-smi -L
# 8 LoRA DPO adapters: 4 folds x {strict, cross}. Fixed dpo_lora.py window.
for i in 0 1 2 3; do
  for name in strict cross; do
    echo "===== TRAIN fold${i} ${name} ($(date +%H:%M:%S)) ====="
    python scripts/dpo_lora.py --model models/Qwen3-8B \
      --data kfold_2026_10_12/fold${i}/${name}_train.jsonl \
      --out runs/kfold/${name}_fold${i} \
      --epochs 2 --lr 5e-6 --beta 0.1 --rank 32 --seed 0
    echo "----- done ${name}_fold${i} -> runs/kfold/${name}_fold${i}/final -----"
  done
done
echo "KFOLD_TRAIN_DONE"
