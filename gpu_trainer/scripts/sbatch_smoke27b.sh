#!/bin/bash
#SBATCH -J es_smoke27b
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/smoke27b_%j.out
#SBATCH -e logs/smoke27b_%j.err
#SBATCH --no-requeue
#SBATCH -t 00:40:00
cd $HOME/WORK/exam-society
. venv/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "NODE=$(hostname)"; nvidia-smi -L
echo "================ PASS A: production path, single A800 (CUDA_VISIBLE_DEVICES=0) ================"
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_dpo_27b.py --device-map single --model models/Qwen3.8-27B || echo "PASS_A_EXIT=$?"
echo "================ PASS B: 2x A800 device_map=auto (sharded) ================"
CUDA_VISIBLE_DEVICES=0,1 python scripts/smoke_dpo_27b.py --device-map auto --model models/Qwen3.8-27B || echo "PASS_B_EXIT=$?"
echo "================ SMOKE DONE ================"
