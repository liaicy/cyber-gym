#!/bin/bash
#SBATCH -J es_grpo_v1
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/grpo_v1_%j.out
#SBATCH -e logs/grpo_v1_%j.err
#SBATCH --no-requeue
#SBATCH -t 06:00:00
cd $HOME/WORK/exam-society
. venv/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "NODE=$SLURMD_NODENAME"
nvidia-smi -L
python scripts/grpo_train_v1.py --model $HOME/WORK/exam-society/models/Qwen3-8B \
  --out $HOME/WORK/exam-society/runs/grpo_v1_long --bridge-port 18666 --max-steps 150
