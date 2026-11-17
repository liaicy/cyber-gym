#!/bin/bash
#SBATCH -J es_dpo
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/dpo_%j.out
#SBATCH -e logs/dpo_%j.err
#SBATCH --no-requeue
#SBATCH -t 01:30:00
cd $HOME/WORK/exam-society
. venv/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
python scripts/dpo_lora.py --data ${DATA:-data/dpo_train.jsonl} --out runs/${RUN:-dpo_v1} "$@"
