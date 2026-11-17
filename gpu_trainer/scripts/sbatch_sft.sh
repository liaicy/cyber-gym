#!/bin/bash
#SBATCH -J es_sft
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/sft_%j.out
#SBATCH -e logs/sft_%j.err
#SBATCH --no-requeue
#SBATCH -t 06:00:00
cd $HOME/WORK/exam-society
. venv/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi -L
python scripts/sft_lora.py --data ${DATA:-data/train.jsonl} --out runs/${RUN:-sft_v1} \
    --exclude-tasks ${EXCLUDE:-arvo:10628,arvo:11007} "$@"
