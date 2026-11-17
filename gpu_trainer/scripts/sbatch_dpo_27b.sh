#!/bin/bash
# DPO-LoRA on Qwen3.8-27B via FSDP (full-shard / ZeRO-3) with accelerate.
# The "DPO first" leg of the 27B staged recipe.
#
# usage (default 4x A800; override --gres to add/remove cards, accelerate
# follows the allocation automatically):
#   DATA=data/dpo_27b_v1.jsonl RUN=dpo_27b_v1 sbatch scripts/sbatch_dpo_27b.sh
#   DATA=... RUN=... sbatch --gres=gpu:8 scripts/sbatch_dpo_27b.sh            # more cards
#   DATA=data/dpo_train.jsonl RUN=smoke_fsdp_27b sbatch scripts/sbatch_dpo_27b.sh --max-steps 3
#
# WHY FSDP (measured 2026-10-27, smoke job 1037362 on g13):
#   1x A800 bf16 (max_len 12288, LoRA r32, grad-ckpt) OOMs by ~0.5 GB (peak
#   78.4/79.27 GiB). device_map="auto" on 2 cards trains but piles activations
#   on GPU0 (peak 78.5 GiB) and runs the cards serially. bitsandbytes is absent
#   so QLoRA is out. FSDP shards params/grads/optim across all cards -> per-card
#   memory ~1/N and all cards compute in parallel; add cards for headroom/speed.
#
# DATA: the real run needs the 27B chosen/rejected export (teacher-chosen vs
#   27B-target rejected), not generated yet (teacher pool still collecting, 27B
#   target rollouts pending). For a mechanics/memory smoke, any dpo jsonl +
#   --max-steps N works (content is irrelevant to the FSDP check).
#SBATCH -J es_dpo27b
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o logs/dpo27b_%j.out
#SBATCH -e logs/dpo27b_%j.err
#SBATCH --no-requeue
#SBATCH -t 04:00:00
cd $HOME/WORK/exam-society
. venv/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(echo ${CUDA_VISIBLE_DEVICES:-0} | tr ',' '\n' | grep -c .)
PORT=$((20000 + RANDOM % 20000))
echo "NODE=$(hostname) NGPU=$NGPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PORT=$PORT"
nvidia-smi -L
accelerate launch --config_file scripts/fsdp_qwen3_5.yaml \
  --num_processes $NGPU --main_process_port $PORT \
  scripts/dpo_lora.py --model models/Qwen3.8-27B \
  --data ${DATA:?set DATA to a dpo export jsonl} --out runs/${RUN:-dpo_27b_v1} "$@"
