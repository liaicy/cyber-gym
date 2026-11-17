#!/bin/bash
# Serve Qwen3.8-27B (the new target) with an OpenAI-compatible API for rollout
# eval, optionally with LoRA adapters. Mirrors sbatch_serve.sh but points at the
# 27B weights and exposes TP / mem-util / max-len as env knobs.
#
# usage:
#   sbatch scripts/sbatch_serve_27b.sh                      # base only, 1 GPU
#   TP=2 sbatch --gres=gpu:2 scripts/sbatch_serve_27b.sh    # tensor-parallel 2
#   ADAPTER=runs/dpo_27b_v1/final sbatch scripts/sbatch_serve_27b.sh
#   PORT=8001 sbatch scripts/sbatch_serve_27b.sh            # 2026-10-29: needed when
#     slurm lands >1 of these jobs on the SAME node (a01 nodes have 8 GPUs each,
#     multiple single-GPU jobs can co-locate) -- --port was hardcoded 8000 before,
#     so a second job on the same node crashed binding the same port.
#
# 27B bf16 weights are ~54 GB; on one 80 GB A800 that leaves ~26 GB for KV at
# util 0.92, enough for a single-stream rollout at 32k ctx. Bump TP=2 (+ override
# --gres=gpu:2) if KV is tight or you want more concurrency.
#SBATCH -J es_serve27b
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/serve27b_%j.out
#SBATCH -e logs/serve27b_%j.err
#SBATCH --no-requeue
#SBATCH -t 02:00:00
cd $HOME/WORK/exam-society
. venv/bin/activate
export HF_HUB_OFFLINE=1
# Same old-nvcc-vs-cu130 JIT workaround as the 8B serve path (see memory
# gpu-vllm-multilora-jit-fail): prefer the pip cu13 nvcc, force torch-native
# sampler + prebuilt FlashAttention so nothing is compiled at first forward.
NVCC_DIR=$(python -c "import os,nvidia.cuda_nvcc as m;print(os.path.dirname(m.__file__))" 2>/dev/null)
if [ -n "$NVCC_DIR" ] && [ -x "$NVCC_DIR/bin/nvcc" ]; then
  export CUDA_HOME="$NVCC_DIR"; export PATH="$NVCC_DIR/bin:$PATH"
  echo "USING_NVCC=$($NVCC_DIR/bin/nvcc --version | tail -1)"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_DISABLE_FLASHINFER_PREFILL=1
TP=${TP:-1}; MEMUTIL=${MEMUTIL:-0.92}; MAXLEN=${MAXLEN:-32768}; PORT=${PORT:-8000}
# 2026-11-01: MODEL/NAME knobs so a merged DPO checkpoint can be served as a plain model
# (LoRA hot-load is out: no pip nvcc in venv -> punica JIT dies). e.g.
#   MODEL=runs/dpo_27b_v1_2026_10_30/merged NAME=dpo PORT=8001 sbatch -t 12:00:00 scripts/sbatch_serve_27b.sh
MODEL=${MODEL:-models/Qwen3.8-27B}; NAME=${NAME:-Qwen/Qwen3.8-27B}
echo "SERVE_NODE=$(hostname) MODEL=$MODEL NAME=$NAME PORT=$PORT TP=$TP MEMUTIL=$MEMUTIL MAXLEN=$MAXLEN"
nvidia-smi -L
EXTRA="--enforce-eager --tensor-parallel-size $TP"
if [ -n "$ADAPTER" ]; then EXTRA="$EXTRA --enable-lora --lora-modules dpo=$ADAPTER --max-lora-rank 64"; fi
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL --served-model-name $NAME \
  --port $PORT --max-model-len $MAXLEN --gpu-memory-utilization $MEMUTIL \
  --dtype bfloat16 $EXTRA
