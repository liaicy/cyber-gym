#!/bin/bash
# serve base + optional LoRA adapter with an OpenAI-compatible API for rollout eval.
# usage: ADAPTER=runs/sft_v1/final sbatch scripts/sbatch_serve.sh   (omit ADAPTER for base only)
#SBATCH -J es_serve
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/serve_%j.out
#SBATCH -e logs/serve_%j.err
#SBATCH --no-requeue
#SBATCH -t 02:00:00
cd $HOME/WORK/exam-society
. venv/bin/activate
export HF_HUB_OFFLINE=1
# Node CUDA toolkit is 12.4; torch is cu130. Its nvcc rejects torch inductor's
# newer flags (--compress-mode=size), so JIT compile fails. Prefer the pip cu13
# nvcc if present; and run eager to skip torch.compile/cudagraph capture entirely.
NVCC_DIR=$(python -c "import os,nvidia.cuda_nvcc as m;print(os.path.dirname(m.__file__))" 2>/dev/null)
if [ -n "$NVCC_DIR" ] && [ -x "$NVCC_DIR/bin/nvcc" ]; then
  export CUDA_HOME="$NVCC_DIR"; export PATH="$NVCC_DIR/bin:$PATH"
  echo "USING_NVCC=$($NVCC_DIR/bin/nvcc --version | tail -1)"
fi
# FlashInfer JIT-compiles its sampling kernel via ninja->nvcc at first forward,
# which hits the same 12.4-nvcc-vs-cu130 mismatch. Force torch-native sampler and
# the prebuilt FlashAttention backend so nothing is compiled at startup.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_DISABLE_FLASHINFER_PREFILL=1
echo "SERVE_NODE=$(hostname) PORT=8000"
EXTRA="--enforce-eager"
if [ -n "$ADAPTER" ]; then EXTRA="$EXTRA --enable-lora --lora-modules sft=$ADAPTER --max-lora-rank 64"; fi
python -m vllm.entrypoints.openai.api_server --model models/Qwen3-8B --served-model-name Qwen/Qwen3-8B \
  --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.9 --dtype bfloat16 $EXTRA
