#!/bin/bash
# serve base Qwen3-8B + 8 k-fold LoRA adapters (strict/cross x fold0-3) for eval.
#SBATCH -J es_serve_kfold
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/serve_kfold_%j.out
#SBATCH -e logs/serve_kfold_%j.err
#SBATCH --no-requeue
#SBATCH -t 02:00:00
cd $HOME/WORK/exam-society
. venv/bin/activate
export HF_HUB_OFFLINE=1
NVCC_DIR=$(python -c "import os,nvidia.cuda_nvcc as m;print(os.path.dirname(m.__file__))" 2>/dev/null)
if [ -n "$NVCC_DIR" ] && [ -x "$NVCC_DIR/bin/nvcc" ]; then
  export CUDA_HOME="$NVCC_DIR"; export PATH="$NVCC_DIR/bin:$PATH"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_DISABLE_FLASHINFER_PREFILL=1
echo "SERVE_NODE=$(hostname) PORT=8000"
python -m vllm.entrypoints.openai.api_server --model models/Qwen3-8B --served-model-name Qwen/Qwen3-8B \
  --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.9 --dtype bfloat16 --enforce-eager \
  --enable-lora --max-lora-rank 64 --max-loras 8 \
  --lora-modules \
    strict_fold0=runs/kfold/strict_fold0/final strict_fold1=runs/kfold/strict_fold1/final \
    strict_fold2=runs/kfold/strict_fold2/final strict_fold3=runs/kfold/strict_fold3/final \
    cross_fold0=runs/kfold/cross_fold0/final cross_fold1=runs/kfold/cross_fold1/final \
    cross_fold2=runs/kfold/cross_fold2/final cross_fold3=runs/kfold/cross_fold3/final
