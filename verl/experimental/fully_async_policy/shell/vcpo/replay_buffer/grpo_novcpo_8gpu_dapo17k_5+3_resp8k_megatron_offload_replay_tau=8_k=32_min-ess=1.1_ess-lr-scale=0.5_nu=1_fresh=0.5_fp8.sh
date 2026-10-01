#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8
set -euo pipefail
export ROLLOUT_QUANT=${ROLLOUT_QUANT:-fp8}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-78}
export gpu_memory_utilization=${gpu_memory_utilization:-0.513}
export lr_warmup_steps=${lr_warmup_steps:-12}
export SEED=${SEED:-1}
export test_freq=${test_freq:-12}
export save_freq=${save_freq:-12}
DEEPGEMM_CUDA_HOME=${DEEPGEMM_CUDA_HOME:-/home/jovyan/ugadiarov/cuda-12.9}
if [[ -z "${CUDA_HOME:-}" && -x "${DEEPGEMM_CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_HOME="${DEEPGEMM_CUDA_HOME}"
    export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/home/jovyan/ugadiarov/cache/deep_gemm}
    export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}
fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5.sh" "$@"
