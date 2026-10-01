#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8
#
# FP8-ROLLOUT variant of the Qwen3-8B replay / min-ESS arm
#   grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5.sh
# (read its header). The ONLY difference is the vLLM rollout precision: ROLLOUT_QUANT=fp8, i.e.
# actor_rollout_ref.rollout.quantization=fp8 (verl's online 128x128-block FP8; the Megatron trainer stays bf16
# and every NCCL weight sync ships bf16 that vLLM re-quantizes). The replay objective already applies token TIS
# (C=2) against the rollout log-probs, which absorbs the FP8/bf16 mismatch. Tagged " rollout-fp8" in exp_name.
#
# Launch from the repo root, with the data paths of the machine, e.g. on remote_h200:
#   source <repo>/activate.sh
#   TRAIN_FILE=/data2/datasets/math_datasets/dapo/dapo-math-17k.parquet \
#   TEST_FILE="['/data2/datasets/math_datasets/dapo/aime-2024.parquet','/data2/datasets/math_datasets/dapo/aime-2025.parquet']" \
#   bash verl/experimental/fully_async_policy/shell/vcpo/replay_buffer/<this script>
# Every env knob / trailing Hydra override of the base script still applies.
#
# NOT COVERED BY VERL'S TESTS: fp8 with fully-async (standalone vLLM replicas that first load the bf16
# checkpoint into an FP8 model, then get FP8-requantized weights over NCCL each sync). Smoke-test before a
# full run and check the rollout logs for "DeepGEMM warmup", "Staged N FP8 layers for refit" (N > 0) and
# "FP8 weights loaded" after each sync; watch rollout_corr/* and staleness/ess (FP8 noise lowers ESS, so the
# min-ESS brake may fire more often than in the bf16 arm).
#
# DEEPGEMM (FP8 rollout kernels on Hopper). vLLM runs block-FP8 linears through DeepGEMM, which JIT-compiles
# its kernels and therefore needs a CUDA toolkit (CUDA_HOME); without one its import fails and vLLM falls
# back to slower kernels. When CUDA_HOME is unset and DEEPGEMM_CUDA_HOME (default: remote_h100's userspace
# toolkit /home/jovyan/ugadiarov/cuda-12.9) has nvcc, this script exports
#   CUDA_HOME=$DEEPGEMM_CUDA_HOME
#   DG_JIT_CACHE_DIR=/home/jovyan/ugadiarov/cache/deep_gemm   compiled kernels on persistent storage
#   VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0   FlashInfer's small-batch block-FP8 GEMM would otherwise JIT too
#       and needs cuBLAS/cuRAND headers the userspace toolkit lacks (engine start-up crash)
# Each is left alone if already set (remote_h200's activate.sh sets all three), and nothing is exported
# where the toolkit is absent. The rollouter's standalone vLLM replicas are Ray actors started from this
# shell, so they inherit the environment.
# H100 EMULATION IS ON BY DEFAULT (for remote_h200's H200s, 139.8 GiB usable each). In this layout every GPU
# holds one role, so each is emulated on its own:
#   * gpu_memory_utilization=0.5    rollout GPUs: vLLM gets 0.5 x 139.8 = 69.9 GiB, i.e. 0.88 of an H100 (79.65 GiB).
#                                   Not the arm's 0.9 (0.513 here, 71.7 GiB): vLLM's budget excludes the NCCL
#                                   weight sync's 2 x 2048 MB receive buffers and the FP8 re-quantization
#                                   temporaries, and at 0.513 the 5+3 smoke (Qwen3-4B) peaked at 83,263 MiB on
#                                   every rollout GPU, over an H100's 81,559.
#   * VERL_GPU_MEM_CAP_GB=78        trainer GPUs: caps the Megatron trainer's allocator at a whole H100 (79.65 GiB)
#                                   minus ~1.5 GiB outside it (CUDA context, NCCL, cuBLAS).
# Both are tagged in exp_name (" h100-emu-78gb-gmu0.5"). The cap cannot bound vLLM: after each FP8 weight
# sync vLLM may hold a few GB over its budget, so sample nvidia-smi on the rollout GPUs; > ~80,000 MiB would not
# fit a real H100 (81,559 MiB).
# ON A REAL H100 OVERRIDE BOTH: VERL_GPU_MEM_CAP_GB= gpu_memory_utilization=0.88 bash <this script>
# (an empty VERL_GPU_MEM_CAP_GB disables the cap).
# SCHEDULE (differs from the bf16 arm): lr_warmup_steps=12 (linear, in optimizer updates; arm: 0),
# validation and hf_model checkpoints every 12 updates (test_freq=12, save_freq=12; arm: 25), SEED=1.
# All env-overridable; the warmup is tagged " warmup-12" in exp_name. The warmup must be shorter than lr_decay_steps
# (default total_rollout_steps = 66000; Megatron asserts it), not than max_updates.
set -euo pipefail
export ROLLOUT_QUANT=${ROLLOUT_QUANT:-fp8}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-78}
export gpu_memory_utilization=${gpu_memory_utilization:-0.5}
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
