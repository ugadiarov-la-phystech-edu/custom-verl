#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-dapo17k-grpo-qwen3-8b-tis

# Qwen3-8B synchronous baseline WITH truncated importance sampling (TIS), in two rollout precisions:
#
#   bash examples/baselines/run_qwen3-8b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh                     # bf16 + TIS
#   ROLLOUT_QUANT=fp8 bash examples/baselines/run_qwen3-8b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh   # fp8 + TIS
#   ROLLOUT_QUANT=int8 bash examples/baselines/run_qwen3-8b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh  # int8 + TIS
#
# FLASHRL PARAMETERS. Four knobs follow FlashRL's DAPO-Qwen2.5-32B recipe (yaof20/verl, branch
# flash-rl, recipe/flash_rl/dapo_qwen32b_{bf16,int8}.sh), overriding the baseline, plus a shorter warmup:
#   * TIS cap C            TIS_THRESHOLD=8        (baseline arm: TIS off; verl's FP8 guide uses 2)
#   * clip ratio low/high  0.2 / 0.28 (DAPO clip-higher; baseline 0.2 / 0.2)
#   * dual-clip c          10.0                   (baseline 3.0)
#   * LR warmup            3 steps, linear, then constant (baseline 0; FlashRL: 10). Both FlashRL's FSDP
#                          worker and v0.9.0's engine step the scheduler once per ROLLOUT step, so this is
#                          3 rollout steps (= 12 optimizer updates here).
#   * weight decay         0.1                    (baseline 0.01)
# All five are env-overridable and tagged in exp_name.
#
# CADENCE. Validation and checkpoints every 3 rollout steps (= 12 optimizer updates; baseline 2):
# test_freq=3, save_freq=3, env-overridable.
#
# H100 EMULATION IS ON BY DEFAULT (for runs on remote_h200's 140.4 GiB H200s; see the baseline's
# "H100 emulation" note and verl/utils/gpu_memory_cap.py):
#   * VERL_GPU_MEM_CAP_GB=76      caps the trainer (actor-role worker) allocator at 76 GiB: 79.65 GiB of
#                                 an H100 minus the sleeping vLLM process and ~1.5 GiB outside the
#                                 allocator. A starting value - re-derive it from nvidia-smi
#                                 --query-compute-apps during update_actor.
#   * gpu_memory_utilization=0.283 gives vLLM the H100's absolute budget, 0.5 x 79.65 / 140.4 GiB.
# Both are tagged in exp_name (" h100-emu-76gb-gmu0.283"). The cap cannot bound the rollout phase
# (resident trainer + vLLM): watch per-GPU memory with an nvidia-smi sampler; > ~78 GB would OOM an H100.
# ON A REAL H100 OVERRIDE BOTH: VERL_GPU_MEM_CAP_GB= gpu_memory_utilization=0.5 bash <this script>
# (0.283 of an 80 GiB card would halve vLLM's KV cache; an empty VERL_GPU_MEM_CAP_GB disables the cap).
#
# DYNAMIC BATCH SIZE is on here (DYNAMIC_BSZ=True, cap DYNAMIC_BSZ_MAX_TOKENS=10240 tokens per GPU per
# micro-batch = one full-length sequence, so the memory peak stays at the measured worst case). It
# packs short sequences into shared passes for old_log_prob / update_actor without changing the loss
# (normalized per mini-batch); see the baseline's "Dynamic batch size" block. FlashRL used dynamic
# batching too (cap = prompt + response). DYNAMIC_BSZ=False restores micro-batch 1. The rest of FlashRL's recipe is NOT adopted:
# 512-prompt batches, 20k responses + DAPO overlong penalty, dynamic sampling (filter_groups),
# FSDP + SP8, rollout TP2, val T=1.0 - see the baseline header for this arm's settings.
#
# Everything else - data, geometry, loss aggregation, optimizer, seeds, checkpoints, memory - is
# run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh, which this script runs with TIS=True
# (see its "Rollout precision / TIS toggles" block). fp8+TIS vs bf16+TIS differ ONLY in
# rollout.quantization, so the pair isolates the rollout precision. bf16+TIS vs the plain baseline
# is NOT a TIS-only ablation: it also changes clip, dual-clip, warmup and weight decay (set those five
# env vars back to the baseline values for that). TIS changes training even in bf16 and can depress the
# LOGGED training reward, which is sampled from vLLM, while improving validation - judge by val-core/*.
#
# TIS: per-token PPO loss x min(pi_trainer_old / pi_vllm, C), C = TIS_THRESHOLD (default 8, FlashRL's
# DAPO-32B value; this is exactly FlashRL's imp_ratio_cap formula, applied after the PPO clip).
# The weights come from the rollout log-probs vLLM already returns (processed_logprobs, i.e. after
# temperature / top-p); old_log_probs are still recomputed by the trainer (bypass_mode=false).
# Watch rollout_corr/* (IS-weight tails, KL, pearson) alongside actor/entropy.
#
# Expected FP8 effect on this 8B arm: little raw kernel speedup (FlashRL measured 0.96-1.05x for FP8
# at 7B; verl's FP8 guide reports 12-18% for Qwen3-8B), but the rollout here is KV-cache bound at
# gpu_memory_utilization=0.5 (~21 GiB KV per GPU), and halving the weights adds ~8 GB of KV.
# Measure before assuming a speedup (verl's published FP8 numbers are from H100).
#
# DEEPGEMM (FP8 rollout kernels). vLLM runs block-FP8 linears through DeepGEMM, which JIT-compiles its
# kernels and therefore needs a CUDA toolkit (CUDA_HOME); without one its import fails and vLLM falls
# back to slower kernels. On remote_h100 (no system toolkit) a userspace CUDA 12.9 toolkit lives at
# DEEPGEMM_CUDA_HOME=/home/jovyan/ugadiarov/cuda-12.9. When CUDA_HOME is unset and that toolkit exists,
# this script exports:
#   CUDA_HOME=$DEEPGEMM_CUDA_HOME
#   DG_JIT_CACHE_DIR=/home/jovyan/ugadiarov/cache/deep_gemm   compiled kernels on persistent NFS (vLLM's
#       default ~/.cache/vllm/deep_gemm is /home/user, which is lost when the job is recreated)
#   VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0   vLLM would otherwise JIT FlashInfer's small-batch (M < 32)
#       block-FP8 GEMM too, which needs cuBLAS/cuRAND headers the userspace toolkit lacks and crashes
#       engine start-up; with 0, DeepGEMM serves every batch size.
# Each is left alone if already set, and nothing is exported where the toolkit is absent. Verified on
# remote_h100 (2026-09-30): DeepGEMM kernels compile with it and run correctly on the 12.6 driver for
# all Qwen3-1.7B/8B linear shapes. bf16 rollout does not use DeepGEMM.
#
# Any env knob or trailing Hydra override of the base script still applies, e.g.
#   TIS_THRESHOLD=2 max_updates=200 ROLLOUT_QUANT=fp8 bash <this script>

set -euo pipefail
export TIS=${TIS:-True}
export TIS_THRESHOLD=${TIS_THRESHOLD:-8}
export clip_ratio_low=${clip_ratio_low:-0.2}
export clip_ratio_high=${clip_ratio_high:-0.28}
export clip_ratio_c=${clip_ratio_c:-10.0}
export lr_warmup_steps=${lr_warmup_steps:-3}
export test_freq=${test_freq:-3}
export save_freq=${save_freq:-3}
export weight_decay=${weight_decay:-0.1}
export DYNAMIC_BSZ=${DYNAMIC_BSZ:-True}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-76}
export gpu_memory_utilization=${gpu_memory_utilization:-0.283}

DEEPGEMM_CUDA_HOME=${DEEPGEMM_CUDA_HOME:-/home/jovyan/ugadiarov/cuda-12.9}
if [[ -z "${CUDA_HOME:-}" && -x "${DEEPGEMM_CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_HOME="${DEEPGEMM_CUDA_HOME}"
    export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/home/jovyan/ugadiarov/cache/deep_gemm}
    export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}
fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh" "$@"
