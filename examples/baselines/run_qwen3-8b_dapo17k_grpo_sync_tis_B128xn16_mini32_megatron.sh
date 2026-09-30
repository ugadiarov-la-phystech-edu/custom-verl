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
#
# Everything else - data, geometry, loss, optimizer, seeds, checkpoints, memory settings - is
# run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh, which this script runs with TIS=True
# (see its "Rollout precision / TIS toggles" block). The pair isolates the precision change:
# fp8+TIS vs bf16+TIS differ ONLY in rollout.quantization, while bf16+TIS vs the plain baseline
# measures what TIS itself does (it changes training even in bf16 and can depress the LOGGED
# training reward, which is sampled from vLLM, while improving validation - judge by val-core/*).
#
# TIS: per-token PPO loss x min(pi_trainer_old / pi_vllm, C), C = TIS_THRESHOLD (default 2.0).
# The weights come from the rollout log-probs vLLM already returns (processed_logprobs, i.e. after
# temperature / top-p); old_log_probs are still recomputed by the trainer (bypass_mode=false).
# Watch rollout_corr/* (IS-weight tails, KL, pearson) alongside actor/entropy.
#
# Expected FP8 effect on this 8B arm: little raw kernel speedup (FlashRL measured 0.96-1.05x for FP8
# at 7B; verl's FP8 guide reports 12-18% for Qwen3-8B), but the rollout here is KV-cache bound at
# gpu_memory_utilization=0.5 (~21 GiB KV per GPU), and halving the weights adds ~8 GB of KV.
# Measure before assuming a speedup (verl's published FP8 numbers are from H100).
#
# Any env knob or trailing Hydra override of the base script still applies, e.g.
#   TIS_THRESHOLD=4 max_updates=200 ROLLOUT_QUANT=fp8 bash <this script>

set -euo pipefail
export TIS=${TIS:-True}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh" "$@"
