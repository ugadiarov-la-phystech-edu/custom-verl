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
# FLASHRL PARAMETERS. Five knobs follow FlashRL's DAPO-Qwen2.5-32B recipe (yaof20/verl, branch
# flash-rl, recipe/flash_rl/dapo_qwen32b_{bf16,int8}.sh), overriding the baseline:
#   * TIS cap C            TIS_THRESHOLD=8        (baseline arm: TIS off; verl's FP8 guide uses 2)
#   * clip ratio low/high  0.2 / 0.28 (DAPO clip-higher; baseline 0.2 / 0.2)
#   * dual-clip c          10.0                   (baseline 3.0)
#   * LR warmup            10 steps, linear, then constant (baseline 0). Both FlashRL's FSDP worker and
#                          v0.9.0's engine step the scheduler once per ROLLOUT step, so this is the same
#                          10 steps (= 40 optimizer updates here) as in FlashRL.
#   * weight decay         0.1                    (baseline 0.01)
# All five are env-overridable and tagged in exp_name. The rest of FlashRL's recipe is NOT adopted:
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
# Any env knob or trailing Hydra override of the base script still applies, e.g.
#   TIS_THRESHOLD=2 max_updates=200 ROLLOUT_QUANT=fp8 bash <this script>

set -euo pipefail
export TIS=${TIS:-True}
export TIS_THRESHOLD=${TIS_THRESHOLD:-8}
export clip_ratio_low=${clip_ratio_low:-0.2}
export clip_ratio_high=${clip_ratio_high:-0.28}
export clip_ratio_c=${clip_ratio_c:-10.0}
export lr_warmup_steps=${lr_warmup_steps:-10}
export weight_decay=${weight_decay:-0.1}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh" "$@"
