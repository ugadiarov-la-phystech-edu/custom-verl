#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-dapo17k-grpo-qwen3-8b-tis


set -euo pipefail
export TIS=${TIS:-True}
export TIS_THRESHOLD=${TIS_THRESHOLD:-8}
export clip_ratio_low=${clip_ratio_low:-0.2}
export clip_ratio_high=${clip_ratio_high:-0.28}
export clip_ratio_c=${clip_ratio_c:-10.0}
export lr_warmup_steps=${lr_warmup_steps:-10}
export weight_decay=${weight_decay:-0.1}
export DYNAMIC_BSZ=${DYNAMIC_BSZ:-True}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh" "$@"
