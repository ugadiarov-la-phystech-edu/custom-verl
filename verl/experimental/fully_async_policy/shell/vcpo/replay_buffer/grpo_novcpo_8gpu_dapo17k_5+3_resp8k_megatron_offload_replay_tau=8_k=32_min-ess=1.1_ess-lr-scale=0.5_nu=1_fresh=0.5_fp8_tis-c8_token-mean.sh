#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-tisc8-tokenmean
set -euo pipefail
export rollout_is_threshold=${rollout_is_threshold:-8}
export loss_agg_mode=${loss_agg_mode:-token-mean}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-78}
export gpu_memory_utilization=${gpu_memory_utilization:-0.5}
export lr_warmup_steps=${lr_warmup_steps:-12}
export SEED=${SEED:-1}
export test_freq=${test_freq:-12}
export save_freq=${save_freq:-12}
exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8.sh" "$@"
