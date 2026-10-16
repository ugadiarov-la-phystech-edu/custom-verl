#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-kvfp8-ppo-decoupled-k4-tisc8-tokenmean-qwen2.5-7b-2+6
#
# The Qwen2.5-7B 2+6 FP8-KV replay arm (..._2+6_..._fp8_kv-fp8_tis-c8_token-mean_qwen2.5-7b.sh, every parameter
# inherited) with the OBJECTIVE OF THE SYNC TIS BASELINE (examples/baselines/run_qwen2.5-7b_dapo17k_grpo_sync_tis_*)
# instead of bypass REINFORCE + TIS:
#   * pi_old recomputed by the trainer once per update (its weights at the start of the update: an extra forward
#     pass over the update's sequences, logged as timing_s/old_log_prob, entropy as actor/entropy);
#   * the update split into UPDATE_MINIBATCHES=4 optimizer mini-batches of 36 groups (36 x 16 = 576 sequences), all
#     against the same pi_old, then ONE weight sync: as the sync run's 144-prompt batch / 36-prompt mini-batches;
#   * PPO-clip on r = pi_theta / pi_old with clip_ratio_low=0.2, clip_ratio_high=0.28 (DAPO clip-higher) and the dual
#     clip c=10 (policy_loss.loss_mode=vanilla, bypass_mode=False);
#   * token TIS min(pi_old / pi_rollout, 8) (rollout_is=token, rollout_is_threshold=8, inherited), computed once per
#     update on the trainer (rollout_corr/* metrics: pi_old vs the FP8 rollout policy).
#
# WHY. Every bypass REINFORCE arm collapsed its entropy (0.06-0.09 by step ~120); the C=8 ones then diverged at steps
# 160-180, the token-mask ones stayed stable at AIME-2025 ~0.05. The sync TIS run kept entropy at 0.2-0.33 and reached
# AIME-2025 0.12 with this objective. With one optimizer step per pi_old (UPDATE_MINIBATCHES=1) pi_theta == pi_old at
# the gradient, the clip never acts and the loss is gradient-identical to bypass REINFORCE + TIS: the clip-higher
# effect needs mini-batches 2-4, where pi_theta has moved away from pi_old.
#
# WHAT CHANGES BEYOND THE LOSS (derived, no extra knobs):
#   * an update consumes 4 x 36 = 144 groups (async_training.require_batches=4), so parameter versions tick once per
#     4 optimizer steps; the fresh-share gate is ceil(0.5 x 144) = 72 newly arrived groups per update and the first
#     update (requires_mini_batches=0.5) is 72 groups = 2 optimizer steps; the rollouter's staleness quota and
#     concurrency ramp follow the same 144-group update size;
#   * the ESS brake still measures each optimizer mini-batch's pi_theta / pi_rollout and brakes that step;
#   * actor/pg_clipfrac and actor/pg_clipfrac_lower become non-zero (mini-batches 2-4); actor/ppo_kl is pi_theta vs
#     pi_old.
# The replay buffer is unchanged (fresh-first, tau=8, k=32, nu=1): replayed groups are corrected by
# min(pi_old / pi_rollout, 8) like the fresh ones.
# Knobs (base launcher, "IS / Rollout Correction"): OBJECTIVE, UPDATE_MINIBATCHES, clip_ratio_low / clip_ratio_high /
# clip_ratio_c. exp_name gains " ppo-decoupled-K4 clip-0.2-0.28-c10".
# test_freq=10, save_freq=10 (2+6 arm: 40): validation and hf_model checkpoints every 10 UPDATES. One update is 4
# optimizer steps over 144 groups here, so this is every 40 optimizer steps / 1,440 groups.
set -euo pipefail

export OBJECTIVE=${OBJECTIVE:-decoupled}
export UPDATE_MINIBATCHES=${UPDATE_MINIBATCHES:-4}
export clip_ratio_low=${clip_ratio_low:-0.2}
export clip_ratio_high=${clip_ratio_high:-0.28}
export clip_ratio_c=${clip_ratio_c:-10}
export test_freq=${test_freq:-10}
export save_freq=${save_freq:-10}

exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_2+6_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_kv-fp8_tis-c8_token-mean_qwen2.5-7b.sh" "$@"
