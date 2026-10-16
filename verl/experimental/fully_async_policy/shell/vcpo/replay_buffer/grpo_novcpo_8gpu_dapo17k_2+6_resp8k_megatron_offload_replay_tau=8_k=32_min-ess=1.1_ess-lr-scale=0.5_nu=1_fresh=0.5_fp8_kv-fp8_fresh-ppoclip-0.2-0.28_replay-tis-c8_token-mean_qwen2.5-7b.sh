#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-kvfp8-freshppoclip-replaytisc8-tokenmean-qwen2.5-7b-2+6
#
# The Qwen2.5-7B 2+6 FP8-KV replay arm (..._2+6_..._fp8_kv-fp8_tis-c8_token-mean_qwen2.5-7b.sh, every parameter
# inherited) with a MIXED objective inside each mini-batch:
#   * fresh sequences (newly arrived groups, never trained on; tagged is_fresh by the trainer): PPO-clip on
#     r = pi_theta / pi_rollout with clip_ratio_low=0.2, clip_ratio_high=0.28 (DAPO's clip-higher) and the dual
#     clip c=3, no IS weight (the ratio carries the correction);
#   * replayed sequences (drawn from the buffer, trained on before): REINFORCE with token-level TIS, C=8, as in
#     the source arm.
# Both parts are aggregated together in one token-mean pass (compute_policy_loss_bypass_mode with
# rollout_correction.replay_loss_type=reinforce, verl/trainer/ppo/core_algos.py), so the batch is weighted as a
# single-loss batch would be.
#
# WHY. In the source arm the entropy fell 0.62 -> 0.06 within ~150 updates while the rollout/trainer Pearson
# correlation dropped 0.996 -> 0.89 and the k3 KL rose 0.007 -> 0.019; the Qwen2.5-7B sync TIS baseline
# (PPO-clip 0.2/0.28) kept its entropy at 0.2-0.6. REINFORCE has no ratio clip, so clip-higher had no effect there.
# Clipping r = pi_theta / pi_rollout on fresh data adds clip-higher's entropy effect and limits how far the trainer
# drifts from the rollout policy. Replayed groups can be up to k=32 versions stale: PPO-clip would cut off most of
# their tokens, so they keep REINFORCE + truncated TIS. Note: one update per training step, so pi_old == pi_theta;
# only the bypass ratio (old = rollout log-probs) moves, through staleness and the FP8-vs-bf16 mismatch, and part of
# what the clip sees is that numerical mismatch. See notes/replay_mixed_loss_and_replay_only_ess.md.
#
# Unchanged: the min-ESS LR brake (global ESS of the whole mini-batch, min_ess=1.1, lr x0.5), the fresh-share gate
# (min_fresh_ratio=0.5), tau / k, C=8 (replayed tokens), token-mean, csr 32 / max_num_seqs 512,
# VERL_QWEN2_K_BIAS_CENTER=1, val 1.0 / 1.0, test / save every 40.
# Logged: actor/pg_clipfrac and actor/pg_clipfrac_lower (fresh tokens only), actor/mixed/{fresh,replay}_token_frac,
# replay/minibatch_fresh_seqs. exp_name gains " fresh-ppoclip-0.2-0.28 replay-reinforce".
# Knobs: loss_type, replay_loss_type, clip_ratio_low, clip_ratio_high (base launcher's "IS / Rollout Correction"
# block). The replay trainer requires REINFORCE on replayed sequences: replay_loss_type=null with loss_type=ppo_clip
# (PPO-clip on every sequence) is rejected at start-up.
set -euo pipefail

export loss_type=${loss_type:-ppo_clip}
export replay_loss_type=${replay_loss_type:-reinforce}
export clip_ratio_low=${clip_ratio_low:-0.2}
export clip_ratio_high=${clip_ratio_high:-0.28}

exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_2+6_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_kv-fp8_tis-c8_token-mean_qwen2.5-7b.sh" "$@"
