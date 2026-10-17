#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-kvfp8-ppo-decoupled-k4-tisc8-tokenmean-csr64-overlong-qwen2.5-7b-1+7
#
# The 1+7 csr64 arm (..._1+7_..._csr64_seqs1024_mbt16k_qwen2.5-7b.sh: 1 rollout + 7 trainer GPUs, decoupled K4 PPO-clip
# 0.2/0.28/c10, token TIS C=8, 64 groups / 1024 sequences in flight, every parameter inherited) WITH DAPO's soft
# overlong punishment, as the sync arm examples/baselines/run_qwen2.5-7b_dapo17k_grpo_sync_tis_overlong_B128xn16_
# mini32_megatron.sh: OVERLONG_BUFFER_LEN=2048, OVERLONG_PENALTY_FACTOR=1.0.
#
# WHY. In the Qwen2.5-7B sync TIS runs truncation at the 8,192-token cap rose to ~1-2% of responses together with an
# entropy rise and falling AIME accuracy; the overlong penalty kept entropy down there. A truncated response scores
# -1 and, under token-mean, puts all 8,192 of its tokens into the gradient.
#
# OVERLONG PENALTY (the base launcher's "DAPO overlong reward shaping" block). reward.reward_manager.name=dapo with
#   reward += min(-(len - (8192 - 2048)) / 2048 * 1.0, 0)
# i.e. 0 up to 6,144 response tokens, then a linear ramp to -1 at the 8,192 cap: a correct answer of 8,000 tokens
# scores ~+0.07, a truncated one -2. Tagged in exp_name (" overlong-2048-pf1.0").
#   * The rollouter's insertion gate sees the shaped rewards: groups whose 16 rewards all agree are still dropped,
#     but an all-wrong group whose responses run into the buffer by different amounts is kept and trains on the
#     length penalty alone (the sync arm trains every group).
#   * critic/score/mean includes the penalty (not comparable with the csr64 run); val-core/*/acc/mean@1 is the
#     scorer's acc and unaffected; val-aux/*/overlong/mean@1 is the share of validation responses in the buffer.
#   * Not adopted from DAPO / FlashRL: overlong filtering (masking truncated responses out of the loss) and dynamic
#     sampling beyond the rollouter's equal-reward gate.
#
# Any env knob or trailing Hydra override of the csr64 script and the base launcher still applies, e.g.
#   OVERLONG_BUFFER_LEN=1024 OVERLONG_PENALTY_FACTOR=0.5 bash <this script>
# OVERLONG_BUFFER_LEN=0 turns the penalty off (= the csr64 script).
set -euo pipefail

export OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-2048}
export OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_1+7_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_kv-fp8_ppo-decoupled-K4-clip-0.2-0.28-c10_tis-c8_token-mean_csr64_seqs1024_mbt16k_qwen2.5-7b.sh" "$@"
