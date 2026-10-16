#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-kvfp8-ppo-decoupled-k4-tisc8-tokenmean-qwen2.5-7b-1+7
#
# The decoupled K4 replay arm (..._2+6_..._fp8_kv-fp8_ppo-decoupled-K4-clip-0.2-0.28-c10_tis-c8_token-mean_qwen2.5-7b.sh,
# every parameter inherited: pi_old recomputed per update, 4 optimizer mini-batches per update, PPO-clip 0.2/0.28/c10,
# token TIS min(pi_old / pi_rollout, 8), staleness 8 / k 8 / tau 2, test / save every 10 updates) on a 1+7 layout:
#   * n_gpus_rollout=1: ONE vLLM rollout GPU, 7 Megatron trainer GPUs (tp1dp7);
#   * train_prompt_mini_bsz=35 (2+6 arm: 36): an optimizer mini-batch must split over the 7 trainer DP ranks, and
#     36 x 16 = 576 does not; 35 x 16 = 560 = 7 x 80 sequences per rank (as the 3+5 arm's B-35). The base launcher
#     rejects a mini-batch that does not split at launch.
#
# DERIVED (no extra knobs):
#   * one update = 4 x 35 = 140 groups (2+6: 144); the first update (requires_mini_batches=0.5) = 2 mini-batches =
#     70 groups; fresh floor ceil(0.5 x 140) = 70 groups per update; rollouter quota 140 x (8 + 1) = 1,260 groups;
#   * one replica: concurrent_samples_per_replica=32 -> 32 groups (512 sequences, ROLLOUT_MAX_NUM_SEQS=512) in flight,
#     concurrency ramp [5, 12, 20] groups until the 1st / 2nd / 3rd update's groups are delivered.
#
# EXPECTED REGIME (estimated from the 2+6 decoupled run, where 2 engines made ~1.9 groups/s and the trainer was the
# bottleneck at ~80 s of compute per 144-group update): one engine makes ~0.9-1.0 groups/s while 7 trainer GPUs need
# ~140 groups per ~70 s, so the run becomes ROLLOUT-bound. Each update waits for the 70-group fresh floor and the
# buffer fills the rest: expect ~40-50% replayed groups per update (spread evenly and staleness-stratified over the 4
# optimizer steps), trainer idle time, and ~2x longer validation pauses (validation runs on the single engine).
# Not step-comparable with the sync run (140 groups per step, not 144): compare by optimizer steps or groups.
# exp_name: "... 1-7 tp1dp7 hdo B-35 ...".
set -euo pipefail

export n_gpus_rollout=${n_gpus_rollout:-1}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-35}

exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_2+6_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_kv-fp8_ppo-decoupled-K4-clip-0.2-0.28-c10_tis-c8_token-mean_qwen2.5-7b.sh" "$@"
