#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-kvfp8-ppo-decoupled-k4-tisc8-tokenmean-csr64-qwen2.5-7b-1+7
#
# The 1+7 decoupled K4 arm (..._1+7_..._fresh=0.5_..._ppo-decoupled-K4-clip-0.2-0.28-c10_tis-c8_token-mean_qwen2.5-7b.sh,
# every parameter inherited: 1 rollout + 7 trainer GPUs, B-35, 4 optimizer mini-batches per update, PPO-clip
# 0.2/0.28/c10, token TIS min(pi_old / pi_rollout, 8), staleness 8 / k 8 / tau 2, test / save every 10 updates) with
# the single rollout engine run at twice the concurrency:
#   * concurrent_samples_per_replica=64 (1+7 arm: 32) groups in flight on the engine;
#   * ROLLOUT_MAX_NUM_SEQS=1024 (= 16 x 64; 1+7 arm: 512): vLLM's running cap matches the in-flight sequences;
#   * actor_rollout_ref.rollout.max_num_batched_tokens=16384 (base: 2048 + 8192 = 10240): ~1k decode tokens per step
#     leave ~15k for prefill (new groups and the re-prefill of every in-flight request after each weight sync);
#   * actor_rollout_ref.rollout.cudagraph_capture_sizes up to 1024 (vLLM default: 512): decode batches above 512
#     would otherwise run eagerly (slower, and where the rollout worker's memory grew);
#   * replay_min_fresh_ratio=0.5 (as the 1+7 arm): an update waits for 70 newly arrived groups and the buffer fills the
#     rest. The 1+7 run at 32 groups was rollout-bound with ~50% replay per update and its entropy blew up; the stable
#     2+6 decoupled run was ~100% fresh. replay_min_fresh_ratio=1.0 makes every update fresh-only (the trainer then
#     waits for 140 fresh groups per update).
#
# WHY these values (measured on the 1+7 / 2+6 decoupled runs): KV-cache use stayed <= 18% of the engine's
# 2,336,160-token cache with 0 preemptions, while only ~130-330 sequences ran on average (a group holds its slot until
# its slowest response ends). 64 groups ~ 1M tokens of context (~45% KV; up to ~70-90% late in a run with long
# responses): watch vllm/num_preemptions and vllm/kv_cache_usage/max. Expected ~1.3-1.6x the generation throughput
# (an estimate), still below the ~2 groups/s 7 trainer GPUs could consume: the trainer idles between updates.
# REQUIRES the rollout memory release before each weight sync (vLLMHttpServer.release_kv_cache ->
# release_cached_memory): 48 groups / 768 sequences died with a CUDA OOM at a sync without it. Watch rollout_mem/*.
# Trailing Hydra overrides still win (they follow these on the command line).
set -euo pipefail

export concurrent_samples_per_replica=${concurrent_samples_per_replica:-64}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
export replay_min_fresh_ratio=${replay_min_fresh_ratio:-0.5}

exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_1+7_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_kv-fp8_ppo-decoupled-K4-clip-0.2-0.28-c10_tis-c8_token-mean_qwen2.5-7b.sh" \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    "actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,544,576,608,640,672,704,736,768,800,832,864,896,928,960,992,1024]" \
    "$@"
