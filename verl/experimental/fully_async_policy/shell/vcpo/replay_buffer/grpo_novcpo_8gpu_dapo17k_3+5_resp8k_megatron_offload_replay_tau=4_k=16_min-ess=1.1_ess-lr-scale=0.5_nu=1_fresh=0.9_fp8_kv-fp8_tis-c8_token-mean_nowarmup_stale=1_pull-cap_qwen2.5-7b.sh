#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-tau4-k16-stale1-pullcap-ess-nu1-fresh0.9-fp8-kvfp8-tisc8-tokenmean-nowarmup-qwen2.5-7b-3+5
#
# FP8 + FP8-KV, PULL-CAP variant of the Qwen2.5-7B (BASE) 3+5 bf16 stale-1 arm
#   ..._replay_tau=4_k=16_..._fresh=0.9_bf16_tis-c8_token-mean_nowarmup_stale=1_qwen2.5-7b.sh (read its header: same
#   layout, batches, cadence and staleness settings). The differences:
#   * replay_pull_cap=True          the trainer pulls at most one mini-batch (35 groups) per update from the sample
#                                   queue, oldest first; the rest stays queued where the rollouter's staleness quota
#                                   counts it (async_training.replay_buffer.pull_cap; " pull-cap" in exp_name). With
#                                   staleness_threshold=1 the quota is 70 groups, so at most one update's worth of
#                                   groups waits in the queue and none is evicted unseen (replay/evicted_unseen_cum,
#                                   replay/queue_backlog, replay/pulled_groups).
#   * ROLLOUT_QUANT=fp8 + DeepGEMM  the vLLM replicas run verl's online block-FP8 weights (" rollout-fp8"); this
#                                   script execs the ..._5+3_..._fp8_tis-c8_token-mean.sh wrapper, which exports the
#                                   DeepGEMM environment (CUDA_HOME, DG_JIT_CACHE_DIR, VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER)
#                                   when a toolkit is found (DEEPGEMM_CUDA_HOME; remote_h200's activate.sh sets all three).
#   * KV_CACHE_DTYPE=fp8_e4m3       FP8 KV cache (" kv-fp8_e4m3"), half the KV bytes per token of bf16.
#   * VERL_QWEN2_K_BIAS_CENTER=1    vLLM caches Qwen2 keys as RoPE(k) - b_k (verl/utils/vllm/qwen2_k_bias_center.py);
#                                   REQUIRED with the FP8 KV cache on Qwen2.5-7B: its large k_proj bias (up to 414) is
#                                   rounded away by fp8_e4m3 and the rollout collapses (0 % accuracy). Exact for the
#                                   trainer (constant per query row, cancelled by softmax). See notes/qwen25_fp8_kv_cache_fix.md.
#   * actor_rollout_ref.rollout.max_num_seqs=512  (env ROLLOUT_MAX_NUM_SEQS; vLLM default 256), passed as the first
#                                   Hydra override so a trailing override still wins. With the quota of 70 groups at most
#                                   ~24 groups = 384 sequences run per engine, below the cap; the cap matters if
#                                   staleness_threshold is raised (quota 105 -> 35 groups = 560 sequences per engine).
#   * ROLLOUT_MAX_MODEL_LEN=10240   actor_rollout_ref.rollout.max_model_len (prompt 2048 + response 8192; untagged).
#   * VLLM_LOG_STATS=True           actor_rollout_ref.rollout.disable_log_stats=False: vLLM's periodic throughput /
#                                   KV-cache-usage lines in the rollouter log (untagged).
# Kept from the bf16 stale-1 arm: replay_tau=4, replay_staleness_threshold=16 (" tau-4 k-16"), staleness_threshold=1
# (" stale-1"), replay_min_fresh_ratio=0.9 (ceil(0.9 x 35) = 32 fresh groups per update, " fresh-0.9"),
# concurrency_ramp=null (no " ramp-" tag), lr_warmup_steps=0 (no " warmup-" tag), test_freq=save_freq=60,
# MODEL_PATH=Qwen/Qwen2.5-7B, n_gpus_rollout=3 (3 vLLM + 5 Megatron GPUs, tp1/dp5), train_prompt_mini_bsz=35,
# concurrent_samples_per_replica=35 (the base default = mini-batch), val_temperature=val_top_p=1.0; and through the
# FP8 TIS-C8 wrapper: rollout_is_threshold=8 (token TIS C=8), loss_agg_mode=token-mean, DYNAMIC_BSZ=True, H100
# emulation (VERL_GPU_MEM_CAP_GB=78, gpu_memory_utilization=0.5), SEED=1. ATTN_BACKEND stays default (no " attn-" tag).
# Every knob is an env default: e.g. replay_pull_cap=False or staleness_threshold=2 give the other variants back.
#
# exp_name reads
#   "GRPO-noVCPO replay tau-4 k-16 stale-1 rmb-0.5 nu-1 fresh-0.9 pull-cap ess-... h100-emu-78gb-gmu0.5 DAPO17K-AIME24
#    Qwen2.5-7B 3-5 tp1dp5 hdo B-35 token-mean 8192-len 0.1-wd dynbsz-10240 tis-C8 rollout-fp8 kv-fp8_e4m3 val-t1.0-p1.0 seed-1".
#
# Launch from the repo root exactly like the other arms, e.g. on remote_h200:
#   source <repo>/activate.sh
#   TRAIN_FILE=/data2/datasets/math_datasets/dapo/dapo-math-17k.parquet \
#   TEST_FILE="['/data2/datasets/math_datasets/dapo/aime-2024.parquet','/data2/datasets/math_datasets/dapo/aime-2025.parquet']" \
#   bash verl/experimental/fully_async_policy/shell/vcpo/replay_buffer/<this script> [trainer.val_before_train=False ...]
set -euo pipefail
export replay_tau=${replay_tau:-4}
export replay_staleness_threshold=${replay_staleness_threshold:-16}
export staleness_threshold=${staleness_threshold:-1}
export replay_min_fresh_ratio=${replay_min_fresh_ratio:-0.9}
export replay_pull_cap=${replay_pull_cap:-True}
export concurrency_ramp=${concurrency_ramp:-null}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
export model_tag=${model_tag:-"Qwen2.5-7B"}
export n_gpus_rollout=${n_gpus_rollout:-3}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-35}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
export VERL_QWEN2_K_BIAS_CENTER=${VERL_QWEN2_K_BIAS_CENTER:-1}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-10240}
export VLLM_LOG_STATS=${VLLM_LOG_STATS:-True}
export lr_warmup_steps=${lr_warmup_steps:-0}
export test_freq=${test_freq:-60}
export save_freq=${save_freq:-60}
export val_temperature=${val_temperature:-1.0}
export val_top_p=${val_top_p:-1.0}
# Default Hydra override first: a trailing actor_rollout_ref.rollout.max_num_seqs=... on the command line still wins.
exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_tis-c8_token-mean.sh" \
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" "$@"
