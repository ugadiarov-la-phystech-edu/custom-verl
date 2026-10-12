#!/usr/bin/env bash
# Replay arm (tau=8, k=32, min-ess=1.1, ess-lr-scale=0.5, nu=1, fresh=0.5) with FP8 rollout, TIS C=8 and token-mean
# loss -- the 5+3 `..._fp8_tis-c8_token-mean.sh` arm -- re-laid out as 4+4 with an FP8 KV cache.
#
# Differences from ..._5+3_..._fp8_tis-c8_token-mean.sh (every knob below is an env default; override as usual):
#   * layout 4+4: n_gpus_rollout=4, so the trainer runs on 4 GPUs (tp1dp4) instead of 3. With FP8 KV the 5 rollout
#     replicas of the 5+3 arm delivered ~47 kept groups per update against the 33 the trainer consumed (trainer-bound,
#     rollouter never idle), so one rollout GPU is moved to the trainer.
#   * train_prompt_mini_bsz=32 (32 x 16 = 512 sequences per update, divisible by trainer DP 4). The per-engine in-flight
#     cap concurrent_samples_per_replica follows it (base launcher default), i.e. 4 x 32 = 128 groups in flight.
#   * rollout engine: KV_CACHE_DTYPE=fp8_e4m3 (FlashAttention 3), ATTN_BACKEND=default, ROLLOUT_MAX_MODEL_LEN=10240
#     (2048 prompt + 8192 response), VLLM_LOG_STATS=True (vLLM stats on; the rollouter logs vllm/* metrics).
#   * no LR warmup: lr_warmup_steps=0.
#   * DYNAMIC_BSZ_MAX_TOKENS=16384: per-GPU token cap of the training micro-batches (default 2048+8192=10240), i.e.
#     fewer, larger micro-batches in update_actor. In the 5+3 FP8-KV run the trainer peaked at ~59 GB allocated under
#     the 78 GB cap; +6K tokens adds an estimated ~9 GB (mostly the full-vocab LM-head temporaries), so watch
#     perf/max_memory_allocated_gb. The log-prob cap follows it by default but is unused here (bypass mode).
#   * validation on both AIME sets: TEST_FILE defaults to AIME-2024 + AIME-2025 (the base launcher's default, set
#     explicitly here). On remote_h200 pass TRAIN_FILE/TEST_FILE with the /data2/datasets/... paths.
set -euo pipefail

export n_gpus_rollout=${n_gpus_rollout:-4}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-32}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
export ATTN_BACKEND=${ATTN_BACKEND:-default}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-10240}
export VLLM_LOG_STATS=${VLLM_LOG_STATS:-True}
export lr_warmup_steps=${lr_warmup_steps:-0}
export DYNAMIC_BSZ_MAX_TOKENS=${DYNAMIC_BSZ_MAX_TOKENS:-16384}
export TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet','/home/jovyan/datasets/math_datasets/dapo/aime-2025.parquet']"}

exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_tis-c8_token-mean.sh" "$@"
