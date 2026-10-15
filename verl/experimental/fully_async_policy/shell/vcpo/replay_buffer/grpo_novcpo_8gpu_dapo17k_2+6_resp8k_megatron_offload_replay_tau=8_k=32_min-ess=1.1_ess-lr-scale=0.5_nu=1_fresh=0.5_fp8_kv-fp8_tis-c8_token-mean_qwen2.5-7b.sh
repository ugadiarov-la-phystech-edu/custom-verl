#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-kvfp8-tisc8-tokenmean-qwen2.5-7b-2+6
#
# The 4+4 FP8-KV replay arm (..._4+4_..._fp8_kv-fp8_tis-c8_token-mean.sh, same parameters below) with
# Qwen/Qwen2.5-7B (BASE) on a 2+6 layout.
#
# Differences from ..._4+4_..._fp8_kv-fp8_tis-c8_token-mean.sh (every knob below is an env default; override as usual):
#   * MODEL_PATH=Qwen/Qwen2.5-7B, model_tag=Qwen2.5-7B (" Qwen2.5-7B " in exp_name). Model notes (QKV biases, KV
#     cache 57,344 B/token in bf16 - 28,672 B in FP8, base model stopping at <|endoftext|>, "zero RL" start) are in
#     ..._3+5_..._fp8_tis-c8_token-mean_qwen2.5-7b.sh.
#   * layout 2+6: n_gpus_rollout=2, so 2 vLLM rollout GPUs and 6 Megatron trainer GPUs (tp1dp6).
#   * train_prompt_mini_bsz=36 (36 x 16 = 576 sequences per update, 96 per trainer rank; mini x 16 must divide by
#     dp=6, which 32 x 16 = 512 does not). As in the Qwen2.5-7B sync runs (b144 / mini36).
#   * DYNAMIC_BSZ_MAX_TOKENS=10240 (4+4 arm: 16384): per-GPU token cap of the training micro-batches, back to the
#     base launcher's default 2048 + 8192 (the minimum: one full-length sequence).
#   * test_freq=40, save_freq=40 (4+4 arm: 12): validation and hf_model checkpoints every 40 updates.
#   * VERL_QWEN2_K_BIAS_CENTER=1: vLLM caches Qwen2 keys as RoPE(k) - b_k (verl/utils/vllm/qwen2_k_bias_center.py).
#     Without it Qwen2.5-7B's large k_proj bias (up to 414, in the lowest-frequency RoPE pairs) is rounded away by
#     the fp8_e4m3 KV cache and the rollout collapses (0% accuracy, ~30% truncation, every group all-wrong). The
#     subtraction is exact (constant per query row, cancelled by softmax), so the trainer is unaffected. Set it to 0
#     to reproduce the broken baseline. See notes/qwen25_fp8_kv_cache_fix.md.
#
# DERIVED FROM THE MINI-BATCH AND LAYOUT (the base arm computes them; no extra knobs):
#   * first update (requires_mini_batches=0.5): 18 groups, 18 x 16 = 288 sequences over 6 ranks;
#   * fresh-share gate: ceil(0.5 x 36) = 18 newly arrived groups per update;
#   * in-flight cap: concurrent_samples_per_replica = 48 groups per engine (set below, decoupled from the
#     mini-batch), 2 x 48 = 96 groups = 768 sequences per engine (4+4 arm: 4 x 32 = 128);
#   * concurrency ramp [5, 12, 20] per engine -> 10 / 24 / 40 groups in flight until the 1st / 2nd / 3rd
#     mini-batch is delivered.
# With 2 engines the run is generation-bound: expect trainer idle time behind the fresh-share gate
# (replay/fresh_wait_s). Qwen2.5-7B's FP8 KV cache needs ~2.5x less memory per token than Qwen3-8B's, so each
# engine has room for more than 36 groups in flight, hence concurrent_samples_per_replica=48: 768 sequences per
# engine, inside the 512-1,024 range where a vLLM engine stays GPU-bound (at 1,536 the single-threaded EngineCore
# loop became the bottleneck in scripts/group_outcome_stats.py runs). The FP8 KV budget at gmu 0.5 is roughly 2M
# tokens per engine (estimate), so 768 x ~2.2k tokens late in training is tight: if vllm/* shows preemptions or
# KV-cache usage near 100%, lower it to 36-42; if KV usage stays low and replay/fresh_wait_s is high, try 54-60.
#   * actor_rollout_ref.rollout.max_num_seqs=768 (= 48 x 16; env ROLLOUT_MAX_NUM_SEQS, vLLM default 256): the
#     per-engine running cap must match the in-flight groups, or the extra groups only queue in vLLM (measured with
#     the default: "Running: 254 reqs, Waiting: 215 reqs" at ~5% KV-cache usage). max_num_batched_tokens stays
#     10,240 (768 decode tokens + ~9.5k for prefill). CUDA graphs are captured up to min(2 x max_num_seqs, 512), so
#     steps with 513-768 running sequences run without full graphs. If rollout GPU utilization drops while the
#     EngineCore process sits near 100% CPU, use 512 together with concurrent_samples_per_replica=32.
#
# Kept from the 4+4 arm: KV_CACHE_DTYPE=fp8_e4m3, ATTN_BACKEND=default, ROLLOUT_MAX_MODEL_LEN=10240,
# VLLM_LOG_STATS=True, lr_warmup_steps=0, AIME-2024 + AIME-2025 validation, and,
# through ..._5+3_..._fp8_tis-c8_token-mean.sh, ROLLOUT_QUANT=fp8, TIS C=8, token-mean, DYNAMIC_BSZ=True, H100
# emulation (VERL_GPU_MEM_CAP_GB=78, gpu_memory_utilization=0.5), SEED=1, DeepGEMM.
# On remote_h200 pass TRAIN_FILE/TEST_FILE with the /data2/datasets/... paths.
set -euo pipefail

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
export model_tag=${model_tag:-"Qwen2.5-7B"}
export n_gpus_rollout=${n_gpus_rollout:-2}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-36}
export concurrent_samples_per_replica=${concurrent_samples_per_replica:-48}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-768}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
export ATTN_BACKEND=${ATTN_BACKEND:-default}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-10240}
export VLLM_LOG_STATS=${VLLM_LOG_STATS:-True}
export lr_warmup_steps=${lr_warmup_steps:-0}
export DYNAMIC_BSZ_MAX_TOKENS=${DYNAMIC_BSZ_MAX_TOKENS:-10240}
export test_freq=${test_freq:-40}
export save_freq=${save_freq:-40}
export VERL_QWEN2_K_BIAS_CENTER=${VERL_QWEN2_K_BIAS_CENTER:-1}
export TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet','/home/jovyan/datasets/math_datasets/dapo/aime-2025.parquet']"}

# Default Hydra override first: a trailing actor_rollout_ref.rollout.max_num_seqs=... on the command line still wins.
exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_tis-c8_token-mean.sh" \
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" "$@"
