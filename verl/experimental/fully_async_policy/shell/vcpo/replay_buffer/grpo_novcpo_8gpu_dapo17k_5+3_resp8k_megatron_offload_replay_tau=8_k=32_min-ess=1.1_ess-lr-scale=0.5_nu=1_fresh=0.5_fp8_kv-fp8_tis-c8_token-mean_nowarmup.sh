#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-kvfp8-tisc8-tokenmean-nowarmup
#
# FP8-rollout replay / min-ESS arm with the sync TIS baseline's TIS cap and loss aggregation, an FP8 KV cache,
# vLLM stats on and no LR warmup:
#   ..._nu=1_fresh=0.5_fp8.sh (read its header and the base arm's) plus
#   * rollout_is_threshold=8   (base arm: 2.0; as in examples/baselines/run_qwen3-8b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh)
#   * loss_agg_mode=token-mean (base arm: seq-mean-token-mean; as in the sync baselines)
#   * DYNAMIC_BSZ=True         (base arm: micro-batch 1; as in the sync TIS baseline): sequences are packed into
#                              micro-batches of up to DYNAMIC_BSZ_MAX_TOKENS=10240 tokens per GPU (one full-length
#                              sequence, so the per-pass memory peak stays at the micro-batch-1 worst case). The
#                              objective and the ESS brake are unchanged; see the base arm's "Dynamic batch size"
#                              block. DYNAMIC_BSZ=False restores micro-batch 1.
#   * KV_CACHE_DTYPE=fp8_e4m3  (base arm: auto = bf16): FP8 KV cache in the vLLM replicas
#                              (+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype), half the KV bytes per
#                              token. Qwen3-8B (the default model) has no QKV biases and needs nothing else; a Qwen2.5
#                              model would also need VERL_QWEN2_K_BIAS_CENTER=1 (see notes/qwen25_fp8_kv_cache_fix.md).
#   * VLLM_LOG_STATS=True      (base arm: False): actor_rollout_ref.rollout.disable_log_stats=False, i.e. vLLM's
#                              periodic throughput / KV-cache-usage lines in the rollouter log.
# All are env-overridable; visible in exp_name: " tis-C8", " token-mean", " dynbsz-10240", " kv-fp8_e4m3" (vLLM stats
# are untagged). ATTN_BACKEND stays default and ROLLOUT_MAX_MODEL_LEN null (vLLM takes the model's context length).
#
# NOT THE SYNC BASELINE'S SEMANTICS. In the sync baseline the TIS weight is pi_old / pi_vllm with pi_old the
# generating parameters, so the cap of 8 bounds only the FP8/bf16 mismatch, and a PPO clip handles policy drift.
# Here the weight is pi_theta / pi_vllm against groups up to k=32 versions old, with no PPO clip: the cap of 8
# bounds staleness too, letting stale replayed groups carry up to 4x the weight the source arm allows (the min-ESS
# brake still applies, per mini-batch). See notes/fp8_mismatch_only_tis_in_replay.md. The FP8 KV cache adds to the
# rollout/trainer mismatch the cap has to absorb; watch actor/rollout_corr/kl and the Pearson correlation.
#
# token-mean is NOT the source arm's objective: the custom_vcpo per-trajectory update corresponds to
# seq-mean-token-mean (tests/workers/utils/test_ppo_loss_ess_on_cpu.py); token-mean weights long responses more.
#
# Launch from the repo root exactly like ..._fp8.sh (same data-path variables, same DeepGEMM handling).
# H100 EMULATION IS ON BY DEFAULT (for remote_h200's H200s, 139.8 GiB usable each). In this layout every GPU
# holds one role, so each is emulated on its own:
#   * gpu_memory_utilization=0.5    rollout GPUs: vLLM gets 0.5 x 139.8 = 69.9 GiB, i.e. 0.88 of an H100 (79.65 GiB).
#                                   Not the arm's 0.9 (0.513 here, 71.7 GiB): vLLM's budget excludes the NCCL
#                                   weight sync's 2 x 2048 MB receive buffers and the FP8 re-quantization
#                                   temporaries, and at 0.513 the 5+3 smoke (Qwen3-4B) peaked at 83,263 MiB on
#                                   every rollout GPU, over an H100's 81,559.
#   * VERL_GPU_MEM_CAP_GB=78        trainer GPUs: caps the Megatron trainer's allocator at a whole H100 (79.65 GiB)
#                                   minus ~1.5 GiB outside it (CUDA context, NCCL, cuBLAS); the standalone vLLM
#                                   workers are capped at the same 78 GiB (verl/utils/gpu_memory_cap.py).
# Both are tagged in exp_name (" h100-emu-78gb-gmu0.5"). Sample nvidia-smi on the rollout GPUs; > ~80,000 MiB would
# not fit a real H100 (81,559 MiB).
# ON A REAL H100 OVERRIDE BOTH: VERL_GPU_MEM_CAP_GB= gpu_memory_utilization=0.88 bash <this script>
# (an empty VERL_GPU_MEM_CAP_GB disables the cap).
# SCHEDULE: lr_warmup_steps=0 (constant LR from the first optimizer update; the ..._fp8_tis-c8_token-mean.sh arm
# uses a 12-update warmup, " warmup-12"; no " warmup-" tag here), validation and hf_model checkpoints every 12
# updates (test_freq=12, save_freq=12; arm: 25), SEED=1. All env-overridable.
# Every default of ..._fp8.sh is also set here explicitly (ROLLOUT_QUANT, DeepGEMM, emulation, schedule), so this
# file shows the arm's full configuration; ..._fp8.sh keeps any value already set.
set -euo pipefail
export ROLLOUT_QUANT=${ROLLOUT_QUANT:-fp8}
export rollout_is_threshold=${rollout_is_threshold:-8}
export loss_agg_mode=${loss_agg_mode:-token-mean}
export DYNAMIC_BSZ=${DYNAMIC_BSZ:-True}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
export VLLM_LOG_STATS=${VLLM_LOG_STATS:-True}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-78}
export gpu_memory_utilization=${gpu_memory_utilization:-0.5}
export lr_warmup_steps=${lr_warmup_steps:-0}
export SEED=${SEED:-1}
export test_freq=${test_freq:-12}
export save_freq=${save_freq:-12}
DEEPGEMM_CUDA_HOME=${DEEPGEMM_CUDA_HOME:-/home/jovyan/ugadiarov/cuda-12.9}
if [[ -z "${CUDA_HOME:-}" && -x "${DEEPGEMM_CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_HOME="${DEEPGEMM_CUDA_HOME}"
    export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/home/jovyan/ugadiarov/cache/deep_gemm}
    export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}
fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8.sh" "$@"
