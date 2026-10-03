#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-tisc8-tokenmean-qwen2.5-7b-3+5
#
# The FP8 TIS-C8 token-mean replay / min-ESS arm (..._fp8_tis-c8_token-mean.sh, same parameters below) with
# Qwen/Qwen2.5-7B (BASE) on a 3+5 layout:
#   * MODEL_PATH=Qwen/Qwen2.5-7B, model_tag=Qwen2.5-7B (" Qwen2.5-7B " in exp_name; Qwen3-8B arms: " Qwen3-8B ")
#   * n_gpus_rollout=3             3 vLLM rollout GPUs + 5 Megatron trainer GPUs (tp1/dp5, " 3-5 tp1dp5 ");
#                                  the source layout is 5+3 (tp1/dp3)
#   * train_prompt_mini_bsz=35     35 x 16 = 560 sequences per update, 112 per trainer rank (must divide by dp=5;
#                                  33 x 16 = 528 does not). " B-35 " in exp_name
#   * test_freq=20, save_freq=20   validation and hf_model checkpoints every 20 updates (TIS-C8 arm: 12)
# All env-overridable. Everything else - ROLLOUT_QUANT=fp8, rollout_is_threshold=8, loss_agg_mode=token-mean,
# DYNAMIC_BSZ=True, H100 emulation (VERL_GPU_MEM_CAP_GB=78, gpu_memory_utilization=0.5), lr_warmup_steps=12,
# SEED=1, DeepGEMM - is the TIS-C8 arm's; read its header, ..._fp8.sh's and the base arm's.
#
# DERIVED FROM THE MINI-BATCH AND LAYOUT (the base arm computes them; no extra knobs):
#   * first update (requires_mini_batches=0.5): 0.5 x 35 = 17.5 -> 18, raised to 20 so that 20 x 16 = 320
#     sequences split over the 5 trainer ranks; full 35-group mini-batches afterwards;
#   * fresh-share gate: ceil(0.5 x 35) = 18 newly arrived groups per update;
#   * in-flight cap: concurrent_samples_per_replica = 35 groups per engine, 3 x 35 = 105 (5+3 arm: 5 x 33 = 165);
#   * concurrency ramp [5, 12, 20] per engine -> 15 / 36 / 60 groups in flight until the 1st / 2nd / 3rd
#     mini-batch is delivered (5+3 arm: 25 / 60 / 100).
# With 3 engines generation is ~40% slower than on 5 while the update is spread over 5 ranks, so expect more
# trainer idle time behind the fresh-share gate (replay/fresh_wait_s) than in the 5+3 arm.
#
# MODEL. Qwen2ForCausalLM, 7.6B parameters, 28 layers, 28 query / 4 KV heads x 128, untied lm_head, QKV BIASES -
# Megatron-Bridge's Qwen2Bridge maps them, no external_lib needed. Its KV cache per token is 57,344 B (Qwen3-8B:
# 147,456 B), so the same vLLM budget holds ~2.5x more tokens; the emulation values, derived for Qwen3-8B, are
# conservative here.
#
# PROMPTS. Same DAPO user message; Qwen2.5's chat template adds its default system prompt ("You are a helpful
# assistant.") and there is no thinking mode. The base model stops only at <|endoftext|> (not <|im_end|>).
# Expect a low initial reward and short responses that grow during RL ("zero RL", as in FlashRL's Qwen2.5-32B
# base runs).
#
# Launch from the repo root exactly like ..._fp8_tis-c8_token-mean.sh.
set -euo pipefail
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
export model_tag=${model_tag:-"Qwen2.5-7B"}
export n_gpus_rollout=${n_gpus_rollout:-3}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-35}
export ROLLOUT_QUANT=${ROLLOUT_QUANT:-fp8}
export rollout_is_threshold=${rollout_is_threshold:-8}
export loss_agg_mode=${loss_agg_mode:-token-mean}
export DYNAMIC_BSZ=${DYNAMIC_BSZ:-True}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-78}
export gpu_memory_utilization=${gpu_memory_utilization:-0.5}
export lr_warmup_steps=${lr_warmup_steps:-12}
export SEED=${SEED:-1}
export test_freq=${test_freq:-20}
export save_freq=${save_freq:-20}
DEEPGEMM_CUDA_HOME=${DEEPGEMM_CUDA_HOME:-/home/jovyan/ugadiarov/cuda-12.9}
if [[ -z "${CUDA_HOME:-}" && -x "${DEEPGEMM_CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_HOME="${DEEPGEMM_CUDA_HOME}"
    export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/home/jovyan/ugadiarov/cache/deep_gemm}
    export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}
fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8.sh" "$@"
