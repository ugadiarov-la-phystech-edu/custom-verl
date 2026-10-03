#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-dapo17k-grpo-qwen2.5-7b-tis

# Qwen2.5-7B (BASE) synchronous baseline WITH truncated importance sampling (TIS), in three rollout precisions:
#
#   bash examples/baselines/run_qwen2.5-7b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh                     # bf16 + TIS
#   ROLLOUT_QUANT=fp8 bash examples/baselines/run_qwen2.5-7b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh   # fp8 + TIS
#   ROLLOUT_QUANT=int8 bash examples/baselines/run_qwen2.5-7b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh  # int8 + TIS
#
# The same arm as run_qwen3-8b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh - identical parameters below
# except BATCH and CADENCE - with the model swapped to Qwen/Qwen2.5-7B (base). FlashRL's large-scale result trains Qwen2.5-32B BASE with
# the DAPO recipe ("zero RL": no thinking mode, the long chain of thought emerges during RL); this is that
# setting at 7B, for comparison with the Qwen3-8B arm, which starts from a post-trained thinking model.
#
# FLASHRL PARAMETERS. Four knobs follow FlashRL's DAPO-Qwen2.5-32B recipe (yaof20/verl, branch
# flash-rl, recipe/flash_rl/dapo_qwen32b_{bf16,int8}.sh), overriding the baseline, plus a shorter warmup:
#   * TIS cap C            TIS_THRESHOLD=8        (baseline arm: TIS off; verl's FP8 guide uses 2)
#   * clip ratio low/high  0.2 / 0.28 (DAPO clip-higher; baseline 0.2 / 0.2)
#   * dual-clip c          10.0                   (baseline 3.0)
#   * LR warmup            3 steps, linear, then constant (baseline 0; FlashRL: 10). Both FlashRL's FSDP
#                          worker and v0.9.0's engine step the scheduler once per ROLLOUT step, so this is
#                          3 rollout steps (= 12 optimizer updates here).
#   * weight decay         0.1                    (baseline 0.01)
# All five are env-overridable and tagged in exp_name.
#
# BATCH. 140 prompts x 16 responses per rollout step (train_prompt_bsz=140; Qwen3-8B TIS script: 128), split into
# mini-batches of 35 prompts (train_prompt_mini_bsz=35; there: 32), i.e. 560 sequences, 70 per trainer rank: the
# mini-batch of the Qwen2.5-7B 3+5 replay arm. Still 4 optimizer updates per rollout step, so the step-based
# warmup and cadence below mean the same number of updates. Both env-overridable and in exp_name
# (" B-140xn16 mini-35 "); the file name keeps the Qwen3-8B script's B128xn16_mini32.
#
# CADENCE. Validation and checkpoints every 5 rollout steps (= 20 optimizer updates; baseline 2):
# test_freq=5, save_freq=5, env-overridable (the Qwen3-8B TIS script: 3).
#
# H100 EMULATION IS ON BY DEFAULT (for runs on remote_h200's 140.4 GiB H200s; see the baseline's
# "H100 emulation" note and verl/utils/gpu_memory_cap.py):
#   * VERL_GPU_MEM_CAP_GB=76      caps the trainer (actor-role worker) allocator at 76 GiB.
#   * gpu_memory_utilization=0.283 gives vLLM the H100's absolute budget, 0.5 x 79.65 / 140.4 GiB.
# Both are tagged in exp_name (" h100-emu-76gb-gmu0.283"). Both values were derived for Qwen3-8B; Qwen2.5-7B is
# smaller in every term (see MODEL below), so they are conservative. Watch per-GPU memory with an nvidia-smi
# sampler; > ~78 GB would OOM an H100.
# ON A REAL H100 OVERRIDE BOTH: VERL_GPU_MEM_CAP_GB= gpu_memory_utilization=0.5 bash <this script>
# (an empty VERL_GPU_MEM_CAP_GB disables the cap).
#
# DYNAMIC BATCH SIZE is on here (DYNAMIC_BSZ=True, cap DYNAMIC_BSZ_MAX_TOKENS=10240 tokens per GPU per
# micro-batch = one full-length sequence). It packs short sequences into shared passes for old_log_prob /
# update_actor without changing the loss (normalized per mini-batch); see the baseline's "Dynamic batch size"
# block. DYNAMIC_BSZ=False restores micro-batch 1. The rest of FlashRL's recipe is NOT adopted:
# 512-prompt batches, 20k responses + DAPO overlong penalty, dynamic sampling (filter_groups),
# FSDP + SP8, rollout TP2, val T=1.0 - see the baseline header for this arm's settings.
#
# Everything else - data, geometry, loss aggregation, optimizer, seeds, checkpoints - is
# run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh (the shared baseline launcher), which this script
# runs with TIS=True, MODEL_PATH=Qwen/Qwen2.5-7B and model_tag=Qwen2.5-7B (" Qwen2.5-7B " in exp_name and the
# log / checkpoint directory) plus the BATCH and CADENCE values above. TIS changes training even in bf16 and can
# depress the LOGGED training reward, which is sampled from vLLM, while improving validation - judge by val-core/*.
#
# TIS: per-token PPO loss x min(pi_trainer_old / pi_vllm, C), C = TIS_THRESHOLD (default 8, FlashRL's
# DAPO-32B value; this is exactly FlashRL's imp_ratio_cap formula, applied after the PPO clip).
# The weights come from the rollout log-probs vLLM already returns (processed_logprobs, i.e. after
# temperature / top-p); old_log_probs are still recomputed by the trainer (bypass_mode=false).
# Watch rollout_corr/* (IS-weight tails, KL, pearson) alongside actor/entropy.
#
# MODEL. Qwen2ForCausalLM, 7.6B parameters, 28 layers, 28 query / 4 KV heads x 128, vocab 152,064, untied
# lm_head, QKV BIASES - Megatron-Bridge's Qwen2Bridge maps them (add_qkv_bias=True), no external_lib needed.
# Weights 15.2 GB (Qwen3-8B: 16.4 GB); KV 28 layers x 4 heads x 128 x 2 (K, V) x 2 B = 57,344 B per token
# (Qwen3-8B: 147,456 B), so the same vLLM budget holds ~2.5x more KV tokens.
#
# PROMPTS. The same DAPO user message, but Qwen2.5's chat template adds its default system prompt
# ("You are a helpful assistant.") and there is no thinking mode (no <think> tokens): the model answers in plain
# step-by-step text, scored by the same math_dapo scorer on the last "Answer:" line. The base model's
# generation_config stops only at <|endoftext|>; the ChatML <|im_end|> is not a stop token (as in FlashRL's
# Qwen2.5-32B base runs). A Qwen2.5-0.5B base probe on these prompts never emitted <|im_end|> (58/64 stopped at
# <|endoftext|>, 6/64 hit the cap). Expect a low initial reward and short responses that grow during RL.
#
# DEEPGEMM (FP8 rollout kernels). vLLM runs block-FP8 linears through DeepGEMM, which JIT-compiles its
# kernels and therefore needs a CUDA toolkit (CUDA_HOME). When CUDA_HOME is unset and
# DEEPGEMM_CUDA_HOME=/home/jovyan/ugadiarov/cuda-12.9 exists, this script exports:
#   CUDA_HOME=$DEEPGEMM_CUDA_HOME
#   DG_JIT_CACHE_DIR=/home/jovyan/ugadiarov/cache/deep_gemm   compiled kernels on persistent storage
#   VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0   no FlashInfer small-batch block-FP8 GEMM JIT (needs headers the
#       userspace toolkit lacks); DeepGEMM serves every batch size.
# Each is left alone if already set, and nothing is exported where the toolkit is absent. bf16 and int8
# rollouts do not use DeepGEMM.
#
# Any env knob or trailing Hydra override of the base script still applies, e.g.
#   TIS_THRESHOLD=2 max_updates=200 ROLLOUT_QUANT=fp8 bash <this script>

set -euo pipefail
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
export model_tag=${model_tag:-"Qwen2.5-7B"}
export train_prompt_bsz=${train_prompt_bsz:-140}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-35}
export TIS=${TIS:-True}
export TIS_THRESHOLD=${TIS_THRESHOLD:-8}
export clip_ratio_low=${clip_ratio_low:-0.2}
export clip_ratio_high=${clip_ratio_high:-0.28}
export clip_ratio_c=${clip_ratio_c:-10.0}
export lr_warmup_steps=${lr_warmup_steps:-3}
export test_freq=${test_freq:-5}
export save_freq=${save_freq:-5}
export weight_decay=${weight_decay:-0.1}
export DYNAMIC_BSZ=${DYNAMIC_BSZ:-True}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-76}
export gpu_memory_utilization=${gpu_memory_utilization:-0.283}

DEEPGEMM_CUDA_HOME=${DEEPGEMM_CUDA_HOME:-/home/jovyan/ugadiarov/cuda-12.9}
if [[ -z "${CUDA_HOME:-}" && -x "${DEEPGEMM_CUDA_HOME}/bin/nvcc" ]]; then
    export CUDA_HOME="${DEEPGEMM_CUDA_HOME}"
    export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/home/jovyan/ugadiarov/cache/deep_gemm}
    export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}
fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh" "$@"
