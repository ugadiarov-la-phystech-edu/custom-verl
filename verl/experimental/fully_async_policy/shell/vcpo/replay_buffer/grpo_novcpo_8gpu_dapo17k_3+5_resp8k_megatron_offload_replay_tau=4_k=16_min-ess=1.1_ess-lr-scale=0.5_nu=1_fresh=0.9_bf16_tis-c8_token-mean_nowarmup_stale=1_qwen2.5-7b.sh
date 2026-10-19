#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-tau4-k16-stale1-ess-nu1-fresh0.9-bf16-tisc8-tokenmean-nowarmup-qwen2.5-7b-3+5
#
# BF16-ROLLOUT, NO-WARMUP, TIGHTER-STALENESS variant of the Qwen2.5-7B (BASE) 3+5 FP8 TIS-C8 token-mean arm
#   ..._replay_tau=8_k=32_..._fresh=0.5_fp8_tis-c8_token-mean_qwen2.5-7b.sh (read its header; same layout, batches
#   and cadence). The ONLY differences:
#   * replay_tau=4                replay draw temperature over staleness (arm: 8; " tau-4" in exp_name)
#   * replay_staleness_threshold=16  groups older than 16 updates are evicted from the replay buffer (arm: 32;
#                                 " k-16")
#   * staleness_threshold=1       rollouter generation quota: at most required_samples x (1 + 1) = 70 groups
#                                 generated ahead of the trainer per sync (arm: 32 -> 1155), so the in-flight
#                                 cap becomes min(3 x 35, 70) = 70 groups and the sample queue holds 70; the
#                                 rollouter idles once it is a full update ahead instead of filling the buffer
#                                 with stale groups. Tagged " stale-1" (the base arm tags any value != 32.0).
#   * replay_min_fresh_ratio=0.9  fresh-share gate: ceil(0.9 x 35) = 32 newly arrived groups per update (arm: 0.5 ->
#                                 18; " fresh-0.9"). In the fresh-0.5 run of this script (2026-10-19) ~27 fresh groups
#                                 arrived per 25 s update under the 70-group quota, so expect the trainer to wait for
#                                 the gate on most updates (replay/fresh_wait_s > 0, trainer idle ~10-20%) and the
#                                 fresh ratio to sit at 0.9; raise staleness_threshold to 2 (quota 105 = the full
#                                 3 x 35 concurrency) if the wait is too long.
#   * test_freq=60, save_freq=60  validation and hf_model checkpoints every 60 updates (arm: 20).
#   * concurrency_ramp=null       no warm-up stages (arm: [5, 12, 20] per engine = 15 / 36 / 60 groups in flight
#                                 until 20 / 55 / 90 groups are delivered): the staleness quota of 70 already bounds
#                                 what is generated on the initial weights, and in the fresh-0.5 run the ramp only
#                                 made the first update take 80 s (two generation waves for the 20-group first
#                                 batch) and changed nothing afterwards. No " ramp-…" tag in exp_name.
#   * ROLLOUT_QUANT=bf16      the vLLM rollout replicas run the bf16 checkpoint as-is (actor_rollout_ref.rollout.
#                             quantization=null, no FP8 re-quantization at the weight sync, no DeepGEMM), so no
#                             " rollout-fp8" tag and no CUDA_HOME / DG_JIT_CACHE_DIR / VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER
#                             exports. The rollout/trainer mismatch is then only the vLLM-vs-Megatron bf16 numerics
#                             (training/rollout_actor_probs_pearson_corr ~ 0.999 in the Qwen2.5-7B sync bf16 baselines).
#   * lr_warmup_steps=0       constant LR from the first optimizer update (FP8 arm: linear warmup over 12 updates,
#                             " warmup-12"); same as the base arm's schedule and the Qwen2.5-7B "nowarmup" sync baselines.
# Everything else is the FP8 arm's: MODEL_PATH=Qwen/Qwen2.5-7B, n_gpus_rollout=3 (3 vLLM + 5 Megatron GPUs, tp1/dp5),
# train_prompt_mini_bsz=35, rollout_is_threshold=8 (token TIS C=8), loss_agg_mode=token-mean, DYNAMIC_BSZ=True,
# H100 emulation (VERL_GPU_MEM_CAP_GB=78 on the trainer GPUs, gpu_memory_utilization=0.5 on the rollout GPUs:
# bf16 vLLM holds a 15.2 GiB model instead of 7.6 GiB FP8, so the 69.9 GiB budget leaves ~2.5x fewer KV tokens
# than the FP8 arm - raise gpu_memory_utilization if the KV cache is short), SEED=1,
# val_temperature=val_top_p=1.0. All env-overridable: ROLLOUT_QUANT=fp8 or lr_warmup_steps=12 give the FP8 arm's
# settings back (but without the DeepGEMM environment; use the FP8 arm for that).
#
# This script execs the base arm directly (not the ..._fp8.sh wrapper), so exp_name reads
#   "GRPO-noVCPO replay tau-4 k-16 stale-1 rmb-0.5 nu-1 fresh-0.9 ess-... h100-emu-78gb-gmu0.5
#    DAPO17K-AIME24 Qwen2.5-7B 3-5 tp1dp5 hdo B-35 token-mean 8192-len 0.1-wd dynbsz-10240 tis-C8 val-t1.0-p1.0 seed-1".
#
# Launch from the repo root exactly like the FP8 arm, e.g. on remote_h200:
#   source <repo>/activate.sh
#   TRAIN_FILE=/data2/datasets/math_datasets/dapo/dapo-math-17k.parquet \
#   TEST_FILE="['/data2/datasets/math_datasets/dapo/aime-2024.parquet','/data2/datasets/math_datasets/dapo/aime-2025.parquet']" \
#   bash verl/experimental/fully_async_policy/shell/vcpo/replay_buffer/<this script> [trainer.val_before_train=False ...]
set -euo pipefail
export replay_tau=${replay_tau:-4}
export replay_staleness_threshold=${replay_staleness_threshold:-16}
export staleness_threshold=${staleness_threshold:-1}
export replay_min_fresh_ratio=${replay_min_fresh_ratio:-0.9}
export concurrency_ramp=${concurrency_ramp:-null}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
export model_tag=${model_tag:-"Qwen2.5-7B"}
export n_gpus_rollout=${n_gpus_rollout:-3}
export train_prompt_mini_bsz=${train_prompt_mini_bsz:-35}
export ROLLOUT_QUANT=${ROLLOUT_QUANT:-bf16}
export rollout_is_threshold=${rollout_is_threshold:-8}
export loss_agg_mode=${loss_agg_mode:-token-mean}
export DYNAMIC_BSZ=${DYNAMIC_BSZ:-True}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-78}
export gpu_memory_utilization=${gpu_memory_utilization:-0.5}
export lr_warmup_steps=${lr_warmup_steps:-0}
export SEED=${SEED:-1}
export test_freq=${test_freq:-60}
export save_freq=${save_freq:-60}
export val_temperature=${val_temperature:-1.0}
export val_top_p=${val_top_p:-1.0}
exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5.sh" "$@"
