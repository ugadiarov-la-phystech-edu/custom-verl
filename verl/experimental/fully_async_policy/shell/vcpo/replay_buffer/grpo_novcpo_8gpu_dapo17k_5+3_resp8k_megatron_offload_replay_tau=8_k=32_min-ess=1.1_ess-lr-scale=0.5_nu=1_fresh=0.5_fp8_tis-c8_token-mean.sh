#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5-fp8-tisc8-tokenmean
#
# FP8-rollout replay / min-ESS arm with the sync TIS baseline's TIS cap and loss aggregation:
#   ..._nu=1_fresh=0.5_fp8.sh (read its header and the base arm's) plus
#   * rollout_is_threshold=8   (base arm: 2.0; as in examples/baselines/run_qwen3-8b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh)
#   * loss_agg_mode=token-mean (base arm: seq-mean-token-mean; as in the sync baselines)
# Both are env-overridable and visible in exp_name (" tis-C8", " token-mean").
#
# NOT THE SYNC BASELINE'S SEMANTICS. In the sync baseline the TIS weight is pi_old / pi_vllm with pi_old the
# generating parameters, so the cap of 8 bounds only the FP8/bf16 mismatch, and a PPO clip handles policy drift.
# Here the weight is pi_theta / pi_vllm against groups up to k=32 versions old, with no PPO clip: the cap of 8
# bounds staleness too, letting stale replayed groups carry up to 4x the weight the source arm allows (the min-ESS
# brake still applies, per mini-batch). See notes/fp8_mismatch_only_tis_in_replay.md.
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
#                                   minus ~1.5 GiB outside it (CUDA context, NCCL, cuBLAS).
# Both are tagged in exp_name (" h100-emu-78gb-gmu0.5"). The cap cannot bound vLLM: after each FP8 weight
# sync vLLM may hold a few GB over its budget, so sample nvidia-smi on the rollout GPUs; > ~80,000 MiB would not
# fit a real H100 (81,559 MiB).
# ON A REAL H100 OVERRIDE BOTH: VERL_GPU_MEM_CAP_GB= gpu_memory_utilization=0.88 bash <this script>
# (an empty VERL_GPU_MEM_CAP_GB disables the cap).
# SCHEDULE (differs from the bf16 arm): lr_warmup_steps=12 (linear, in optimizer updates; arm: 0),
# validation and hf_model checkpoints every 12 updates (test_freq=12, save_freq=12; arm: 25), SEED=1.
# All env-overridable; the warmup is tagged " warmup-12" in exp_name. The warmup must be shorter than lr_decay_steps
# (default total_rollout_steps = 66000; Megatron asserts it), not than max_updates.
set -euo pipefail
export rollout_is_threshold=${rollout_is_threshold:-8}
export loss_agg_mode=${loss_agg_mode:-token-mean}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-78}
export gpu_memory_utilization=${gpu_memory_utilization:-0.5}
export lr_warmup_steps=${lr_warmup_steps:-12}
export SEED=${SEED:-1}
export test_freq=${test_freq:-12}
export save_freq=${save_freq:-12}
exec bash "$(dirname "${BASH_SOURCE[0]}")/grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8.sh" "$@"
