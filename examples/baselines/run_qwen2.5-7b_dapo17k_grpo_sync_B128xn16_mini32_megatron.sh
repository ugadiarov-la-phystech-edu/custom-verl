#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-dapo17k-grpo-qwen2.5-7b
# Qwen2.5-7B (BASE) synchronous GRPO baseline: run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh (the shared
# baseline launcher) with the model swapped to Qwen/Qwen2.5-7B. Every other parameter - data, batch 128 x 16,
# mini-batch 32, loss, optimizer, rollout, seeds - is the baseline's default.
#
#   bash examples/baselines/run_qwen2.5-7b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh
#
# Differences from the baseline launcher (all env-overridable):
#   * MODEL      MODEL_PATH=Qwen/Qwen2.5-7B, model_tag=Qwen2.5-7B (" Qwen2.5-7B " in exp_name and the log /
#                checkpoint directory). Qwen2ForCausalLM, 28 layers, 28 query / 4 KV heads x 128, QKV biases; see
#                run_qwen2.5-7b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh for the model and prompt notes.
#   * CADENCE    validation and checkpoints every 6 rollout steps (= 24 optimizer updates): test_freq=6, save_freq=6
#                (baseline 2).
#   * VALIDATION SAMPLING  val_temperature=1.0, val_top_p=1.0 (baseline: 0.8 / 0.7): plain sampling at Hugging
#                Face's default temperature and top_p (Qwen2.5-7B base's generation_config.json sets neither; it
#                defaults to greedy, do_sample=false, which verl 0.9 ignores). The 32 repeats of each AIME problem
#                are independent samples, so val-core/*/acc/mean@1 is avg@32.
#   * H100 EMULATION  on by default, as in run_qwen2.5-7b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh (for runs on
#                remote_h200's 140.4 GiB H200s; see the baseline's "H100 emulation" note and verl/utils/gpu_memory_cap.py):
#                VERL_GPU_MEM_CAP_GB=76 caps the trainer allocator at 76 GiB, and gpu_memory_utilization=0.283 gives vLLM
#                the H100's absolute budget (0.5 x 79.65 / 140.4 GiB). Both are tagged in exp_name
#                (" h100-emu-76gb-gmu0.283"). On a real H100 override both:
#                VERL_GPU_MEM_CAP_GB= gpu_memory_utilization=0.5 bash <this script> (an empty value disables the cap).
#
# Any env knob or trailing Hydra override of the base script still applies, e.g.
#   TIS=True ROLLOUT_QUANT=fp8 max_updates=200 bash <this script>
set -euo pipefail
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B"}
export model_tag=${model_tag:-"Qwen2.5-7B"}
export test_freq=${test_freq:-6}
export save_freq=${save_freq:-6}
export val_temperature=${val_temperature:-1.0}
export val_top_p=${val_top_p:-1.0}
export VERL_GPU_MEM_CAP_GB=${VERL_GPU_MEM_CAP_GB-76}
export gpu_memory_utilization=${gpu_memory_utilization:-0.283}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh" "$@"
