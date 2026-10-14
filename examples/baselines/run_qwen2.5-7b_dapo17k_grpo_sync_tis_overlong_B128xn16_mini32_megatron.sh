#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=main-ppo-sync-dapo17k-grpo-qwen2.5-7b-tis-overlong

# Qwen2.5-7B (BASE) synchronous TIS arm WITH DAPO's soft overlong punishment:
# run_qwen2.5-7b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh with OVERLONG_BUFFER_LEN=2048 and
# OVERLONG_PENALTY_FACTOR=1.0. Every other parameter - model, batch 140 x 16 / mini-batch 35, FlashRL's TIS C=8,
# clip 0.2/0.28/c10, warmup 3, weight decay 0.1, dynamic batch size, H100 emulation, cadence 5 - is that script's.
#
#   bash examples/baselines/run_qwen2.5-7b_dapo17k_grpo_sync_tis_overlong_B128xn16_mini32_megatron.sh
#
# WHY. In the Qwen2.5-7B TIS runs on remote_h200 truncation at the 8,192-token cap rose from ~0.05% to ~1-2% of
# responses after ~210 rollout steps, together with an entropy rise (0.5 -> 1.4) and falling AIME accuracy. A
# truncated response scores -1 and, under token-mean, puts all 8,192 of its tokens into the gradient. DAPO and
# FlashRL (yaof20/verl flash-rl, recipe/flash_rl/dapo_qwen32b_*.sh) shape the reward of long responses instead.
#
# OVERLONG PENALTY (the base launcher's "DAPO overlong reward shaping" block). reward.reward_manager.name=dapo with
#   reward += min(-(len - (8192 - 2048)) / 2048 * 1.0, 0)
# i.e. 0 up to 6,144 response tokens, then a linear ramp to -1 at the 8,192 cap: a correct answer of 8,000 tokens
# scores ~+0.07, a truncated one -2. DAPO / FlashRL used a 4,096 buffer under a 20,480 cap (the last 20%); 2,048 is
# the last 25% of this arm's cap. Tagged in exp_name (" overlong-2048-pf1.0"). val-core/*/acc/mean@1 is the
# scorer's acc and unaffected; val-aux/*/overlong/mean@1 is the share of validation responses in the buffer.
#
# NOT ADOPTED from DAPO / FlashRL:
#   * dynamic sampling (algorithm.filter_groups) - off: every prompt group trains, including all-correct and
#     all-wrong ones (zero advantage).
#   * overlong filtering (masking truncated responses out of the loss) - not used by verl's DAPO recipe or
#     FlashRL either, and not implemented in this fork.
#
# Any env knob or trailing Hydra override of the TIS script and the base launcher still applies, e.g.
#   OVERLONG_BUFFER_LEN=1024 train_prompt_bsz=144 train_prompt_mini_bsz=36 lr_warmup_steps=0 \
#   val_temperature=1.0 val_top_p=1.0 bash <this script>
# OVERLONG_BUFFER_LEN=0 turns the penalty off (= the TIS script).
set -euo pipefail
export OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-2048}
export OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen2.5-7b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh" "$@"
