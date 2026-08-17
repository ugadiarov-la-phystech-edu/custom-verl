#!/usr/bin/env bash
# GRPO replay-buffer + ESS-brake arm on the fully-async FSDP2 stack
# (port of the fork's grpo_novcpo_..._fsdp2_replay_tau=16_k=64_ess-sqrt_base=auto_trig=0.33333.sh).
#
# Algorithmic knobs identical to the winning trigger arm:
#   * trainer-side replay buffer: tau=16 score decay, eviction at staleness 64,
#     requires_mini_batches=1, weight sync after EVERY update, DAPO insertion
#     gate at the rollouter (degenerate groups never enqueued), frozen
#     insertion-time rewards/advantages (advantage_scalar broadcast).
#   * loss_mode=seq_adv_post_scale: clipped surrogate with UNIT advantages
#     (ratio anchored on the update's own forward), truncated token-level IS
#     (min(pi/mu, 2)) vs the cached behavior log-probs, per-sequence advantage
#     post-scaling, seq-mean-token-mean aggregation.
#   * ESS brake: lr *= sqrt(min(1, ess_ratio/base)) only when ess_ratio/base
#     falls below trigger_ratio=1/3; base auto-calibrated from the first
#     update's measured (on-policy) ESS ratio and persisted in replay_buffer.pt.
#   * stop-the-world accounting: serialize_validation + pause_generation_during_save,
#     both excluded from fully_async/timing/cumulative_training_time (virtual clock).
#   * checkpoints carry replay buffer + message queue + timing state, so resume
#     continues the run (in-flight server-side generations are the only loss).
# B = 33 prompts x 16 responses = 528 sequences per update, lr 1e-6, 8K responses,
# 5 rollout GPUs + 3 trainer GPUs on one 8xH100 node.

set -xeuo pipefail

project_name='vcpo'
exp_name='GRPO-replay-tau16-k64-ess-sqrt-base-auto-trig-0.33333-fsdp2-fully-async-5-3'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

rollout_mode="async"
rollout_name="vllm"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

# Sequence lengths
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))

# Loss: exact parity with the fork's trigger arm
loss_agg_mode="seq-mean-token-mean"
loss_mode="seq_adv_post_scale"
rollout_is_threshold=2.0

# ESS-guided LR scaling (VCPO)
ess_enable=${ess_enable:-True}
ess_rule=${ess_rule:-sqrt}
ess_base=${ess_base:-null}          # null = auto-calibrate from the first update
ess_use_clipped=False               # brake watches unclipped ratios (paper)
ess_trigger=${ess_trigger:-0.33333} # deadband: full lr at or above base/3

# Sampling
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=0.8
val_top_p=0.7

# Performance
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
ref_offload=True
actor_offload=False
gen_tp=1
sp_size=1
fsdp_size=${fsdp_size:-3} # full sharding over the 3 trainer GPUs

# GPU layout: one 8-GPU node, 5 rollout + 3 trainer
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_TRAIN=${NGPUS_TRAIN:-3}
NGPUS_ROLLOUT=${NGPUS_ROLLOUT:-5}

# Batching: 33 prompt-groups per update (33*16=528 sequences)
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=${train_prompt_mini_bsz:-33}
total_rollout_steps=${total_rollout_steps:-66000}
test_freq=${test_freq:-20} # param-version units: validate/save every 20 updates
save_freq=${save_freq:-20}

# Async/replay: generation licensed up to the eviction horizon
staleness_threshold=${staleness_threshold:-64}
trigger_parameter_sync_step=1 # REQUIRED by replay mode: sync after every update
require_batches=1             # REQUIRED by replay mode: one mini-batch per update
partial_rollout=True
bsz_per_dp_rank=${bsz_per_dp_rank:-${train_prompt_mini_bsz}}

replay_tau=${replay_tau:-16}
replay_staleness_threshold=${replay_staleness_threshold:-64}
replay_requires_mini_batches=${replay_requires_mini_batches:-1}
replay_sampling_seed=${replay_sampling_seed:-1234}

python -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.log_probs_pearson_corr=True \
    actor_rollout_ref.actor.ess_scaling.enable=${ess_enable} \
    actor_rollout_ref.actor.ess_scaling.scaling_rule=${ess_rule} \
    actor_rollout_ref.actor.ess_scaling.base_ess_ratio=${ess_base} \
    actor_rollout_ref.actor.ess_scaling.use_clipped=${ess_use_clipped} \
    actor_rollout_ref.actor.ess_scaling.trigger_ratio=${ess_trigger} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward.reward_manager.name=dapo \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_TRAIN}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_ROLLOUT}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.total_epochs=10000000 \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.serialize_validation=True \
    async_training.pause_generation_during_save=True \
    async_training.save_queue_state=True \
    async_training.bsz_per_dp_rank="${bsz_per_dp_rank}" \
    async_training.replay_buffer.enable=True \
    async_training.replay_buffer.tau="${replay_tau}" \
    async_training.replay_buffer.staleness_threshold="${replay_staleness_threshold}" \
    async_training.replay_buffer.requires_mini_batches="${replay_requires_mini_batches}" \
    async_training.replay_buffer.sampling_seed="${replay_sampling_seed}"
