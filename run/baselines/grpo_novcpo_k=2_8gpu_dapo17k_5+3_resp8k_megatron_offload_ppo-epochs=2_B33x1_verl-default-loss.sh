#!/usr/bin/env bash
# grpo_novcpo_k=2_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1_verl-default-loss.sh
#
# Stock-verl-v0.8.0 port of the fork arm
#   custom_vcpo:recipe/fully_async_policy/shell/vcpo/dapo/opp-epochs_dapo-filter/
#     grpo_novcpo_k=2_8gpu_dapo17k_5+3_resp8k_megatron_offload_ppo-epochs=2_B33x1.sh
# Run from the repo root: the recipe yaml's hydra searchpath is file://verl/trainer/config
# (CWD-relative), so `cd` to the checkout before launching.
#
# THE ARM (unchanged from the fork):
#   * async_training.require_batches=1: the pull is ONE 33-group mini-batch per trainer
#     step (B-33x1) and with ppo_epochs=2 the trainer runs 2 AdamW updates per step —
#     two passes over the same 33 groups. Model versions tick per 33-group step.
#   * async_training.staleness_threshold=2 (k=2 gating): the rollouter is licensed to
#     generate up to (2+1) trainer batches ahead — 99 in-flight/queued groups.
#   * total_rollout_steps=66000 explicit, licensing up to ~2000 trainer steps of 33 groups.
#   * test_freq=10 / save_freq=10 in param-version units (versions tick per 33-group
#     step here) — validation/checkpointing every 330 groups.
#   * trainer tp=1/dp=3 (sequence_parallel needs TP>1), 33*16=528 seqs divide by DP=3,
#     HDO full CPU offload with bf16 master weights (do NOT swap for
#     use_precision_aware_optimizer without optimizer_cpu_offload: silent stall).
#
# THE LOSS (deliberately NOT the fork's):
#   This variant is the sibling of ..._B33x1_bypass-reinforce.sh and differs from it ONLY
#   in the loss: it sets no algorithm.rollout_correction.* and no policy_loss override,
#   i.e. stock verl's out-of-the-box fully-async objective. The recipe yaml's
#   bypass_mode=True still aliases old_log_probs := rollout_log_probs and skips the
#   old-log-prob forward pass, and the worker's default loss_mode=vanilla then computes a
#   LIVE clipped ratio pi_theta/pi_rollout with clip_ratio_low/high=0.2 and dual-clip
#   c=3.0, with NO importance weighting.
#   This is a different objective from what the fork ran (there the ratio was identically
#   1 and the correction entered as truncated IS weights). Setting rollout_is here would
#   change nothing: in bypass mode the driver-side IS path is skipped
#   (separation/ray_trainer.py:596) and the vanilla loss receives rollout_is_weights=None,
#   so the honest encoding of "verl default" is to leave the whole block out.
#
# RESIDUAL DIFFERENCES vs the fork (cannot be closed by config — read the comparison
# with these in mind):
#   * Rollout concurrency cap: fork min(5*bsz_per_dp_rank, 99)=99; upstream hardcodes 16
#     (fully_async_rollouter.py:541) -> min(5*16, 99)=80.
#   * No stop-the-world validation/save: the fork's serialize_validation and
#     pause_generation_during_save have no upstream equivalent. Upstream validates on the
#     rollout replicas concurrently with training generation, and there is no
#     cumulative_training_time virtual clock.
#   * Both ppo epochs run inside ONE update_actor call (same 2 AdamW steps over the same
#     528 sequences) instead of the fork's two driver-side calls; the LR scheduler
#     advances differently (immaterial at lr_decay_style=constant).
#   * Metrics: no training/rollout_actor_probs_pearson_corr (bypass skips the debug
#     path); no rollout_corr/* stats at all in this variant (the driver-side path is
#     skipped by bypass mode and the vanilla loss computes none); entropy logs as
#     actor/entropy_loss. This arm emits a non-zero actor/pg_clipfrac — the check that
#     the clipped-vanilla path really ran.
#   * MATH-500 validation dropped: data_source=math500_dapo has no scorer in stock verl
#     (reward_score/__init__.py raises NotImplementedError). AIME-2024 only.
#   * OPPORTUNISTIC PPO EPOCHS and DAPO FILTERING were off in the fork arm and have no
#     upstream counterpart, so they are simply absent here. VCPO mechanisms stay off.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_V1=1
export RAY_ADDRESS="local"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=disabled
export VLLM_USE_FLASHINFER_SAMPLER=0

# ================= Paths =================
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
# Single validation set: aime-2024.parquet (data_source=math_dapo) ->
# val-core/math_dapo/acc/mean@1. The fork's second file (math500.parquet,
# data_source=math500_dapo) is dropped: its scorer is fork-only.
TEST_FILE=${TEST_FILE:-"/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet"}

project_name='vcpo'

# ================= GPU Layout =================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
n_gpus_rollout=${n_gpus_rollout:-5}
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

# ================= Rollout =================
rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"
gen_tp=1
n_resp_per_prompt=${n_resp_per_prompt:-16}
gpu_memory_utilization=0.9
enable_chunked_prefill=True
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=2048
max_response_length=8192
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# ================= Megatron Parallelism =================
train_tp=1 # only valid TP for 3 trainer GPUs (pure DP, no TP comm)
train_pp=1
train_cp=1
sequence_parallel=False # requires TP>1
use_remove_padding=True
precision_dtype="bfloat16"

# ================= Batch Sizes =================
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=33 # 33*16=528 seqs; must divide by trainer DP=3 (528/3=176)
micro_bsz_per_gpu=1
use_dynamic_bsz=False
log_prob_micro_bsz_per_gpu=1

# ================= Algorithm =================
adv_estimator=grpo
loss_agg_mode="seq-mean-token-mean"
clip_ratio=0.2
clip_ratio_low=0.2
clip_ratio_high=0.2
clip_ratio_c=3.0
use_kl_loss=False
kl_loss_coef=0.0
use_kl_in_reward=False
kl_coef=0.0
entropy_coeff=0
calculate_entropy=True # log actor/entropy_loss even with entropy_coeff=0
grad_clip=1.0

# ================= Optimizer =================
lr=1e-6
lr_warmup_steps=0
weight_decay=0.1

# ================= IS / Rollout Correction =================
# Left at stock verl defaults on purpose: the recipe yaml gives bypass_mode=True
# (old_log_probs := rollout_log_probs, old-log-prob pass skipped) and rollout_is=null,
# and the actor keeps loss_mode=vanilla -> PPO-clip on pi_theta/pi_rollout, no IS.
# The clip_ratio* values above DO bind in this variant.

# ================= Async Training =================
# k=2 matches the VCPO baseline's staleness gating: the rollouter is licensed to
# generate up to (2+1) trainer batches ahead — at B-33x1 that is 99 in-flight/queued
# groups (upstream additionally caps concurrency at 5*16=80, see the header).
staleness_threshold=${staleness_threshold:-2.0}
updates_per_param_sync=1
num_minibatches_per_update=1 # require_batches=1: ONE 33-group mini-batch per trainer step (B-33x1)
partial_rollout=True

# ================= PPO epochs =================
# actor.ppo_epochs=2 -> 2 AdamW updates per trainer step: two passes over the single
# 33-group mini-batch of the pull. shuffle/data_loader_seed replace the fork's
# ppo_epochs_shuffle_seed (a no-op at require_batches=1: the pull IS the mini-batch).
ppo_epochs=${ppo_epochs:-2}
ppo_epochs_shuffle_seed=${ppo_epochs_shuffle_seed:-1234}

# ================= Training/Rollout Steps =================
# Explicit 66000 (NOT the base arms' 500-step formula, which at B-33x1 would shrink to
# 500*1*1*33 = 16500): same generation budget as the B-33x4 arms, licensing up to ~2000
# trainer steps of 33 groups.
total_rollout_steps=${total_rollout_steps:-66000}
epochs=10000000
# test/save freq are in param-version units; versions tick per 33-group step here, so
# 10 = every 330 groups. Use -1 to disable (0 raises ZeroDivisionError upstream).
test_freq=${test_freq:-10}
save_freq=${save_freq:-10}
max_actor_ckpt_to_keep=1 # keep only the most recent checkpoint

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO-v080 verl-default-loss k-${staleness_threshold} DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} tp1dp3 hdo B-${train_prompt_mini_bsz}x${num_minibatches_per_update} ppo-epochs-${ppo_epochs} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"}
exp_name_safe=${exp_name//\//_}
log_dir="logs/${exp_name_safe}"
CKPTS_DIR="${log_dir}"
mkdir -p -- "${log_dir}"
export TENSORBOARD_DIR="${log_dir}/tensorboard"

trainer_logger="['console','tensorboard']"
log_val_generations=0
val_before_train=${val_before_train:-True}

# ================= LR decay =================
lr_decay_style="constant"
lr_decay_steps=${total_rollout_steps}

# ================= Run =================
python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name=fully_async_ppo_megatron_trainer.yaml \
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
    actor_rollout_ref.actor.strategy=megatron \
    critic.strategy=megatron \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.data_loader_seed=${ppo_epochs_shuffle_seed} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.sequence_parallel=${sequence_parallel} \
    actor_rollout_ref.actor.megatron.dtype=${precision_dtype} \
    actor_rollout_ref.actor.megatron.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr_decay_style=${lr_decay_style} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.clip_grad=${grad_clip} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_torch_optimizer_for_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.main_params_dtype=bfloat16 \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.calculate_entropy=${calculate_entropy} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.sequence_parallel=${sequence_parallel} \
    actor_rollout_ref.ref.megatron.dtype=${precision_dtype} \
    actor_rollout_ref.ref.megatron.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dtype=${precision_dtype} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n:-1} \
    actor_rollout_ref.rollout.calculate_log_probs=${calculate_log_probs} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    critic.megatron.tensor_model_parallel_size=${train_tp} \
    critic.megatron.pipeline_model_parallel_size=${train_pp} \
    critic.megatron.context_parallel_size=${train_cp} \
    critic.megatron.sequence_parallel=${sequence_parallel} \
    critic.megatron.dtype=${precision_dtype} \
    trainer.logger=${trainer_logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${epochs} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    trainer.rollout_data_dir="${log_dir}" \
    trainer.log_val_generations=${log_val_generations} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${updates_per_param_sync}" \
    async_training.require_batches="${num_minibatches_per_update}" \
    async_training.partial_rollout="${partial_rollout}" "$@"
