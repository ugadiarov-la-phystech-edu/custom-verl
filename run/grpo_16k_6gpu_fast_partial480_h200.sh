#!/bin/bash

SETUP_PATH="${SETUP_PATH:-./setup.sh}"

echo "SETUP_PATH = $SETUP_PATH"
if [ ! -f "$SETUP_PATH" ]; then
    echo "Error: Setup file not found at $SETUP_PATH"
    exit 1
fi

source "$SETUP_PATH"
export RUN_NAME="grpo_8B_dapo_16k_6gpu_fast_partial480_h200_$(date +%Y%m%d-%H%M%S)"

export VLLM_DISABLE_CUSTOM_MM_ALLOCATOR=1
unset PYTORCH_CUDA_ALLOC_CONF

export VERL_ROLLOUT_BATCH_STATS=1
export VERL_OVERSAMPLE_DISCARD=0
export VERL_PARTIAL_ROLLOUT=1

# Per-GPU phase monitoring (monitoring/USAGE.md). The PPO runtime env forwards the
# VERL_GPU_PHASE_* vars to Ray workers; the RAY_* vars let the Ray dashboard embed Grafana.
export VERL_GPU_PHASE_MONITOR=${VERL_GPU_PHASE_MONITOR:-1}
export VERL_GPU_PHASE_HEARTBEAT_S=${VERL_GPU_PHASE_HEARTBEAT_S:-10}
export RAY_PROMETHEUS_HOST=${RAY_PROMETHEUS_HOST:-http://localhost:9090}
export RAY_GRAFANA_HOST=${RAY_GRAFANA_HOST:-http://localhost:3000}
export RAY_GRAFANA_IFRAME_HOST=${RAY_GRAFANA_IFRAME_HOST:-${RAY_GRAFANA_HOST}}


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=True \
    data.train_files="${TRAIN_PATH}" \
    data.val_files="${TEST_PATH}" \
    data.train_batch_size=384 \
    +data.gen_batch_size=480 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=96 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1  \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_model_len=18432 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.rollout.scheduling_policy=priority \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='grpo' \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPU} \
    trainer.nnodes=1 \
    trainer.save_freq=5 trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@
