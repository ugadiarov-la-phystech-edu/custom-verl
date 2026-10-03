#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_V1=1
export RAY_ADDRESS="local"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=disabled
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
model_tag=${model_tag:-"Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet','/home/jovyan/datasets/math_datasets/dapo/aime-2025.parquet']"}
project_name='vcpo'

SEED=${SEED:-1}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
n_gpus_rollout=${n_gpus_rollout:-5}
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"
gen_tp=1
n_resp_per_prompt=${n_resp_per_prompt:-16}
gpu_memory_utilization=${gpu_memory_utilization:-0.9}
emu_tag=""
if [[ -n "${VERL_GPU_MEM_CAP_GB:-}" ]]; then emu_tag=" h100-emu-${VERL_GPU_MEM_CAP_GB}gb-gmu${gpu_memory_utilization}"; fi
enable_chunked_prefill=True
calculate_log_probs=True

max_prompt_length=${max_prompt_length:-2048}
max_response_length=${max_response_length:-8192}
max_num_batched_tokens=$((max_prompt_length + max_response_length))

train_tp=1
train_pp=1
train_cp=1
sequence_parallel=False
use_remove_padding=True
precision_dtype="bfloat16"

train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=${train_prompt_mini_bsz:-33}
micro_bsz_per_gpu=1
log_prob_micro_bsz_per_gpu=1
concurrent_samples_per_replica=${concurrent_samples_per_replica:-${train_prompt_mini_bsz}}

DYNAMIC_BSZ=${DYNAMIC_BSZ:-False}
DYNAMIC_BSZ_MAX_TOKENS=${DYNAMIC_BSZ_MAX_TOKENS:-$((max_prompt_length + max_response_length))}
DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS=${DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS:-${DYNAMIC_BSZ_MAX_TOKENS}}
dynbsz_args=()
dynbsz_tag=""
case "${DYNAMIC_BSZ}" in
    True|true|1)
        use_dynamic_bsz=True
        for cap in "${DYNAMIC_BSZ_MAX_TOKENS}" "${DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS}"; do
            [[ "${cap}" =~ ^[1-9][0-9]*$ ]] || { echo "DYNAMIC_BSZ token caps must be positive integers, got '${cap}'" >&2; exit 2; }
            (( cap >= max_prompt_length + max_response_length )) || { echo "DYNAMIC_BSZ token caps must be >= max_prompt_length + max_response_length = $((max_prompt_length + max_response_length)), got ${cap}" >&2; exit 2; }
        done
        dynbsz_args=(
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${DYNAMIC_BSZ_MAX_TOKENS}"
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS}"
        )
        dynbsz_tag=" dynbsz-${DYNAMIC_BSZ_MAX_TOKENS}"
        ;;
    False|false|0) use_dynamic_bsz=False ;;
    *) echo "DYNAMIC_BSZ must be True or False, got '${DYNAMIC_BSZ}'" >&2; exit 2 ;;
esac

adv_estimator=grpo
loss_agg_mode=${loss_agg_mode:-"seq-mean-token-mean"}
clip_ratio=0.2
clip_ratio_low=0.2
clip_ratio_high=0.2
clip_ratio_c=3.0
use_kl_loss=False
kl_loss_coef=0.0
use_kl_in_reward=False
kl_coef=0.0
entropy_coeff=0
calculate_entropy=True
grad_clip=1.0

lr=1e-6
lr_warmup_steps=${lr_warmup_steps:-0}
[[ "${lr_warmup_steps}" =~ ^[0-9]+$ ]] || { echo "lr_warmup_steps must be a non-negative integer, got '${lr_warmup_steps}'" >&2; exit 2; }
warmup_tag=""
if [[ "${lr_warmup_steps}" != "0" ]]; then warmup_tag=" warmup-${lr_warmup_steps}"; fi
weight_decay=0.1
lr_decay_style="constant"

ess_enable=${ess_enable:-True}
min_ess=${min_ess:-1.1}
ess_lr_scale=${ess_lr_scale:-0.5}
ess_use_clipped=False
ess_tag="min-ess-${min_ess}-lrscale-${ess_lr_scale}"

bypass_mode=True
loss_type=reinforce
rollout_is="token"
rollout_is_threshold=${rollout_is_threshold:-2.0}
[[ "${rollout_is_threshold}" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk "BEGIN{exit !(${rollout_is_threshold} > 0)}" \
    || { echo "rollout_is_threshold must be a positive number, got '${rollout_is_threshold}'" >&2; exit 2; }
tis_tag=""
if awk "BEGIN{exit !(${rollout_is_threshold} != 2.0)}"; then tis_tag=" tis-C${rollout_is_threshold}"; fi
rollout_rs=null
rollout_rs_threshold=null
log_probs_pearson_corr=${log_probs_pearson_corr:-True}

ROLLOUT_QUANT=${ROLLOUT_QUANT:-bf16}
case "${ROLLOUT_QUANT}" in
    bf16) rollout_quantization=null; quant_tag="" ;;
    fp8) rollout_quantization=fp8; quant_tag=" rollout-fp8" ;;
    *) echo "ROLLOUT_QUANT must be bf16 or fp8, got '${ROLLOUT_QUANT}'" >&2; exit 2 ;;
esac

staleness_threshold=${staleness_threshold:-32.0}
updates_per_param_sync=1
num_minibatches_per_update=1
partial_rollout=True
use_rollout_log_probs=True

replay_enable=${replay_enable:-True}
replay_tau=${replay_tau:-8}
replay_staleness_threshold=${replay_staleness_threshold:-32}
replay_requires_mini_batches=${replay_requires_mini_batches:-0.5}
concurrency_ramp=${concurrency_ramp:-"[5, 12, 20]"}
ramp_tag=""
if [[ "${concurrency_ramp}" != "null" ]]; then ramp_tag=" ramp-$(echo "${concurrency_ramp}" | tr -d '[] ' | tr ',' '-')"; fi
replay_sampling_seed=${replay_sampling_seed:-${SEED}}
replay_reuse_halflife=${replay_reuse_halflife:-1}
replay_reuse_tag=""
if [[ "${replay_reuse_halflife}" != "null" ]]; then replay_reuse_tag=" nu-${replay_reuse_halflife}"; fi
replay_min_fresh_ratio=${replay_min_fresh_ratio:-0.5}
replay_fresh_tag=""
if [[ "${replay_min_fresh_ratio}" != "0" ]]; then replay_fresh_tag=" fresh-${replay_min_fresh_ratio}"; fi

serialize_validation=${serialize_validation:-True}
pause_generation_during_save=${pause_generation_during_save:-True}

total_rollout_steps=${total_rollout_steps:-66000}
lr_decay_steps=${lr_decay_steps:-${total_rollout_steps}}
[[ "${lr_decay_steps}" =~ ^[1-9][0-9]*$ ]] || { echo "lr_decay_steps must be a positive integer, got '${lr_decay_steps}'" >&2; exit 2; }
if (( lr_warmup_steps >= lr_decay_steps )); then
    echo "lr_warmup_steps=${lr_warmup_steps} must be < lr_decay_steps=${lr_decay_steps} (Megatron LR scheduler); lower lr_warmup_steps or raise lr_decay_steps / total_rollout_steps" >&2
    exit 2
fi
max_updates=${max_updates:-null}
epochs=10000000
test_freq=${test_freq:-25}
save_freq=${save_freq:-25}
max_actor_ckpt_to_keep=null
ckpt_save_contents="['hf_model']"
resume_mode=disable

exp_name=${exp_name:-"GRPO-noVCPO replay tau-${replay_tau} k-${replay_staleness_threshold} rmb-${replay_requires_mini_batches}${replay_reuse_tag}${replay_fresh_tag} ess-${ess_tag}${emu_tag}${ramp_tag} DAPO17K-AIME24 ${model_tag} ${n_gpus_rollout}-${n_gpus_training} tp${train_tp}dp${n_gpus_training} hdo B-${train_prompt_mini_bsz} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd${warmup_tag}${dynbsz_tag}${tis_tag}${quant_tag} seed-${SEED}"}
exp_name_safe=${exp_name//\//_}
log_dir=${log_dir:-"logs/${exp_name_safe}"}
CKPTS_DIR=${CKPTS_DIR:-"${log_dir}"}
mkdir -p -- "${log_dir}" "${CKPTS_DIR}"
export TENSORBOARD_DIR="${log_dir}/tensorboard"

trainer_logger="['console','tensorboard']"
log_val_generations=0
val_before_train=${val_before_train:-True}

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name=fully_async_ppo_megatron_trainer.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.seed=${SEED} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.loss_type=${loss_type} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.log_probs_pearson_corr=${log_probs_pearson_corr} \
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    '+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}' \
    actor_rollout_ref.actor.strategy=megatron \
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
    actor_rollout_ref.actor.ess_scaling.enable=${ess_enable} \
    actor_rollout_ref.actor.ess_scaling.min_ess=${min_ess} \
    actor_rollout_ref.actor.ess_scaling.lr_scale=${ess_lr_scale} \
    actor_rollout_ref.actor.ess_scaling.use_clipped=${ess_use_clipped} \
    actor_rollout_ref.actor.data_loader_seed=${SEED} \
    actor_rollout_ref.actor.megatron.seed=${SEED} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.sequence_parallel=${sequence_parallel} \
    actor_rollout_ref.actor.megatron.dtype=${precision_dtype} \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=False \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
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
    actor_rollout_ref.actor.use_rollout_log_probs=${use_rollout_log_probs} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.seed=${SEED} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dtype=${precision_dtype} \
    actor_rollout_ref.rollout.quantization=${rollout_quantization} \
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
    trainer.logger=${trainer_logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    "actor_rollout_ref.actor.checkpoint.save_contents=${ckpt_save_contents}" \
    trainer.resume_mode=${resume_mode} \
    trainer.rollout_data_dir=null \
    trainer.log_val_generations=${log_val_generations} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    trainer.total_epochs="${epochs}" \
    trainer.total_training_steps="${max_updates}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${updates_per_param_sync}" \
    async_training.require_batches="${num_minibatches_per_update}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.concurrent_samples_per_replica="${concurrent_samples_per_replica}" \
    async_training.concurrency_ramp="${concurrency_ramp}" \
    async_training.serialize_validation="${serialize_validation}" \
    async_training.pause_generation_during_save="${pause_generation_during_save}" \
    async_training.replay_buffer.enable="${replay_enable}" \
    async_training.replay_buffer.tau="${replay_tau}" \
    async_training.replay_buffer.staleness_threshold="${replay_staleness_threshold}" \
    async_training.replay_buffer.requires_mini_batches="${replay_requires_mini_batches}" \
    async_training.replay_buffer.sampling_seed="${replay_sampling_seed}" \
    async_training.replay_buffer.reuse_halflife="${replay_reuse_halflife}" \
    async_training.replay_buffer.min_fresh_ratio="${replay_min_fresh_ratio}" "${dynbsz_args[@]}" "$@"
