#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=grpo-novcpo-replay-ess-nu1-fresh0.5
#
# Qwen3-8B fully-async GRPO, REPLAY-BUFFER arm with the MIN-ESS LR brake, reuse decay and fresh-share
# gate. 5 vLLM rollout GPUs + 3 Megatron trainer GPUs (tp1/dp3, HDO: optimizer fully CPU-offloaded with
# bf16 master weights). Launch from the repo root.
#
# What it runs (all knobs env-overridable, the tagged ones also named in exp_name):
#   * replay buffer on the trainer (verl/experimental/fully_async_policy/replay_buffer.py): groups scored
#     2^(-staleness/tau), tau=8, evicted at staleness > k=32; every update composes one mini-batch of 33
#     groups FRESH-FIRST (groups arrived since the previous composition, newest first), then a draw
#     weighted by the staleness score x 2^(-times_trained/nu), nu=1 (reuse decay);
#   * fresh-share gate: an update waits for ceil(0.5 x 33) = 17 newly arrived groups (waived once the
#     rollouter is done or after replay_buffer.min_fresh_wait_timeout_s, logged replay/fresh_floor_waived);
#   * first update: requires_mini_batches=0.5 -> an all-fresh 18-group mini-batch (16.5 rounded up so the
#     18 x 16 sequences split over the 3 trainer ranks), full 33-group mini-batches afterwards;
#   * rollouter insertion gate: groups whose 16 rewards are all equal are dropped (their generation
#     quota slot is released); kept groups carry frozen GRPO advantages; one update per weight sync;
#   * objective: REINFORCE with token-level truncated IS (C=2) against the cached rollout log-probs;
#   * min-ESS brake: the sequence-level Kish ESS of the mini-batch's IS weights (unclipped) is measured
#     before each optimizer step; ESS <= min_ess (1.1 effective samples, i.e. near the structural floor of
#     1) steps at lr * 0.5, every other step at the full lr. Logged: staleness/ess, actor/ess_lr_mult,
#     actor/ess_scaled_lr;
#   * concurrency ramp [5, 12, 20] groups per engine until the 1st/2nd/3rd mini-batch is delivered;
#   * stop-the-world validation and checkpoint saves (generation frozen meanwhile);
#   * checkpoints: hf_model only every 25 updates, resume disabled.
#
# PORT NOTES (custom_vcpo branch replay_buffer_vcpo_ess_threshold_dyn-batch_min-ess_reuse-penalty_
# freshness_first-update_emu_openpangu -> verl v0.9.0 fully_async_policy, same script name):
#   * Objective. The source's skip_recompute_old_log_prob per-trajectory path (PPO-clip with
#     old = log pi.detach(), i.e. ratio 1, token TIS weights, advantage folded into a per-trajectory loss
#     scale, micro-batch 1) is gradient-identical to bypass_mode + loss_type=reinforce + rollout_is=token
#     with seq-mean-token-mean (tests/workers/utils/test_ppo_loss_ess_on_cpu.py). The two policy_loss keys
#     are required: losses.py reads only actor.policy_loss, algorithm.rollout_correction alone never
#     reaches the workers. update_policy_per_traj / grad_baselining (OPOB) do not exist here.
#   * ESS brake: engine pre-optimizer-step hook, the ESS all-reduced over the trainer DP group (requires
#     pp=1, cp=1). Pearson diagnostic: algorithm.rollout_correction.log_probs_pearson_corr.
#   * bsz_per_dp_rank -> async_training.concurrent_samples_per_replica.
#   * rollout.total_epochs / rollout.test_freq -> trainer.total_epochs / trainer.test_freq.
#   * Stop-the-world pauses abort in-flight requests and hold their partial-rollout resume at the load
#     balancer; validation runs on the resumed engines (requires partial_rollout=True).
#   * Dropped (no-ops or not ported): actor.megatron.use_remove_padding / grad_offload, ref.*, critic.*,
#     async_training.{skip_recompute_old_log_prob, compute_prox_log_prob, dynamic_filtering.*,
#     opportunistic_epochs.*, ppo_epochs*, save_queue_state, replay_buffer.save_state} (the trainer
#     rejects the unported ones if enabled). The recompute_* overrides lost their `+` (the keys exist).
#   * Added: actor_rollout_ref.rollout.seed=${SEED} (the source's vLLM seed was always 0).
#   * Micro-batch size 1 is no longer required (the loss is micro-batch invariant); kept at 1 so the
#     memory envelope of the source runs carries over.
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_V1=1
export RAY_ADDRESS="local"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=disabled
export VLLM_USE_FLASHINFER_SAMPLER=0
# Unbuffered worker stdout: Ray block-buffers prints otherwise, lagging the live log by minutes.
export PYTHONUNBUFFERED=1

# ================= Paths =================
# Two validation sets, reported separately by data_source:
#   aime-2024.parquet (data_source=math_dapo) -> val-core/math_dapo/acc/mean@1
#   aime-2025.parquet (data_source=aime2025_dapo) -> val-core/aime2025_dapo/acc/mean@1
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/home/jovyan/datasets/math_datasets/dapo/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"['/home/jovyan/datasets/math_datasets/dapo/aime-2024.parquet','/home/jovyan/datasets/math_datasets/dapo/aime-2025.parquet']"}
project_name='vcpo'

# ================= Seeds =================
# One SEED feeds data.seed (prompt order), actor megatron.seed, actor data_loader_seed, the replay draw
# (replay_sampling_seed unless overridden) and the vLLM sampling seed. Part of exp_name.
SEED=${SEED:-1}

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
gpu_memory_utilization=${gpu_memory_utilization:-0.9}
# H100 emulation on bigger cards (verl/utils/gpu_memory_cap.py): VERL_GPU_MEM_CAP_GB=80 exported in the
# launching shell caps the TRAINER allocator; pair it with gpu_memory_utilization=0.5 on an H200. Only
# read here, never set; tagged in exp_name.
emu_tag=""
if [[ -n "${VERL_GPU_MEM_CAP_GB:-}" ]]; then emu_tag=" h100-emu-${VERL_GPU_MEM_CAP_GB}gb-gmu${gpu_memory_utilization}"; fi
enable_chunked_prefill=True
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=${max_prompt_length:-2048}
max_response_length=${max_response_length:-8192}
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# ================= Megatron Parallelism =================
train_tp=1 # only valid TP for 3 trainer GPUs (pure DP); the ESS brake also needs pp=1, cp=1
train_pp=1
train_cp=1
sequence_parallel=False # requires TP>1
use_remove_padding=True
precision_dtype="bfloat16"

# ================= Batch Sizes =================
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=${train_prompt_mini_bsz:-33} # 33*16=528 seqs; mini*n must divide by trainer DP=3
micro_bsz_per_gpu=1
use_dynamic_bsz=False
log_prob_micro_bsz_per_gpu=1
# Per-engine in-flight group cap (the source's bsz_per_dp_rank): 5 engines x 33 = 165 groups
concurrent_samples_per_replica=${concurrent_samples_per_replica:-${train_prompt_mini_bsz}}

# ================= Algorithm =================
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
calculate_entropy=True # log actor/entropy even with entropy_coeff=0
grad_clip=1.0

# ================= Optimizer =================
lr=1e-6
lr_warmup_steps=0
weight_decay=0.1
lr_decay_style="constant"

# ================= Min-ESS LR brake =================
ess_enable=${ess_enable:-True}
min_ess=${min_ess:-1.1}
ess_lr_scale=${ess_lr_scale:-0.5}
ess_use_clipped=False # ESS of unclipped ratios: the brake must see what truncation hides
ess_tag="min-ess-${min_ess}-lrscale-${ess_lr_scale}"

# ================= IS / Rollout Correction =================
# REINFORCE with token-level truncated IS against the cached behavior (rollout) log-probs.
bypass_mode=True
loss_type=reinforce
rollout_is="token"
rollout_is_threshold="2.0"
rollout_rs=null
rollout_rs_threshold=null
log_probs_pearson_corr=${log_probs_pearson_corr:-True}

# ================= Async Training =================
# Generation quota aligned with the eviction horizon: groups older than k updates are evicted anyway.
staleness_threshold=${staleness_threshold:-32.0}
updates_per_param_sync=1     # REQUIRED by replay mode: sync after every update
num_minibatches_per_update=1 # REQUIRED by replay mode: one mini-batch per update
partial_rollout=True         # REQUIRED by the stop-the-world pauses
use_rollout_log_probs=True

# ================= Replay buffer =================
replay_enable=${replay_enable:-True}
replay_tau=${replay_tau:-8}
replay_staleness_threshold=${replay_staleness_threshold:-32}
# (0, 1): a smaller all-fresh FIRST mini-batch only (0.5 x 33 -> 18 groups); >= 1: the pause watermark in
# mini-batches. Tagged rmb-<value>.
replay_requires_mini_batches=${replay_requires_mini_batches:-0.5}
# Per-engine caps for the warm-up stages ("null" = off): 25 groups in flight until the first mini-batch
# (18) is delivered, 60 until 51, 100 until 84, then the full 165. Tagged ramp-5-12-20.
concurrency_ramp=${concurrency_ramp:-"[5, 12, 20]"}
ramp_tag=""
if [[ "${concurrency_ramp}" != "null" ]]; then ramp_tag=" ramp-$(echo "${concurrency_ramp}" | tr -d '[] ' | tr ',' '-')"; fi
replay_sampling_seed=${replay_sampling_seed:-${SEED}}
# Reuse-decay half-life in trainings (null = staleness-only draw). Tagged nu-<value>.
replay_reuse_halflife=${replay_reuse_halflife:-1}
replay_reuse_tag=""
if [[ "${replay_reuse_halflife}" != "null" ]]; then replay_reuse_tag=" nu-${replay_reuse_halflife}"; fi
# Fresh-share gate (0 = never wait). Watch replay/minibatch_new_ratio, replay/fresh_wait_s and
# replay/minibatch_fresh_staleness_mean. Tagged fresh-<value>.
replay_min_fresh_ratio=${replay_min_fresh_ratio:-0.5}
replay_fresh_tag=""
if [[ "${replay_min_fresh_ratio}" != "0" ]]; then replay_fresh_tag=" fresh-${replay_min_fresh_ratio}"; fi

# ================= Stop-the-world accounting =================
serialize_validation=${serialize_validation:-True}
pause_generation_during_save=${pause_generation_during_save:-True}

# ================= Training/Rollout Steps =================
# 66000-prompt generation budget (fed prompts, not kept groups).
total_rollout_steps=${total_rollout_steps:-66000}
# Cap on OPTIMIZER UPDATES (= parameter versions: one sync per update) via trainer.total_training_steps;
# null = the prompt budget decides. At the cap the trainer validates, writes a final hf_model checkpoint
# and the rollouter is cancelled.
max_updates=${max_updates:-null}
epochs=10000000
# Model versions tick once per UPDATE: validate / checkpoint every 25 updates.
test_freq=${test_freq:-25}
save_freq=${save_freq:-25}
max_actor_ckpt_to_keep=null
ckpt_save_contents="['hf_model']"
resume_mode=disable

# ================= Logging =================
exp_name=${exp_name:-"GRPO-noVCPO replay tau-${replay_tau} k-${replay_staleness_threshold} rmb-${replay_requires_mini_batches}${replay_reuse_tag}${replay_fresh_tag} ess-${ess_tag}${emu_tag}${ramp_tag} DAPO17K-AIME24 Qwen3-8B ${n_gpus_rollout}-${n_gpus_training} tp1dp3 hdo B-${train_prompt_mini_bsz} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd seed-${SEED}"}
exp_name_safe=${exp_name//\//_}
# log_dir: TensorBoard and the rollout / validation dumps; CKPTS_DIR: global_step_N/ checkpoints.
log_dir=${log_dir:-"logs/${exp_name_safe}"}
CKPTS_DIR=${CKPTS_DIR:-"${log_dir}"}
mkdir -p -- "${log_dir}" "${CKPTS_DIR}"
export TENSORBOARD_DIR="${log_dir}/tensorboard"

trainer_logger="['console','tensorboard']"
log_val_generations=0
val_before_train=${val_before_train:-True}

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
    trainer.rollout_data_dir="${log_dir}" \
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
    async_training.replay_buffer.min_fresh_ratio="${replay_min_fresh_ratio}" "$@"
