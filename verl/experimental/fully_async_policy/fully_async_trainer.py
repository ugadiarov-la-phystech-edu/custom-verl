# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    assemble_batch_from_rollout_samples,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.fully_async_policy.replay_buffer import ReplayBuffer
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        # distillation config needed by _update_actor in ray_trainer.py
        from verl.trainer.distillation.losses import is_distillation_enabled

        if is_distillation_enabled(self.config.get("distillation")):
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.distillation_config = None

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self._init_dump_executor()
        self.validation_generations_logger = None
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async config ====================

        self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # Hybrid checkpoint manager for trainer-side validation (use_trainer_do_validate)
        # Uses naive backend to sync weights from trainer to hybrid rollout replicas.
        # Initialized in _setup_hybrid_checkpoint_manager_and_sleep() via set_rollouter().
        self.hybrid_checkpoint_manager = None

        # ==================== Replay-buffer training loop (VCPO port) ====================
        replay_cfg = config.async_training.get("replay_buffer", None)
        self.replay_enable = bool(replay_cfg is not None and replay_cfg.get("enable", False))
        if self.replay_enable:
            assert self.trigger_parameter_sync_step == 1, "replay mode syncs weights after every update"
            assert self.require_batches == 1, "replay mode consumes one mini-batch per update"
            self.replay_buffer = ReplayBuffer(
                tau=float(replay_cfg.get("tau", 16)),
                staleness_threshold=int(replay_cfg.get("staleness_threshold", 64)),
                seed=int(replay_cfg.get("sampling_seed", 1234)),
            )
            self.replay_requires_mini_batches = float(replay_cfg.get("requires_mini_batches", 1))
            self.replay_warmup_updates = math.ceil(self.replay_requires_mini_batches)
            self.replay_updates_done = 0
            self.rollout_done = False
            # Auto-calibrated ESS reference: with ess_scaling.enable=True and
            # base_ess_ratio=null, the first update runs unscaled and its measured
            # (on-policy, staleness-0 warm-up) ESS ratio becomes the base, passed
            # to the actor via meta_info["ess_base_override"] and persisted in
            # replay_buffer.pt across restarts.
            actor_cfg = config.actor_rollout_ref.actor
            self.replay_ess_auto_base = bool(
                actor_cfg.ess_scaling.get("enable", False) and actor_cfg.ess_scaling.get("base_ess_ratio", None) is None
            )
            self.replay_ess_use_clipped = bool(actor_cfg.ess_scaling.get("use_clipped", False))
            self.replay_ess_base = None
        self.pause_generation_during_save = bool(config.async_training.get("pause_generation_during_save", False))
        self.save_queue_state = bool(config.async_training.get("save_queue_state", True))

        # Virtual clock for the cumulative_training_time metric: the wall time an
        # identical run with neither validation nor checkpointing would have
        # needed (see _add_cumulative_time_metrics).
        self.rollouter_first_sample_time = None
        self.cumulative_save_time = 0.0
        self.cumulative_validation_time = 0.0
        self.virtual_free_time = None
        self._step_virtual_start = None
        self._step_actual_start = None
        self._step_valid_time = 0.0
        self._step_save_time = 0.0
        # Offsets restored from timing_state.json on resume.
        self.timing_wall_offset = 0.0
        self.timing_validation_offset = 0.0
        self.timing_save_offset = 0.0
        self.virtual_training_time_offset = 0.0

    async def _setup_checkpoint_manager(self):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    async def _setup_hybrid_checkpoint_manager(self):
        """Setup hybrid checkpoint manager and perform initial sleep of hybrid replicas.

        When use_trainer_do_validate is enabled:
          1. Creates a CheckpointEngineManager with naive backend for trainer-side
             weight sync to hybrid rollout replicas.
          2. Fetches hybrid replicas from the rollouter's ALM (created during
             rollouter.init_workers()).
          3. Registers them with the hybrid CP manager and calls sleep_replicas()
             to release GPU memory for training.

        Must be called AFTER set_rollouter() so that self.rollouter is available,
        and AFTER rollouter.init_workers() so that hybrid replicas exist.
        This mirrors the colocate pattern in ray_trainer.py:882-889 but fetches
        replicas from the rollouter's ALM via RPC since they live on the rollout side.
        """
        if not self.config.async_training.use_trainer_do_validate:
            return

        # --- Part 1: Create hybrid CheckpointEngineManager with naive backend ---
        print("[FullyAsyncTrainer] Setting up hybrid checkpoint manager (naive backend)")

        # Create hybrid CheckpointEngineManager with naive backend.
        checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
        original_backend = checkpoint_engine_cfg.backend
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = "naive"
        checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

        self.hybrid_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=[],  # Start empty; will be populated below
        )

        # Restore original backend value
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = original_backend

        print("[FullyAsyncTrainer] Hybrid checkpoint manager initialized (naive backend)")

        # --- Part 2: Fetch hybrid replicas from rollouter's ALM ---
        print("[FullyAsyncTrainer] Fetching hybrid replicas from rollouter...")
        hybrid_replicas_dict = ray.get(self.rollouter.get_all_hybrid_replicas.remote())
        print(
            f"[FullyAsyncTrainer] Got {len(hybrid_replicas_dict)} hybrid replicas: {list(hybrid_replicas_dict.keys())}"
        )

        if not hybrid_replicas_dict:
            print("[FullyAsyncTrainer] No hybrid replicas found, skipping initial sleep")
            return

        # --- Part 3: Register replicas and perform initial sleep ---
        for resource_id, replica in hybrid_replicas_dict.items():
            self.hybrid_checkpoint_manager.replicas.append(replica)
            print(
                f"[FullyAsyncTrainer] Registered '{resource_id}' "
                f"(mode={getattr(replica, 'rollout_mode', '?')}, "
                f"addr={getattr(replica, '_server_address', '?')})"
            )

        # Step 3: Sleep all hybrid replicas
        print(
            f"[FullyAsyncTrainer] Calling sleep_replicas() on "
            f"{len(self.hybrid_checkpoint_manager.replicas)} replicas..."
        )
        await self.hybrid_checkpoint_manager.sleep_replicas()
        print("[FullyAsyncTrainer] Initial sleep complete, GPU memory now owned by training engine")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    async def set_rollouter(self, rollouter):
        """Set rollouter reference and initialize all checkpoint managers."""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        await self._setup_checkpoint_manager()
        await self._setup_hybrid_checkpoint_manager()

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = await self.message_queue_client.get_sample()

            if sample is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            queue_samples.append(sample)

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds. "
            f"mq_len: {queue_len}"
        )

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor — always use Role.Actor (not ActorRollout) even when
        # use_trainer_do_validate is enabled. Rollout capability on trainer GPUs
        # is handled by ElasticAgentLoopManager's hybrid replicas.
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_reward_model_class(self):
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        if self.replay_enable:
            return await self._fit_replay()

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        steps = self.config.global_profiler.steps
        should_profile = steps is not None and (self.current_param_version + 1) in steps
        self._fit_start_profile(should_profiler=should_profile)

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile(should_profiler=should_profile)
        self._fit_collect_metrics(batch)
        self._fit_postprocess_step()

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        steps = self.config.global_profiler.steps
        last_profiler_step = self.current_param_version
        if steps is not None and last_profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._stop_profiling.remote().future())

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        profiler_step = last_profiler_step + 1

        if steps is not None and profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._start_profiling.remote().future())

        # Reset staleness in rollouter
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return
        # Execute validation
        if self.config.async_training.use_trainer_do_validate:
            await self._trainer_side_validate()
        else:
            val_metrics = await self.rollouter.do_validate.remote()
            self.logger.log(data=val_metrics, step=self.current_param_version)

    async def _trainer_side_validate(self):
        """Run trainer-side validation using hybrid rollout replicas."""
        print("[FullyAsyncTrainer] _trainer_side_validate === START ===")
        validate_start = time.time()
        # ================================================================
        # Phase 1: Switch ALL trainer GPUs to ROLLOUT mode
        # ================================================================
        phase_1_start = time.time()
        print("[FullyAsyncTrainer] Phase 1: Switching all GPUs to ROLLOUT mode")
        await self.hybrid_checkpoint_manager.update_weights(global_steps=self.current_param_version)
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        hybrid_replicas_dict = await self.rollouter.get_all_hybrid_replicas.remote()
        hybrid_resource_ids = list(hybrid_replicas_dict.keys())
        await self.rollouter.add_replicas.remote(hybrid_resource_ids)
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()
        print(f"[FullyAsyncTrainer] Phase 1 done ({time.time() - phase_1_start:.2f}s)")

        # ================================================================
        # Phase 2: Run validation via RPC to rollouter
        # ================================================================
        print("[FullyAsyncTrainer] Phase 2: Running validation")
        val_metrics = await self.rollouter.do_validate.remote()
        self.logger.log(data=val_metrics, step=self.current_param_version)

        # ================================================================
        # Phase 3: Switch hybrid GPUs back to TRAIN mode
        # ================================================================
        print("[FullyAsyncTrainer] Phase 3: Switching hybrid GPUs back to TRAIN mode")
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        # Batch remove all hybrid replicas from the load balancer in a single RPC.
        await self.rollouter.remove_replicas.remote(hybrid_resource_ids)
        await self.hybrid_checkpoint_manager.sleep_replicas()
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

        total_time = time.time() - validate_start
        print(f"[FullyAsyncTrainer] _trainer_side_validate === END === (total: {total_time:.2f}s)")

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            force or self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        self._load_replay_checkpoint_extras(global_step_folder)

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

    # ==================== Replay-buffer training loop (VCPO port) ====================

    async def _drain_queue_into_buffer(self) -> int:
        """Move everything currently in the transport queue into the replay
        buffer without blocking. Returns the number of groups added."""
        drained = await self.message_queue_client.get_available_samples()
        added = 0
        for raw in drained:
            if raw is None:
                self.rollout_done = True
                continue
            self.replay_buffer.add(ray.cloudpickle.loads(raw), self.current_param_version)
            added += 1
        return added

    async def _wait_one_sample_into_buffer(self) -> bool:
        """Block for one sample from the transport queue and add it to the
        buffer. Returns False when the termination sentinel arrived instead."""
        result = await self.message_queue_client.get_sample()
        if result is None:
            self.rollout_done = True
            return False
        sample, _ = result
        if sample is None:
            self.rollout_done = True
            return False
        self.replay_buffer.add(ray.cloudpickle.loads(sample), self.current_param_version)
        return True

    async def _acquire_replay_minibatch(self):
        """Compose the next mini-batch of groups from the replay buffer.

        Warm-up (first ceil(requires_mini_batches) updates): wait until
        mini_size *unseen* groups are buffered and use exactly those, oldest
        first. Steady state: pause only while the buffer holds fewer than
        requires_mini_batches x mini_size groups (fractional values allowed),
        then compose all unseen groups (oldest first, capped) plus a
        score-weighted sample of used ones. Returns (entries, info) or
        (None, None) when generation has finished and the buffer cannot
        support another mini-batch."""
        mini_size = self.required_samples
        watermark = self.replay_requires_mini_batches * mini_size
        await self._drain_queue_into_buffer()
        if self.replay_updates_done < self.replay_warmup_updates:
            while self.replay_buffer.new_count() < mini_size:
                if self.rollout_done:
                    print(
                        f"[FullyAsyncTrainer][Replay] rollout finished during warm-up with "
                        f"{self.replay_buffer.new_count()}/{mini_size} unseen groups; stopping"
                    )
                    return None, None
                await self._wait_one_sample_into_buffer()
            entries = self.replay_buffer.take_oldest_new(mini_size)
            info = {
                "n_new": mini_size,
                "n_replayed": 0,
                "staleness": [e.staleness(self.current_param_version) for e in entries],
            }
        else:
            while self.replay_buffer.size() < watermark:
                if self.rollout_done:
                    print(
                        f"[FullyAsyncTrainer][Replay] rollout finished with buffer "
                        f"{self.replay_buffer.size()} < watermark {watermark}; stopping"
                    )
                    return None, None
                await self._wait_one_sample_into_buffer()
            entries, info = self.replay_buffer.compose_minibatch(mini_size, self.current_param_version)
        # Open the virtual (no-validation-no-save) step: only the unseen
        # entries' arrival stamps gate this step — replayed groups were ready
        # long ago (a pure-replay mini-batch never waits on generation).
        consumer_end = time.time()
        self._open_virtual_step(consumer_end, [e.sample for e in entries if e.is_new])
        return entries, info

    def _build_replay_batch(self, entries):
        """Assemble a training DataProto from buffered groups using the frozen
        insertion-time statistics: advantages broadcast from advantage_scalar
        and the cached behavior log-probs as the off-policy reference (the
        seq_adv_post_scale loss anchors its ratio on the update's own forward,
        so old_log_probs is only an alias to satisfy field selection)."""
        rollout_samples = [e.sample for e in entries]
        balance = self._balance_batch if self.config.trainer.balance_batch else None
        batch = assemble_batch_from_rollout_samples(rollout_samples, self.tokenizer, self.config, balance)
        if "traj_uid" not in batch.non_tensor_batch and "uid" in batch.non_tensor_batch:
            uids = batch.non_tensor_batch.get("uid")
            batch.non_tensor_batch["traj_uid"] = np.array(
                [f"group-{uid}_traj-{idx}" for idx, uid in enumerate(uids)], dtype=object
            )
        response_mask = batch.batch["response_mask"]
        adv_scalars = torch.from_numpy(np.asarray(batch.non_tensor_batch["advantage_scalar"], dtype=np.float32))
        advantages = adv_scalars.unsqueeze(-1) * response_mask.float()
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = advantages
        # Sparse last-token rewards from the frozen insertion-time scalars, for
        # the data metrics (critic/score, critic/rewards) — not used by the loss.
        reward_scalars = torch.from_numpy(np.asarray(batch.non_tensor_batch["reward_scalar"], dtype=np.float32))
        token_level_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        lengths = response_mask.sum(dim=-1).long()
        valid = lengths > 0
        rows = torch.arange(response_mask.shape[0])[valid]
        token_level_scores[rows, (lengths[valid] - 1)] = reward_scalars[valid]
        batch.batch["token_level_scores"] = token_level_scores
        batch.batch["token_level_rewards"] = token_level_scores
        batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
        batch.meta_info["trainer_param_version"] = self.current_param_version
        if self.replay_ess_auto_base:
            # None until the first update's measurement is captured; the actor
            # skips LR scaling while the override is unresolved.
            batch.meta_info["ess_base_override"] = self.replay_ess_base
        return batch

    def _capture_ess_base(self, metrics):
        """Auto-calibration of ess_scaling.base_ess_ratio: capture the first
        update's measured ESS ratio (the staleness-0 warm-up mini-batch, i.e.
        the empirical on-policy rho_on) from the actor's structured
        staleness/ess entries. No-op once captured. The field matches
        ess_scaling.use_clipped so the reference and the scaling numerator
        measure the same quantity."""
        if self.replay_ess_base is not None:
            return
        key = "minibatch_ess_ratio_clipped" if self.replay_ess_use_clipped else "minibatch_ess_ratio"
        entries = metrics.get("staleness/ess") or []
        values = [float(e[key]) for e in entries if isinstance(e, dict) and e.get(key) is not None]
        if values:
            self.replay_ess_base = float(np.mean(values))
            print(
                f"[FullyAsyncTrainer][Replay] auto-calibrated ess_scaling.base_ess_ratio="
                f"{self.replay_ess_base:.4f} from the first update ({key})"
            )

    def _add_replay_metrics(self, metrics, info, new_version):
        """Replay/ESS metrics, computed after this update's eviction/rescoring
        at the post-update model version. The structured staleness/ess entries
        are consumed into scalars here; the raw staleness lists go to the
        tensorboard backend only (native histograms) via _fit_replay."""
        minibatch_staleness = info["staleness"]
        buffer_staleness = self.replay_buffer.staleness_list(new_version)
        metrics.update(
            {
                "replay/buffer_size": self.replay_buffer.size(),
                "replay/buffer_new": self.replay_buffer.new_count(),
                "replay/buffer_max_staleness": float(self.replay_buffer.max_staleness(new_version) or 0),
                "replay/minibatch_new": info["n_new"],
                "replay/minibatch_replayed": info["n_replayed"],
                "replay/minibatch_new_ratio": info["n_new"] / (info["n_new"] + info["n_replayed"]),
                "replay/minibatch_staleness_mean": float(np.mean(minibatch_staleness)),
                "replay/minibatch_staleness_max": float(np.max(minibatch_staleness)),
                "replay/evicted_cum": self.replay_buffer.evicted_total,
                "replay/evicted_unseen_cum": self.replay_buffer.evicted_unseen_total,
                "replay/total_added": self.replay_buffer.total_added,
            }
        )
        if buffer_staleness:
            metrics["replay/buffer_staleness_mean"] = float(np.mean(buffer_staleness))
        self._replay_hist_payload = {
            "replay/minibatch_staleness_hist": [float(s) for s in minibatch_staleness],
            "replay/buffer_staleness_hist": [float(s) for s in buffer_staleness],
        }
        # Consume the structured staleness/ess entries into dashboard scalars:
        # the effective (possibly ESS-braked) lr and the reference the actor
        # actually resolved and used this update.
        ess_entries = metrics.pop("staleness/ess", None) or []
        scaled_lrs = [
            float(e["ess_scaled_lr"]) for e in ess_entries if isinstance(e, dict) and e.get("ess_scaled_lr") is not None
        ]
        if scaled_lrs:
            metrics["replay/ess_scaled_lr"] = float(np.mean(scaled_lrs))
            # dashboard-parity alias with the source fork's actor-side scalar
            metrics["actor/ess_scaled_lr"] = float(np.mean(scaled_lrs))
        for src_key, dst_key in (
            ("minibatch_ess_ratio", "staleness/ess_ratio"),
            ("minibatch_ess_ratio_clipped", "staleness/ess_ratio_clipped"),
            ("base_ess_ratio", "staleness/base_ess_ratio"),
        ):
            values = [float(e[src_key]) for e in ess_entries if isinstance(e, dict) and e.get(src_key) is not None]
            if values:
                metrics[dst_key] = float(np.mean(values))
        if getattr(self, "replay_ess_base", None) is not None:
            metrics["replay/ess_base"] = self.replay_ess_base

    async def _replay_sync_weights(self, timing_raw):
        """Parameter sync after every replay update (trigger_parameter_sync_step=1)."""
        with marked_timer("param_sync", timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        staleness_timing = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        return staleness_timing

    async def _replay_maybe_validate(self, metrics):
        """Stop-the-world validation on the rollout GPUs at the test_freq
        cadence (in param-version units). The awaited window counts into
        cumulative_validation_time and is excluded from the virtual clock."""
        test_freq = self.config.trainer.test_freq
        if not (test_freq > 0 and self.current_param_version > 0 and self.current_param_version % test_freq == 0):
            return
        valid_start = time.time()
        val_metrics = await self.rollouter.do_validate.remote()
        valid_time = time.time() - valid_start
        self.cumulative_validation_time += valid_time
        self._step_valid_time += valid_time
        self.logger.log(data=val_metrics, step=self.current_param_version)

    async def _replay_maybe_save(self, timing_raw, force=False):
        """Checkpoint at the save_freq cadence (param-version units), optionally
        freezing generation for the whole save (pause_generation_during_save) so
        the save is a pure time translation of the pipeline."""
        save_freq = self.config.trainer.save_freq
        if self.current_param_version == self.last_ckpt_version:
            return
        if not (save_freq > 0 and (force or self.current_param_version % save_freq == 0)):
            return
        save_start = time.time()
        if self.pause_generation_during_save:
            await self.rollouter.begin_save_pause.remote()
        try:
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()
                local_global_step_folder = os.path.join(
                    self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
                )
                self._save_timing_state(local_global_step_folder, save_start)
                replay_path = os.path.join(local_global_step_folder, "replay_buffer.pt")
                torch.save(self._replay_checkpoint_state(), replay_path)
                print(
                    f"[FullyAsyncTrainer][Replay] Saved replay buffer "
                    f"({self.replay_buffer.size()} groups) to {replay_path}"
                )
                if self.save_queue_state:
                    self.message_queue_client.save_state_sync(local_global_step_folder)
                self.last_ckpt_version = self.current_param_version
        finally:
            if self.pause_generation_during_save:
                await self.rollouter.end_save_pause.remote()
        save_time = time.time() - save_start
        self.cumulative_save_time += save_time
        self._step_save_time += save_time

    def _replay_post_update_maintenance(self, entries):
        """Post-update buffer maintenance at the version this update just
        produced (stamped by the subsequent sync): retire the used groups'
        is_new flag, then evict too-stale groups and decay scores.
        mark_used runs BEFORE evict so a just-trained group past the staleness
        bound is not miscounted as evicted-unseen (wasted rollout compute)."""
        new_version = self.current_param_version + 1
        self.replay_buffer.mark_used(entries)
        self.replay_buffer.evict(new_version)
        self.replay_buffer.recompute_scores(new_version)
        self.replay_updates_done += 1
        return new_version

    def _replay_checkpoint_state(self) -> dict:
        return {
            "replay_buffer": self.replay_buffer.state_dict(),
            "replay_updates_done": self.replay_updates_done,
            "replay_ess_base": self.replay_ess_base,
        }

    def _load_replay_checkpoint_extras(self, global_step_folder):
        """Restore replay-buffer, ESS auto-base, timing offsets and queue state
        saved next to the actor checkpoint. Missing files degrade gracefully."""
        self._restore_timing_state(global_step_folder)
        if not self.replay_enable:
            return
        replay_path = os.path.join(global_step_folder, "replay_buffer.pt")
        if os.path.exists(replay_path):
            state = torch.load(replay_path, weights_only=False)
            self.replay_buffer.load_state_dict(state["replay_buffer"])
            self.replay_updates_done = int(state.get("replay_updates_done", 0))
            self.replay_ess_base = state.get("replay_ess_base", None)
            print(
                f"[FullyAsyncTrainer][Replay] Restored replay buffer "
                f"({self.replay_buffer.size()} groups, ess_base={self.replay_ess_base}) from {replay_path}"
            )
        else:
            print(f"[FullyAsyncTrainer][Replay] No replay_buffer.pt in {global_step_folder}; starting empty")
        if self.save_queue_state:
            self.message_queue_client.load_state_sync(global_step_folder)

    # ==================== Virtual clock (cumulative_training_time) ====================

    def _open_virtual_step(self, consumer_end: float, queue_samples: list):
        """Start this step on the virtual (no-validation-no-save) timeline: at
        max(trainer free, batch ready), where the batch is ready when its last
        sample would have arrived without the rollouter's validation and
        checkpoint-save pauses. Samples restored from an old-format queue
        snapshot may lack the stamps; fall back to the actual ready time (no
        pause correction) for them."""
        virtual_ready_times = [
            s.enqueue_time - s.validation_pause_before - getattr(s, "checkpoint_pause_before", 0.0)
            for s in queue_samples
            if getattr(s, "enqueue_time", 0.0)
        ]
        batch_virtual_ready = max(virtual_ready_times) if virtual_ready_times else consumer_end
        self._step_actual_start = consumer_end
        if self.virtual_free_time is not None:
            self._step_virtual_start = max(self.virtual_free_time, batch_virtual_ready)
        else:
            self._step_virtual_start = batch_virtual_ready

    def _virtual_now(self, now: float):
        """Current position on the virtual (no-validation-no-save) timeline.

        Mid-step, the step began at _step_virtual_start and has been busy for
        the actual elapsed time minus any awaited validation and checkpoint
        saving; between steps it is wherever the last step ended. None until
        the first batch arrives."""
        if self._step_virtual_start is not None:
            return (
                self._step_virtual_start
                + (now - self._step_actual_start)
                - self._step_valid_time
                - self._step_save_time
            )
        return self.virtual_free_time

    def _advance_virtual_clock(self, now: float = None):
        """Close the current step on the virtual timeline (called after the
        checkpoint save, whose duration _virtual_now excludes)."""
        if self._step_virtual_start is None:
            return
        self.virtual_free_time = self._virtual_now(time.time() if now is None else now)
        self._step_virtual_start = None
        self._step_actual_start = None

    def _add_cumulative_time_metrics(self, step_data: dict, now: float = None):
        """cumulative_training_time: the wall clock (since the rollouter's first
        processed sample) that an identical run with *neither validation nor
        checkpointing* would have needed to reach this point. Reconstructed by
        replaying the pipeline schedule on a virtual timeline: each step starts
        at max(trainer free, batch ready), with sample-ready times shifted back
        by the rollouter's validation and checkpoint-save pauses, and trainer
        stalls on awaited validation and checkpoint saving excluded from the
        busy time. Exact in rollout-bound, trainer-bound and balanced regimes
        alike (a naive wall - validation - save subtraction over-subtracts
        whenever validation overlaps saves or backlog training). No-op until
        the rollouter reports its first processed sample."""
        if self.rollouter_first_sample_time is None:
            return
        now = time.time() if now is None else now
        wall_time = now - self.rollouter_first_sample_time + self.timing_wall_offset
        validation_time = self.cumulative_validation_time + self.timing_validation_offset
        save_time = self.cumulative_save_time + self.timing_save_offset
        step_data["fully_async/timing/wall_time_since_first_sample"] = wall_time
        step_data["fully_async/timing/cumulative_validation_time"] = validation_time
        step_data["fully_async/timing/cumulative_save_time"] = save_time
        virtual_now = self._virtual_now(now)
        if virtual_now is not None:
            step_data["fully_async/timing/cumulative_training_time"] = (
                virtual_now - self.rollouter_first_sample_time + self.virtual_training_time_offset
            )

    def _save_timing_state(self, local_global_step_folder, save_start):
        """Persist the cumulative timing totals so a resumed run continues the
        fully_async/timing/* metrics instead of restarting them from zero.
        The snapshot is taken at save_start, so the in-progress save's own
        duration is excluded — it is exactly the state a resume reconstructs."""
        virtual_now = self._virtual_now(save_start)
        if self.rollouter_first_sample_time is not None:
            wall_time = save_start - self.rollouter_first_sample_time + self.timing_wall_offset
            validation_time = self.cumulative_validation_time + self.timing_validation_offset
            save_time = self.cumulative_save_time + self.timing_save_offset
            if virtual_now is not None:
                virtual_training_time = (
                    virtual_now - self.rollouter_first_sample_time + self.virtual_training_time_offset
                )
            else:
                virtual_training_time = self.virtual_training_time_offset
        else:
            wall_time = self.timing_wall_offset
            validation_time = self.timing_validation_offset
            save_time = self.timing_save_offset
            virtual_training_time = self.virtual_training_time_offset
        timing_state = {
            "wall_time_since_first_sample": wall_time,
            "cumulative_validation_time": validation_time,
            "cumulative_save_time": save_time,
            "cumulative_training_time": virtual_training_time,
        }
        with open(os.path.join(local_global_step_folder, "timing_state.json"), "w") as f:
            json.dump(timing_state, f, indent=2)

    def _restore_timing_state(self, global_step_folder):
        timing_state_path = os.path.join(global_step_folder, "timing_state.json")
        if not os.path.exists(timing_state_path):
            print("[FullyAsyncTrainer] No timing_state.json in checkpoint; timing metrics restart from zero")
            return
        with open(timing_state_path) as f:
            timing_state = json.load(f)
        self.timing_wall_offset = timing_state.get("wall_time_since_first_sample", 0.0)
        self.timing_validation_offset = timing_state.get("cumulative_validation_time", 0.0)
        self.timing_save_offset = timing_state.get("cumulative_save_time", 0.0)
        self.virtual_training_time_offset = timing_state.get(
            "cumulative_training_time",
            self.timing_wall_offset - self.timing_validation_offset - self.timing_save_offset,
        )
        print(
            f"[FullyAsyncTrainer] Restored timing state from {timing_state_path}: "
            f"cumulative_training_time resumes at {self.virtual_training_time_offset:.1f}s"
        )

    async def _fit_replay(self):
        """Replay-buffer training loop: one optimizer update per iteration,
        weight sync after every update, staleness-based eviction and score
        decay, warm-up on fresh groups. See replay_buffer.py for the buffer
        semantics."""
        self.max_steps_duration = 0
        timing_raw = {}
        while True:
            metrics = {}
            timing_raw = {}
            self._step_valid_time = 0.0
            self._step_save_time = 0.0
            if self.rollouter_first_sample_time is None:
                self.rollouter_first_sample_time = await asyncio.wrap_future(
                    self.rollouter.get_first_sample_time.remote().future()
                )

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    entries, info = await self._acquire_replay_minibatch()
                    if entries is None:
                        break
                    batch = self._build_replay_batch(entries)
                    self._collect_metrics_from_samples(batch, metrics)
                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self._update_actor(batch)
                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                if self.replay_ess_auto_base:
                    self._capture_ess_base(metrics)

            new_version = self._replay_post_update_maintenance(entries)
            self._add_replay_metrics(metrics, info, new_version)

            # Model versions tick once per UPDATE: sync, then validate/save at
            # the param-version cadence.
            self.current_param_version = new_version
            staleness_timing = await self._replay_sync_weights(timing_raw)
            metrics.update(staleness_timing)
            await self._replay_maybe_validate(metrics)
            await self._replay_maybe_save(timing_raw)

            metrics["training/global_step"] = self.global_steps
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            metrics.update({f"timing_s/{k}": float(v) for k, v in timing_raw.items()})
            self._add_cumulative_time_metrics(metrics)
            time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(
                f"[FullyAsyncTrainer][Replay] update: {self.replay_updates_done} "
                f"param_version: {self.current_param_version} "
                f"buffer: {self.replay_buffer.size()} "
                f"(new: {self.replay_buffer.new_count()}) {time_str}"
            )
            self.logger.log(data=metrics, step=self.current_param_version)
            hist_payload = getattr(self, "_replay_hist_payload", None)
            if hist_payload and "tensorboard" in self.config.trainer.logger:
                self.logger.log(data=hist_payload, step=self.current_param_version, backend=["tensorboard"])
            if self.progress_bar is not None:
                self.progress_bar.update(1)
            self.global_steps += 1
            self._advance_virtual_clock()

        # Final checkpoint (force) covers whatever the last cadence missed.
        await self._replay_maybe_save(timing_raw, force=True)
        print("[FullyAsyncTrainer][Replay] Training finished")
