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
from verl.experimental.fully_async_policy.dynamic_schedule import DynamicScheduleContext
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.fully_async_policy.replay_buffer import ReplayBuffer
from verl.experimental.fully_async_policy.replay_sizing import first_minibatch_groups, trainer_dp_size
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

logger = logging.getLogger(__name__)

# async_training keys of the custom_vcpo recipe that this port does not implement. Scripts may keep
# them only in their disabled form; an enabled value raises instead of being silently ignored.
_UNSUPPORTED_ASYNC_KEYS = {
    "dynamic_filtering.enable": False,
    "opportunistic_epochs.enable": False,
    "ppo_epochs": None,
    "save_queue_state": False,
    "replay_buffer.save_state": False,
    "resumable_ckpts_to_keep": None,
    "bsz_per_dp_rank": None,
}


def check_unsupported_async_keys(async_training) -> None:
    """Raise when an unsupported custom_vcpo async_training key is set to anything but its off value."""
    bad = []
    for key, off in _UNSUPPORTED_ASYNC_KEYS.items():
        value = OmegaConf.select(async_training, key, default=off)
        if value != off and not (off is None and value in (0, "null")):
            bad.append(f"async_training.{key}={value!r}")
    if bad:
        raise ValueError(
            f"Unsupported async_training settings: {', '.join(bad)} (not ported from custom_vcpo; "
            "use concurrent_samples_per_replica for bsz_per_dp_rank)"
        )


def parse_max_train_steps(value) -> int | None:
    """trainer.total_training_steps as a cap on optimizer updates: None/null = no cap."""
    if value is None:
        return None
    steps = int(value)
    if steps < 1:
        raise ValueError(f"trainer.total_training_steps must be >= 1 or null, got {value!r}")
    return steps


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
        # When use_trainer_do_validate OR use_dynamic_resource_scheduling is enabled, trainer
        # workers must carry a rollout engine (Role.ActorRollout) so that the master rank
        # can push weight updates directly to the colocated hybrid rollout instance via the
        # naive path (hybrid_checkpoint_manager).
        needs_hybrid_rollout = config.async_training.use_trainer_do_validate or config.async_training.get(
            "use_dynamic_resource_scheduling", False
        )
        self.train_role = Role.ActorRollout if needs_hybrid_rollout else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        check_unsupported_async_keys(config.async_training)
        # Cap on optimizer updates (verl's trainer.total_training_steps; null = the rollouter's prompt budget
        # alone ends the run). Reaching it forces a final checkpoint and ends the run.
        self.max_train_steps = parse_max_train_steps(config.trainer.get("total_training_steps", None))
        self.stopped_by_step_cap = False
        replay_cfg = config.async_training.get("replay_buffer", None)
        self.replay_enable = bool(replay_cfg.get("enable", False)) if replay_cfg else False
        if self.replay_enable:
            self._init_replay(config, replay_cfg)
        # Virtual timeline (fully_async/timing/cumulative_training_time): the wall clock an identical run
        # with neither validation nor checkpointing would have needed. Each step starts at max(trainer free,
        # its batch's virtual ready time from the rollouter's sample stamps) and advances by its busy time
        # minus the validation wait and checkpoint-save time.
        self.virtual_free_time = None
        self._step_virtual_start = None
        self._step_actual_start = None
        self._step_wait_valid_time = 0.0
        self._step_save_time = 0.0
        self.cumulative_save_time = 0.0
        self._step_wait_times: list[float] = []  # per-collection wait times within the current step (seconds)
        # Per-collection count of samples that actually had to be waited on (not
        # already sitting in the queue at collection start). Parallel to
        # _step_wait_times; see _get_samples_from_queue().
        self._step_wait_samples: list[int] = []
        # Hybrid GPUs (trainer-node GPUs that switch between rollout/train under dynamic
        # resource scheduling) and standalone GPUs (dedicated rollout-node GPUs, always
        # doing rollout). Used both for the existing throughput metric and to combine
        # dynamic_resource/{train,rollout}_resource_utilization into a single
        # dynamic_resource/resource_utilization metric (see MetricsAggregator).
        hybrid_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node
        standalone_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
        total_gpus = hybrid_gpus + standalone_gpus
        self.metrics_aggregator = MetricsAggregator(
            total_gpus=total_gpus, hybrid_gpus=hybrid_gpus, standalone_gpus=standalone_gpus
        )

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # Hybrid checkpoint manager for trainer-side validation (use_trainer_do_validate)
        # and/or dynamic resource scheduling (use_dynamic_resource_scheduling).
        # Uses naive backend to sync weights from trainer to hybrid rollout replicas.
        # Initialized in _setup_hybrid_checkpoint_manager() via set_rollouter().
        self.hybrid_checkpoint_manager = None

        # Dynamic resource controller — activated when use_dynamic_resource_scheduling=True.
        self.dynamic_resource_controller = None
        self.dynamic_schedule_enabled: bool = config.async_training.get("use_dynamic_resource_scheduling", False)
        # Name of the scheduling policy (resolved in _setup_dynamic_resource_controller).
        self._dynamic_schedule_policy_name: str = config.async_training.get("dynamic_schedule_policy", "default")
        # Initial deactivate_ratio forwarded to the policy constructor.
        self._dynamic_schedule_deactivate_ratio_init: float = config.async_training.get(
            "dynamic_schedule_deactivate_ratio", 0.3
        )
        # Whether to enable request rebalancing (abort + clear sticky cache +
        # resume) after hybrid replica activation. Default False.
        self._dynamic_schedule_enable_rebalance: bool = config.async_training.get(
            "dynamic_schedule_enable_rebalance", True
        )
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)

        # When standalone rollout resources are 0 (rollout.nnodes == 0), there are no
        # standalone replicas: all rollout happens on hybrid (trainer-side) GPUs.
        self.only_hybrid: bool = self.dynamic_schedule_enabled and config.rollout.nnodes == 0

        # Per-step dynamic scheduling context — built once at init, mutable fields updated each step.
        self.dynamic_schedule_ctx = DynamicScheduleContext(
            required_samples=self.required_samples,
            trigger_parameter_sync_step=self.trigger_parameter_sync_step,
            total_generated_samples=0,
            expected_samples=0,
            buffer_samples=0,
            only_hybrid=self.only_hybrid,
        )

    def _init_replay(self, config, replay_cfg):
        """Replay-buffer mode: the trainer keeps drained groups in a version-aware buffer and composes every
        optimizer mini-batch from it (fresh groups first, then a staleness/reuse-weighted draw), syncing
        weights after every update. See replay_buffer.py."""
        assert self.trigger_parameter_sync_step == 1, (
            "replay_buffer mode syncs weights after every update: set async_training.trigger_parameter_sync_step=1"
        )
        assert self.require_batches == 1, (
            "replay_buffer mode composes one mini-batch per update: set async_training.require_batches=1"
        )
        assert int(config.actor_rollout_ref.actor.ppo_epochs) == 1, "replay_buffer mode needs actor.ppo_epochs=1"
        assert str(config.algorithm.adv_estimator) == "grpo", (
            f"replay_buffer mode freezes GRPO group advantages at insertion; got {config.algorithm.adv_estimator}"
        )
        assert not config.algorithm.use_kl_in_reward, "replay_buffer mode does not support use_kl_in_reward"
        assert not self.use_critic, "replay_buffer mode does not support a critic"
        assert not config.async_training.get("use_dynamic_resource_scheduling", False), (
            "replay_buffer mode does not support use_dynamic_resource_scheduling"
        )
        # The update trains against the cached behavior log-probs (bypass mode). losses.py reads only
        # actor.policy_loss, so the worker-side keys must be set explicitly, not just algorithm.*.
        rollout_corr = config.algorithm.get("rollout_correction", None)
        assert rollout_corr is not None and rollout_corr.get("bypass_mode", False), (
            "replay_buffer mode trains against rollout log-probs: set algorithm.rollout_correction.bypass_mode=True"
        )
        policy_loss = config.actor_rollout_ref.actor.policy_loss
        assert policy_loss.get("loss_mode", None) == "bypass_mode", (
            "replay_buffer mode: set actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode "
            "(algorithm.rollout_correction.bypass_mode alone never reaches the workers)"
        )
        worker_corr = policy_loss.get("rollout_correction", None)
        assert worker_corr is not None and worker_corr.get("loss_type", None) == "reinforce", (
            "replay_buffer mode: pass +actor_rollout_ref.actor.policy_loss.rollout_correction="
            "'${algorithm.rollout_correction}' with algorithm.rollout_correction.loss_type=reinforce"
        )

        self.replay_requires_mini_batches = float(replay_cfg.get("requires_mini_batches", 1.0))
        assert self.replay_requires_mini_batches > 0, "replay_buffer.requires_mini_batches must be > 0"
        n = int(config.actor_rollout_ref.rollout.n)
        # requires_mini_batches in (0, 1) sizes only the FIRST mini-batch (rounded up so its sequences split
        # evenly over the trainer DP ranks); later mini-batches are full-size with the watermark at one
        # mini-batch. Values >= 1 are the pause watermark in mini-batches.
        self.replay_first_mini_size = first_minibatch_groups(
            self.replay_requires_mini_batches, self.required_samples, n, trainer_dp_size(config)
        )
        self.replay_min_fresh_ratio = float(replay_cfg.get("min_fresh_ratio", 0.0))
        assert 0.0 <= self.replay_min_fresh_ratio <= 1.0, (
            f"replay_buffer.min_fresh_ratio must be in [0, 1], got {self.replay_min_fresh_ratio}"
        )
        self.replay_min_fresh_wait_timeout_s = float(replay_cfg.get("min_fresh_wait_timeout_s", 3600.0))
        self.replay_fresh_poll_interval_s = 0.5
        reuse_halflife = replay_cfg.get("reuse_halflife", None)
        self.replay_buffer = ReplayBuffer(
            tau=float(replay_cfg.get("tau", 8.0)),
            staleness_threshold=int(replay_cfg.get("staleness_threshold", 32)),
            seed=int(replay_cfg.get("sampling_seed", 1234)),
            reuse_halflife=float(reuse_halflife) if reuse_halflife is not None else None,
        )
        self.replay_updates_done = 0
        self.rollout_done = False
        self._replay_fresh_wait_s = 0.0
        self._replay_fresh_floor_waived = 0
        print(
            f"[FullyAsyncTrainer][Replay] tau={self.replay_buffer.tau} k={self.replay_buffer.staleness_threshold} "
            f"reuse_halflife={self.replay_buffer.reuse_halflife} mini-batch={self.required_samples} groups, "
            f"first={self.replay_first_mini_size or self.required_samples}, "
            f"min_fresh_ratio={self.replay_min_fresh_ratio}"
        )

    async def _setup_checkpoint_manager(self):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=replicas
        )
        print(f"[FullyAsyncTrainer] Checkpoint manager initialized (backend={checkpoint_engine_config.backend})")

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
        needs_hybrid = self.config.async_training.use_trainer_do_validate or self.config.async_training.get(
            "use_dynamic_resource_scheduling", False
        )
        if not needs_hybrid:
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
            actor_wg=self.actor_rollout_wg,
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

    async def _setup_dynamic_resource_controller(self) -> None:
        """Initialise :class:`DynamicResourceController` with the configured policy.

        The policy is selected by ``async_training.dynamic_schedule_policy`` (a
        registered name string) or can be overridden by subclasses.  The
        ``only_hybrid`` flag is derived from the number of standalone replicas
        so the policy can adjust its behaviour accordingly.

        Pre-conditions:
          - ``self.hybrid_checkpoint_manager`` already set up.
          - ``self.rollouter`` is set.
        """
        from verl.experimental.fully_async_policy.dynamic_schedule import (
            DynamicResourceController,
            build_policy,
        )

        num_standalone = len(ray.get(self.rollouter.get_standalone_replicas.remote()))
        num_hybrid = len(ray.get(self.rollouter.get_all_hybrid_replicas.remote()))
        only_hybrid = num_standalone == 0

        policy = build_policy(
            self._dynamic_schedule_policy_name,
            deactivate_ratio=self._dynamic_schedule_deactivate_ratio_init,
            only_hybrid=only_hybrid,
        )
        print(
            f"[FullyAsyncTrainer] Dynamic scheduling policy '{self._dynamic_schedule_policy_name}' "
            f"instantiated (deactivate_ratio={self._dynamic_schedule_deactivate_ratio_init}, "
            f"only_hybrid={only_hybrid})"
        )

        self.dynamic_resource_controller = DynamicResourceController(
            rollouter=self.rollouter,
            hybrid_checkpoint_manager=self.hybrid_checkpoint_manager,
            num_standalone_replicas=num_standalone,
            num_hybrid_replicas=num_hybrid,
            policy=policy,
        )
        print(
            f"[FullyAsyncTrainer] DynamicResourceController initialised "
            f"(standalone={num_standalone}, hybrid={num_hybrid})"
        )

    async def set_rollouter(self, rollouter):
        """Set rollouter reference and initialize all checkpoint managers."""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        await self._setup_checkpoint_manager()
        await self._setup_hybrid_checkpoint_manager()
        # Setup dynamic resource controller if enabled
        if self.dynamic_schedule_enabled:
            await self._setup_dynamic_resource_controller()

    def set_total_train_steps(self, total_training_steps):
        if self.max_train_steps is not None:
            total_training_steps = min(total_training_steps, self.max_train_steps)
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
        # Snapshot the queue backlog at collection start: samples already sitting
        # in the queue are served instantly and don't reflect actual generation
        # rate, so they must be excluded from the wait-time-per-sample estimate.
        # Only queried when dynamic scheduling is enabled, since it's the sole consumer
        # of this signal and the extra RPC would otherwise be pure overhead.
        if self.dynamic_schedule_enabled:
            queue_size_at_start = await self.message_queue_client.get_queue_size()
            pending_wait_samples = max(0, self.required_samples - queue_size_at_start)
        else:
            pending_wait_samples = 0
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
            print(
                f"[FullyAsyncTrainer] sample collected {len(queue_samples)}/{self.required_samples}. "
                f"mq_len: {queue_len}",
                flush=True,
            )

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
        self._open_virtual_step(consumer_end, queue_samples)
        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        self._step_wait_times.append(total_wait_time)
        # pending_wait_samples may be 0 when this collection was served entirely
        # from queue backlog (no real waiting for generation happened); the policy
        # layer special-cases that when estimating the generation rate.
        self._step_wait_samples.append(pending_wait_samples)
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor — the role is Role.ActorRollout when use_trainer_do_validate or
        # use_dynamic_resource_scheduling is enabled (so the trainer worker also hosts a
        # local rollout engine for naive weight sync via hybrid_checkpoint_manager).
        # Otherwise it is Role.Actor.  Rollout capability is managed by ElasticAgentLoopManager's
        # hybrid replicas.
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

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while not self._train_step_cap_reached():
            try:
                if self.replay_enable:
                    await self._fit_replay_step()
                else:
                    await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            rollout_reset_timing_raw = await self._fit_update_weights()
            if rollout_reset_timing_raw is not None:
                self._fit_log_aggregated_training_metrics(rollout_reset_timing_raw)
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    def _train_step_cap_reached(self) -> bool:
        """True once trainer.total_training_steps optimizer updates have been done (global_steps starts at 1
        and counts updates, so it reads N + 1 after N)."""
        if self.max_train_steps is None or self.global_steps <= self.max_train_steps:
            return False
        if not self.stopped_by_step_cap:
            print(f"[FullyAsyncTrainer] reached trainer.total_training_steps={self.max_train_steps}; stopping")
            self.stopped_by_step_cap = True
        return True

    # ==================== replay-buffer training mode ====================

    async def _drain_queue_into_buffer(self) -> int:
        """Move everything currently in the message queue into the replay buffer without blocking. Returns
        the number of groups added; the termination sentinel sets rollout_done."""
        added = 0
        for raw in await self.message_queue_client.get_available_samples():
            if raw is None:
                self.rollout_done = True
                continue
            self.replay_buffer.add(ray.cloudpickle.loads(raw), self.current_param_version)
            added += 1
        return added

    async def _wait_one_sample_into_buffer(self) -> bool:
        """Block for one sample and add it to the buffer; False when the rollouter has finished instead."""
        result = await self.message_queue_client.get_sample()
        sample = result[0] if result is not None else None
        if sample is None:
            self.rollout_done = True
            return False
        self.replay_buffer.add(ray.cloudpickle.loads(sample), self.current_param_version)
        return True

    def _replay_minibatch_size(self) -> int:
        """Groups in the next composition: the smaller first mini-batch (requires_mini_batches < 1) once,
        then ppo_mini_batch_size."""
        if self.replay_first_mini_size is not None and self.replay_updates_done == 0:
            return int(self.replay_first_mini_size)
        return int(self.required_samples)

    def _replay_min_fresh_groups(self, mini_size: int) -> int:
        """The fresh-share floor in groups: ceil(min_fresh_ratio x mini_size); 0 when the gate is off."""
        if self.replay_min_fresh_ratio <= 0.0:
            return 0
        return min(mini_size, int(math.ceil(self.replay_min_fresh_ratio * mini_size - 1e-9)))

    async def _wait_for_fresh_floor(self, mini_size: int) -> tuple[float, int]:
        """Wait (polling the queue) until enough groups arrived since the previous composition, the rollouter
        is done, or the wall-clock cap is hit. Returns (seconds waited, 1 if the floor was waived).

        The wait is capped because the rollouter stops generating once its staleness quota is used up, and
        dropped degenerate groups count against it without ever reaching the buffer: after a reward collapse
        an uncapped wait could leave both sides waiting on each other."""
        floor = self._replay_min_fresh_groups(mini_size)
        if floor <= 0 or self.replay_buffer.pending_fresh_count() >= floor:
            return 0.0, 0
        start = time.time()
        announced = False
        while self.replay_buffer.pending_fresh_count() < floor:
            waited = time.time() - start
            if self.rollout_done:
                print(f"[FullyAsyncTrainer][Replay] rollout finished: fresh floor {floor} waived")
                return waited, 1
            timeout = self.replay_min_fresh_wait_timeout_s
            if timeout > 0.0 and waited >= timeout:
                print(
                    f"[FullyAsyncTrainer][Replay] WARNING: fresh floor {floor} not met after {waited:.0f}s "
                    f"({self.replay_buffer.pending_fresh_count()} fresh groups); composing anyway"
                )
                return waited, 1
            if await self._drain_queue_into_buffer() == 0:
                await asyncio.sleep(self.replay_fresh_poll_interval_s)
                if not announced and waited >= 60.0:
                    announced = True
                    print(
                        f"[FullyAsyncTrainer][Replay] waiting for fresh groups: "
                        f"{self.replay_buffer.pending_fresh_count()}/{floor} after {waited:.0f}s"
                    )
        return time.time() - start, 0

    async def _acquire_replay_minibatch(self):
        """Compose the next mini-batch of groups, or (None, None) once the rollouter has finished and the
        buffer cannot fill another one.

        Waits until the buffer holds the watermark (requires_mini_batches x mini_size groups, or one
        mini-batch when requires_mini_batches < 1), then for the fresh-share floor, then composes."""
        mini_size = self._replay_minibatch_size()
        if self.replay_requires_mini_batches < 1:
            watermark = mini_size
        else:
            watermark = self.replay_requires_mini_batches * mini_size
        await self._drain_queue_into_buffer()
        while self.replay_buffer.size() < watermark:
            if self.rollout_done:
                print(
                    f"[FullyAsyncTrainer][Replay] rollout finished with buffer {self.replay_buffer.size()} "
                    f"< watermark {watermark}; stopping"
                )
                return None, None
            await self._wait_one_sample_into_buffer()
        self._replay_fresh_wait_s, self._replay_fresh_floor_waived = await self._wait_for_fresh_floor(mini_size)
        entries, info = self.replay_buffer.compose_minibatch(mini_size, self.current_param_version)
        # replayed groups were ready long ago: only the fresh prefix's arrivals gate this step
        self._open_virtual_step(time.time(), [e.sample for e in entries[: info["n_new"]]])
        return entries, info

    def _replay_post_update_maintenance(self, entries, new_version: int) -> None:
        """After an update, at the version it produced: count the trainings BEFORE evicting (a just-trained
        group crossing the threshold is not generated-but-never-trained waste), evict, rescore."""
        self.replay_buffer.mark_trained(entries)
        self.replay_buffer.evict(new_version)
        self.replay_buffer.recompute_scores(new_version)

    def _build_replay_batch(self, entries) -> DataProto:
        """Training batch from buffered groups with the insertion-time statistics frozen by the rollouter:
        advantages broadcast from advantage_scalar, sparse token-level rewards from reward_scalar, and the
        cached rollout log-probs as the behavior policy (bypass mode) -- nothing is recomputed."""
        balance = self._balance_batch if self.config.trainer.balance_batch else None
        batch = assemble_batch_from_rollout_samples([e.sample for e in entries], self.tokenizer, self.config, balance)
        response_mask = batch.batch["response_mask"]
        adv = torch.from_numpy(np.asarray(batch.non_tensor_batch["advantage_scalar"], dtype=np.float32))
        advantages = adv.unsqueeze(-1) * response_mask.float()
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = advantages
        rewards = torch.from_numpy(np.asarray(batch.non_tensor_batch["reward_scalar"], dtype=np.float32))
        scores = torch.zeros_like(response_mask, dtype=torch.float32)
        lengths = response_mask.sum(dim=-1).long()
        valid = lengths > 0
        scores[torch.arange(response_mask.shape[0])[valid], lengths[valid] - 1] = rewards[valid]
        batch.batch["token_level_scores"] = scores
        batch.batch["token_level_rewards"] = scores
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _add_replay_metrics(self, metrics: dict, info: dict, new_version: int) -> None:
        """replay/* metrics after this update's maintenance, at the version it produced."""
        buf = self.replay_buffer
        staleness = info["staleness"]
        n_new = int(info["n_new"])
        size = n_new + int(info["n_replayed"])
        buffer_staleness = buf.staleness_list(new_version)
        times_trained = list(info.get("times_trained", []))
        buffer_times_trained = buf.times_trained_list()
        metrics.update(
            {
                "replay/buffer_size": buf.size(),
                "replay/buffer_untrained": buf.untrained_count(),
                "replay/buffer_max_staleness": float(buf.max_staleness(new_version) or 0),
                "replay/minibatch_size": size,
                "replay/minibatch_new": n_new,
                "replay/minibatch_replayed": size - n_new,
                "replay/minibatch_new_ratio": n_new / size,
                "replay/minibatch_staleness_mean": float(np.mean(staleness)),
                "replay/minibatch_staleness_max": float(np.max(staleness)),
                "replay/minibatch_staleness_p50": float(np.percentile(staleness, 50)),
                "replay/evicted_cum": buf.evicted_total,
                "replay/evicted_unseen_cum": buf.evicted_unseen_total,
                "replay/evicted_trained_once_cum": buf.evicted_trained_once_total,
                "replay/total_added": buf.total_added,
                "replay/updates_done": self.replay_updates_done,
                "replay/fresh_floor": float(self._replay_min_fresh_groups(size)),
                "replay/fresh_wait_s": float(self._replay_fresh_wait_s),
                "replay/fresh_floor_waived": float(self._replay_fresh_floor_waived),
            }
        )
        if buffer_staleness:
            metrics["replay/buffer_staleness_mean"] = float(np.mean(buffer_staleness))
            metrics["replay/buffer_staleness_p50"] = float(np.percentile(buffer_staleness, 50))
        fresh = list(info.get("fresh_staleness", staleness[:n_new]))
        replayed = list(staleness[n_new:])
        if fresh:
            metrics["replay/minibatch_fresh_staleness_mean"] = float(np.mean(fresh))
            metrics["replay/minibatch_fresh_staleness_max"] = float(np.max(fresh))
        if replayed:
            metrics["replay/minibatch_replayed_staleness_mean"] = float(np.mean(replayed))
        if times_trained:
            metrics["replay/minibatch_times_trained_mean"] = float(np.mean(times_trained))
            metrics["replay/minibatch_times_trained_max"] = float(np.max(times_trained))
        if buffer_times_trained:
            metrics["replay/buffer_times_trained_mean"] = float(np.mean(buffer_times_trained))

    async def _fit_replay_step(self):
        """One replay-mode update: compose a mini-batch from the buffer, update the actor on it against the
        cached rollout log-probs, maintain the buffer at the new version, sync weights, validate, save."""
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        self.reward_extra_infos_dict = {}
        self._step_wait_valid_time = 0.0
        self._step_save_time = 0.0

        steps = self.config.global_profiler.steps
        should_profile = steps is not None and (self.current_param_version + 1) in steps
        self._fit_start_profile(should_profiler=should_profile)

        with marked_timer("step", self.timing_raw):
            with marked_timer("gen", self.timing_raw, color="red"):
                entries, info = await self._acquire_replay_minibatch()
                if entries is None:
                    raise TrainingStopException("Training terminated: rollout finished and replay buffer drained")
                batch = self._build_replay_batch(entries)
                self._collect_metrics_from_samples(batch, self.metrics)
            _allocated_start = time.time()
            batch = self._fit_compute_log_prob(batch)  # bypass mode: old_log_probs = rollout_log_probs
            with marked_timer("update_actor", self.timing_raw, color="red"):
                # the batch is exactly one optimizer mini-batch, also when it is the smaller first one
                actor_output = self._update_actor(batch, mini_batch_size=len(batch))
            self.metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
            self._fit_update_local_step()
            self.replay_updates_done += 1
            self._replay_post_update_maintenance(entries, self.current_param_version)
            self._add_replay_metrics(self.metrics, info, self.current_param_version)
            rollout_reset_timing_raw = await self._fit_update_weights()
            self._fit_dump_data(batch)
            self._record_train_resource_utilization(allocated_time=time.time() - _allocated_start)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._advance_virtual_clock()
        self._fit_stop_profile(should_profiler=should_profile)
        self._fit_collect_metrics(batch)
        # add this update's metrics before logging the cycle, so they land at the version it produced
        self._fit_postprocess_step()
        if rollout_reset_timing_raw is not None:
            self._fit_log_aggregated_training_metrics(rollout_reset_timing_raw)

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
        self._step_wait_valid_time = 0.0
        self._step_save_time = 0.0
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        steps = self.config.global_profiler.steps
        should_profile = steps is not None and (self.current_param_version + 1) in steps
        self._fit_start_profile(should_profiler=should_profile)

        with marked_timer("step", self.timing_raw):
            ctrl = self.dynamic_resource_controller
            if self.dynamic_schedule_enabled and ctrl.policy.should_deactivate(
                global_steps=self.current_param_version,
                is_hybrid_active=ctrl.is_hybrid_active,
                ctx=self.dynamic_schedule_ctx,
            ):
                threshold_samples = ctrl.policy.deactivate_wait_samples(self.dynamic_schedule_ctx)
                with marked_timer("wait_for_enough_samples", self.timing_raw):
                    _ = ray.get(self.rollouter.wait_for_enough_samples.remote(threshold_samples))
                _deact_start = time.time()
                await ctrl.deactivate_hybrid_replicas(self.current_param_version)
                deactivate_duration = time.time() - _deact_start
                self.dynamic_schedule_ctx.last_deactivate_duration_s += deactivate_duration
                print(
                    f"[FullyAsyncTrainer] step={self.current_param_version} "
                    f"deactivation took {deactivate_duration:.2f}s "
                    f"(accumulated this cycle: {self.dynamic_schedule_ctx.last_deactivate_duration_s:.2f}s)",
                    flush=True,
                )
            elif self.dynamic_schedule_enabled:
                # Not deactivating this step: still emit the metric as 0 so the
                # timing_s/wait_for_enough_samples curve stays continuous.
                self.timing_raw["wait_for_enough_samples"] = 0.0

            _allocated_start = time.time()
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            rollout_reset_timing_raw = await self._fit_update_weights()
            self._fit_dump_data(batch)
            self._record_train_resource_utilization(allocated_time=time.time() - _allocated_start)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._advance_virtual_clock()
        self._fit_stop_profile(should_profiler=should_profile)
        self._fit_collect_metrics(batch)
        if rollout_reset_timing_raw is not None:
            self._fit_log_aggregated_training_metrics(rollout_reset_timing_raw)
        self._fit_postprocess_step()

    # Timing-raw keys that represent actual training-GPU compute, from
    # _fit_compute_reward() through _fit_update_actor(). Some keys may be
    # absent for a given step (e.g. "values"/"update_critic" when
    # use_critic=False, or "old_log_prob" under rollout_correction bypass
    # mode), so callers must default missing keys to 0.0.
    _TRAIN_COMPUTE_TIMING_KEYS = (
        "reward",
        "old_log_prob",
        str(Role.RefPolicy),
        "values",
        "adv",
        "update_critic",
        "update_actor",
    )

    def _record_train_resource_utilization(self, allocated_time: float) -> None:
        """Record raw (unratioed) numerator/denominator seconds for train-resource utilization.

        Numerator: time spent on actual training-GPU compute, i.e. the sum of
        the timing_raw entries from _fit_compute_reward() through
        _fit_update_actor() (reward, old_log_prob, ref, values, adv,
        update_critic, update_actor).

        Denominator: wall-clock time allocated to this fit_step()'s "training
        turn", i.e. from _fit_generate() through _fit_update_weights() and
        _fit_dump_data() (includes timing_s/param_sync and any hybrid
        activation — these are intentionally NOT subtracted).

        Both quantities are logged as raw seconds (not a ratio) so that
        MetricsAggregator can sum them across all micro-steps in a sync
        cycle first, and the ratio is computed once from the summed totals
        in _special_metrics_aggergate() (see "dynamic_resource/train_resource_utilization").
        """
        train_compute_time = sum(self.timing_raw.get(key, 0.0) for key in self._TRAIN_COMPUTE_TIMING_KEYS)
        self.metrics["dynamic_resource/train_compute_time_s"] = train_compute_time
        self.metrics["dynamic_resource/train_allocated_time_s"] = allocated_time

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

    async def _fit_update_weights(self) -> dict | None:
        """Sync updated weights to rollout replicas.

        Returns:
            The timing_raw dict returned by the rollouter's reset_staleness() (contains
            dynamic_resource/rollout_resource_utilization, used by
            _fit_log_aggregated_training_metrics()) if weights were actually updated this
            call, or None if this call was a no-op (not the last local_trigger_step). Callers
            should treat "weights were updated" and "return value is not None" as equivalent.
        """
        if self.local_trigger_step != 1:
            return None

        steps = self.config.global_profiler.steps
        last_profiler_step = self.current_param_version
        if steps is not None and last_profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._stop_profiling.remote().future())

        _total_generated_samples, _completed_steps = ray.get(
            [self.rollouter.get_total_produced_samples.remote(), self.rollouter.get_completed_steps.remote()]
        )
        _expect_samples = self.dynamic_schedule_ctx.step_required_samples * _completed_steps
        _buffer_sampels = self.dynamic_schedule_ctx.step_required_samples * self.staleness_threshold

        if self.dynamic_schedule_enabled:
            ctrl = self.dynamic_resource_controller
            # Update per-step mutable fields on the persistent context.
            ctx = self.dynamic_schedule_ctx
            ctx.total_generated_samples = _total_generated_samples
            ctx.expected_samples = _expect_samples
            ctx.buffer_samples = _buffer_sampels
            ctx.step_wait_times = list(self._step_wait_times)
            ctx.step_wait_samples = list(self._step_wait_samples)
            should_activate = ctrl.policy.should_activate_after_step(
                global_steps=self.current_param_version,
                is_hybrid_active=ctrl.is_hybrid_active,
                ctx=ctx,
            )

        with marked_timer("timing_s/param_sync", self.timing_raw):
            # Step 1: NCCL broadcast from trainer to standalone rollout replicas.
            # Skipped when there are no standalone replicas (e.g. rollout.nnodes=0,
            # all rollout is hybrid) -- there is nothing to sync weights to.
            if not self.only_hybrid:
                await self.checkpoint_manager.update_weights(
                    global_steps=self.current_param_version,
                )
            # Step 2: When dynamic resource scheduling is enabled, the Trainer GPUs
            # also co-host hybrid rollout replicas.  Push weights to them via
            # a separate naive sync (same mechanism as colocated training).
            if self.dynamic_schedule_enabled and should_activate:
                _act_start = time.time()
                await self.dynamic_resource_controller.sync_hybrid_weights(
                    global_steps=self.current_param_version,
                )
                await self.dynamic_resource_controller.activate_hybrid_replicas(self.current_param_version)

                # Allow policy to redistribute requests across newly activated replicas.
                if self._dynamic_schedule_enable_rebalance:
                    self.dynamic_resource_controller.policy.request_rebalance(
                        global_steps=self.current_param_version,
                        ctx=ctx,
                    )

                self.dynamic_schedule_ctx.last_activate_duration_s += time.time() - _act_start

        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())

        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        profiler_step = last_profiler_step + 1

        if steps is not None and profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._start_profiling.remote().future())

        if self.dynamic_schedule_enabled:
            # Let the policy update its internal state (e.g. adapt deactivate_ratio).
            self.dynamic_resource_controller.policy.update_after_step(
                global_steps=self.current_param_version,
                ctx=ctx,
            )
            # Now that update_after_step() has consumed this cycle's switch timing,
            # reset it so it doesn't leak into the next sync cycle.
            self.dynamic_schedule_ctx.last_deactivate_duration_s = 0.0
            self.dynamic_schedule_ctx.last_activate_duration_s = 0.0

        self._step_wait_times = []  # reset for next step
        self._step_wait_samples = []  # reset for next step

        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        return timing_raw

    # ==================== virtual (no-validation, no-checkpoint) timeline ====================

    def _open_virtual_step(self, consumer_end: float, samples: list) -> None:
        """Start a step on the virtual timeline at max(trainer free, batch virtual-ready time), where a
        sample is virtually ready at enqueue_time minus the rollouter's validation and save pauses before
        it. Without stamped samples (a pure-replay mini-batch, whose groups were all ready long ago) the
        step starts when the trainer is free; before any step, at the actual time."""
        ready = [
            s.enqueue_time - s.validation_pause_before - s.checkpoint_pause_before
            for s in samples
            if getattr(s, "enqueue_time", None) is not None
        ]
        if ready:
            batch_ready = max(ready)
        else:
            batch_ready = self.virtual_free_time if self.virtual_free_time is not None else consumer_end
        self._step_actual_start = consumer_end
        self._step_virtual_start = (
            max(self.virtual_free_time, batch_ready) if self.virtual_free_time is not None else batch_ready
        )

    def _virtual_now(self, now: float):
        """Position on the virtual timeline: mid-step, the step's virtual start plus its busy time so far
        (validation wait and checkpoint saving excluded); between steps, where the last step ended."""
        if self._step_virtual_start is not None:
            busy = (now - self._step_actual_start) - self._step_wait_valid_time - self._step_save_time
            return self._step_virtual_start + busy
        return self.virtual_free_time

    def _advance_virtual_clock(self, now: float = None) -> None:
        """Close the current step on the virtual timeline (after its checkpoint save)."""
        if self._step_virtual_start is None:
            return
        self.virtual_free_time = self._virtual_now(time.time() if now is None else now)
        self._step_virtual_start = None
        self._step_actual_start = None

    def _add_cumulative_time_metrics(self, data: dict, rollouter_timing: dict, now: float = None) -> None:
        """fully_async/timing/*: wall time since the rollouter's first training draw, the cumulative
        validation and checkpoint-save times, and cumulative_training_time, the wall clock an identical run
        with neither validation nor checkpointing would have needed (the virtual timeline). No-op until the
        first training sample was drawn. Exact (up to the re-prefill of requests aborted by the freeze) only with
        async_training.serialize_validation; otherwise in-flight requests keep decoding during validation and
        cumulative_training_time is biased low (see FullyAsyncRollouter.do_validate)."""
        first = rollouter_timing.get("first_sample_time")
        if first is None:
            return
        now = time.time() if now is None else now
        data["fully_async/timing/wall_time_since_first_sample"] = now - first
        data["fully_async/timing/cumulative_validation_time"] = rollouter_timing["cumulative_validation_time"]
        data["fully_async/timing/cumulative_checkpoint_pause"] = rollouter_timing["cumulative_checkpoint_pause"]
        data["fully_async/timing/cumulative_save_time"] = self.cumulative_save_time
        virtual_now = self._virtual_now(now)
        if virtual_now is not None:
            data["fully_async/timing/cumulative_training_time"] = virtual_now - first

    def _fit_log_aggregated_training_metrics(self, rollout_reset_timing_raw: dict):
        """Log aggregated training metrics for the sync cycle just finished.

        Args:
            rollout_reset_timing_raw: timing_raw dict returned by _fit_update_weights()
                (i.e. the rollouter's reset_staleness()). rollout_resource_utilization is
                computed on the rollouter side and passed in here (rather than flowing
                through add_step_metrics()) so it can be combined with
                dynamic_resource/train_resource_utilization into
                dynamic_resource/resource_utilization -- see
                MetricsAggregator._special_metrics_aggergate().
        """
        aggregated_metrics = self.metrics_aggregator.get_aggregated_metrics(
            rollout_resource_utilization=rollout_reset_timing_raw.get("dynamic_resource/rollout_resource_utilization"),
        )
        self._add_cumulative_time_metrics(aggregated_metrics, ray.get(self.rollouter.get_timing_state.remote()))
        if aggregated_metrics:
            self.logger.log(
                data=aggregated_metrics,
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
        validate_start = time.time()
        if self.config.async_training.use_trainer_do_validate:
            await self._trainer_side_validate()
        else:
            val_metrics = await self.rollouter.do_validate.remote()
            self.logger.log(data=val_metrics, step=self.current_param_version)
        # the trainer idles on validation: not part of the no-validation timeline's busy time
        self._step_wait_valid_time += time.time() - validate_start

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
            # saving is not training: excluded from the virtual timeline's busy time
            self.cumulative_save_time += timing_raw["save_checkpoint"]
            self._step_save_time += timing_raw["save_checkpoint"]

    def _fit_postprocess_step(self):
        self.global_steps += 1

        # Snapshot of samples left in the message queue for subsequent steps, taken right
        # after this fit_step() (including any partial-rollout resumption) has fully
        # finished. Registered under the "last" aggregation rule (see MetricsAggregator)
        # so the value reported per sync-cycle is this end-of-cycle snapshot rather than
        # an average across the cycle's micro-steps.
        self.metrics["dynamic_resource/mq_size"] = self.message_queue_client.get_queue_size_sync()

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        if not self.config.async_training.get("pause_generation_during_save", False):
            return self._save_checkpoint_inner()
        # Stop-the-world save: generation is frozen for the whole save, so the save is a pure time
        # translation of the pipeline instead of letting the queue build a surplus meanwhile.
        ray.get(self.rollouter.begin_save_pause.remote())
        try:
            self._save_checkpoint_inner()
        finally:
            ray.get(self.rollouter.end_save_pause.remote())

    def _save_checkpoint_inner(self):
        # Snapshot of the timing totals for timing_state.json, taken before anything is written so the
        # save's own duration is excluded (as from cumulative_training_time).
        save_start = time.time()
        rollouter_timing = ray.get(self.rollouter.get_timing_state.remote())
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
        self._save_timing_state(local_global_step_folder, save_start, rollouter_timing)
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    def _save_timing_state(self, local_global_step_folder: str, save_start: float, rollouter_timing: dict) -> None:
        """Write global_step_N/timing_state.json: the fully_async/timing/* totals at the start of this save
        (same keys as the custom_vcpo source), so every checkpoint records the training time it took to reach.

        cumulative_training_time is the virtual-timeline value (the wall clock a run with neither validation
        nor checkpointing would have needed), matching the metric logged at this step. All totals are 0 before
        the rollouter's first training draw. Informational only: a resume does not read it back, so a resumed
        run restarts the timing metrics at zero."""
        first = rollouter_timing.get("first_sample_time")
        if first is not None:
            virtual_now = self._virtual_now(save_start)
            wall_time = save_start - first
            validation_time = rollouter_timing["cumulative_validation_time"]
            save_time = self.cumulative_save_time
            training_time = virtual_now - first if virtual_now is not None else 0.0
        else:
            wall_time = validation_time = save_time = training_time = 0.0
        fmt = "%Y-%m-%d %H:%M:%S"
        timing_state = {
            "wall_time_since_first_sample": wall_time,
            "cumulative_validation_time": validation_time,
            "cumulative_save_time": save_time,
            "cumulative_training_time": training_time,
            # human-readable anchors (local time)
            "first_sample_datetime": datetime.fromtimestamp(first).strftime(fmt) if first is not None else None,
            "checkpoint_saved_datetime": datetime.fromtimestamp(save_start).strftime(fmt),
        }
        os.makedirs(local_global_step_folder, exist_ok=True)
        with open(os.path.join(local_global_step_folder, "timing_state.json"), "w") as f:
            json.dump(timing_state, f, indent=2)

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
