# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
This trainer supports model-agonistic model initialization with huggingface
"""

import asyncio
import os
import uuid
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.vllm.batch_stats import build_cdf_figure, counter_to_histogram_raw
from verl.workers.rollout.llm_server import LLMServerManager


def _env_flag(name: str) -> bool:
    """Parse a boolean env var. True only for "1"/"true"/"yes" (case-insensitive); unset, empty,
    "0", "false", etc. are False. Avoids the ``bool(os.environ.get(...))`` pitfall where the string
    "0" is truthy."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


class OneStepOffRayTrainer(SeparateRayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        # Optional diagnostic: per-GPU, per-rollout vLLM forward batch-size histograms.
        # Enabled by setting VERL_ROLLOUT_BATCH_STATS=1; off when unset/0/false.
        self._rollout_batch_stats = _env_flag("VERL_ROLLOUT_BATCH_STATS")

        # Optional: over-sample + discard-slow-tail generation. With data.gen_batch_size =
        # k * train_batch_size, generate k*train_batch_size prompt-groups and keep only the fastest
        # train_batch_size complete groups, aborting the slow long-tail. Trades wasted generation
        # FLOPs and a short-completion selection bias for lower per-step generation latency.
        # Enabled by VERL_OVERSAMPLE_DISCARD=1; off when unset/0/false.
        self._oversample_discard = _env_flag("VERL_OVERSAMPLE_DISCARD")

        # Skip rollout worker mapping and let agentloop create it.
        role_worker_mapping.pop(Role.Rollout, None)
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)

        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

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

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== SeparateRayPPOTrainer config ====================

        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.logger = None
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

    def _create_actor_rollout_classes(self):
        for role in [Role.Actor]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    def _init_async_rollout_manager(self):
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        self.llm_server_manager = LLMServerManager.create(config=self.config)
        self.async_rollout_mode = True
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        if self._rollout_batch_stats:
            # Install the (dormant) per-forward batch-size recorder on every rollout worker.
            self.llm_server_manager.init_batch_stats()

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _async_gen_next_batch(self, continuous_iterator):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in async_gen_next_batch: {e}")
            return None

        metrics = {}
        timing_raw = {}

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        if self._oversample_discard:
            # Keep only the fastest train_batch_size complete groups; discard the rest. gen_batch
            # holds the over-sampled k*train_batch_size prompts (via data.gen_batch_size).
            gen_batch_output.meta_info["keep_complete_groups"] = self.config.data.train_batch_size

        # async generation
        if self._rollout_batch_stats:
            # Bracket this rollout so each generation GPU records a fresh forward
            # batch-size histogram; tag it with the step this batch is generated at
            # (one-step-off: it is consumed one step later).
            gen_step = self.global_steps
            await self.llm_server_manager.reset_batch_stats()
        with marked_timer("generate_async", timing_raw, color="purple"):
            gen_batch_output = await self.async_rollout_manager.generate_sequences(gen_batch_output)
        if self._oversample_discard:
            # Flush the discarded long-tail's in-flight vLLM requests, then re-arm the engine
            # (abort_all_requests leaves the engine paused on vLLM >= 0.12).
            await self.llm_server_manager.abort_all_requests()
            await self.llm_server_manager.resume_generation()
        if self._rollout_batch_stats:
            await self._log_rollout_batch_stats(gen_step)

        if self._oversample_discard:
            # gen_batch_output holds only the surviving train_batch_size groups (uid attached).
            # Select the matching prompts from the over-sampled batch, in surviving-group order,
            # so the positional union below aligns each rollout with its prompt.
            ordered_uids = list(dict.fromkeys(gen_batch_output.non_tensor_batch["uid"].tolist()))
            uid_to_row = {uid: row for row, uid in enumerate(batch.non_tensor_batch["uid"])}
            batch = batch.select_idxs([uid_to_row[uid] for uid in ordered_uids])
            # Drop uid from the generation output to avoid a union key conflict (batch supplies uid).
            gen_batch_output.non_tensor_batch.pop("uid", None)

        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # Launch individual reward computations as each generation completes
        future_reward = None

        # Return the original, now-modified `batch` and the `future_reward`
        return metrics, timing_raw, epoch, batch, future_reward

    async def _log_rollout_batch_stats(self, gen_step: int):
        """Collect per-GPU forward batch-size histograms for the just-finished rollout
        and log them to TensorBoard (HISTOGRAMS tab), keyed by the generation step."""
        try:
            per_gpu = await self.llm_server_manager.collect_batch_stats()
        except Exception as e:
            print(f"Failed to collect rollout batch stats: {e}")
            return
        if not per_gpu or self.logger is None:
            return
        hist_data = {}
        figure_data = {}
        scalar_data = {}
        tokens_generated_total = 0
        for stats in per_gpu:
            gpu_id = stats["gpu_id"]
            # Histograms over batch size: number of forward calls, and total forward time.
            decode_hist, decode_calls = counter_to_histogram_raw(stats["decode"])
            total_hist, total_calls = counter_to_histogram_raw(stats["total"])
            decode_time_hist, _ = counter_to_histogram_raw(stats["decode_time"])
            total_time_hist, forward_time_total = counter_to_histogram_raw(stats["total_time"])
            if decode_hist is not None:
                hist_data[f"rollout_batch/decode/{gpu_id}"] = decode_hist
            if total_hist is not None:
                hist_data[f"rollout_batch/total/{gpu_id}"] = total_hist
            if decode_time_hist is not None:
                hist_data[f"rollout_batch/decode_time/{gpu_id}"] = decode_time_hist
            if total_time_hist is not None:
                hist_data[f"rollout_batch/total_time/{gpu_id}"] = total_time_hist
            # Cumulative (CDF) line plots: fraction of forward calls / forward time at batch
            # sizes <= x (rises to 1). Logged as matplotlib figures to the IMAGES tab.
            calls_fig = build_cdf_figure(
                f"forward-call CDF - {gpu_id}",
                "batch size",
                [("decode", stats["decode"]), ("total", stats["total"])],
            )
            time_fig = build_cdf_figure(
                f"forward-time CDF - {gpu_id}",
                "batch size",
                [("decode_time", stats["decode_time"]), ("total_time", stats["total_time"])],
            )
            if calls_fig is not None:
                figure_data[f"rollout_batch/calls_cdf/{gpu_id}"] = calls_fig
            if time_fig is not None:
                figure_data[f"rollout_batch/time_cdf/{gpu_id}"] = time_fig
            scalar_data[f"rollout_batch/decode_calls/{gpu_id}"] = decode_calls
            scalar_data[f"rollout_batch/total_calls/{gpu_id}"] = total_calls
            # Total forward wall time (seconds); also the normalizer for the time histograms.
            scalar_data[f"rollout_batch/forward_time_total/{gpu_id}"] = forward_time_total
            # Tokens generated on this GPU = sum over forwards of (#sequences decoding a token),
            # i.e. the decode-count histogram weighted by batch size.
            gpu_tokens = sum(batch_size * calls for batch_size, calls in stats["decode"].items())
            scalar_data[f"rollout_batch/tokens_generated/{gpu_id}"] = gpu_tokens
            tokens_generated_total += gpu_tokens
        # Total tokens generated this generation phase (summed across all rollout GPUs).
        scalar_data["rollout_batch/tokens_generated_total"] = tokens_generated_total
        self.logger.log_histogram_raw(hist_data, step=gen_step)
        self.logger.log_figure(figure_data, step=gen_step)
        self.logger.log(scalar_data, step=gen_step)

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(batch, config, tokenizer):
        reward_tensor, reward_extra_info = extract_reward(batch)
        return reward_tensor, reward_extra_info

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        # Reset the training-time clock; _load_checkpoint restores train_time_s on resume.
        self.train_time_s = 0.0
        self._train_clock_mark = None

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self._fit_update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            # Give the pre-training validation point a training-time coordinate (0.0, or the
            # restored clock on resume) so score-vs-train-time plots include it.
            val_metrics["training/train_time_s"] = self.train_time_s
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.last_val_metrics = None
        self.max_steps_duration = 0

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()
        # Training-time clock starts here: the task below immediately samples from the train set.
        self._train_clock_start()
        # Start the first asynchronous generation task.
        batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
        while batch_data_future is not None:
            batch_data_future = await self.fit_step(batch_data_future, continuous_iterator)
            if self.is_last_step:
                return

    async def fit_step(self, batch_data_future, continuous_iterator):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_data_future: batch future
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_prepare_step()
        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch, batch_data_future = await self._fit_generate(batch_data_future, continuous_iterator)

            # await asyncio.sleep(0) ensures:
            # Asynchronous tasks can start executing immediately
            # The event loop can handle other pending coroutines
            # Prevents computations in a certain phase from blocking the entire asynchronous workflow
            #
            # The purpose here is to ensure that after triggering
            # `self.async_rollout_manager.generate_sequences(gen_batch_output)`,
            # the subsequent relevant logic can proceed in a timely manner
            await asyncio.sleep(0)
            batch = self._fit_compute_reward(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_ref_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_critic(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_advantage(batch)
            await asyncio.sleep(0)
            batch = self._fit_update_critic(batch)
            await asyncio.sleep(0)
            batch = self._fit_update_actor(batch)
            await asyncio.sleep(0)
            self._fit_dump_data(batch)
            await asyncio.sleep(0)

        self._fit_validate()
        await asyncio.sleep(0)
        self._fit_save_checkpoint()
        await asyncio.sleep(0)
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_experimental(batch)
        self._fit_postprocess_step()

        return batch_data_future

    async def _fit_generate(self, batch_data_future, continuous_iterator):
        metrics = self.metrics
        timing_raw = self.timing_raw

        with marked_timer("gen", timing_raw, color="red"):
            _metrics, _timing_raw, epoch, batch, future_reward = await batch_data_future
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            timing_raw.update(batch.meta_info["timing"])
            timing_raw.update(_timing_raw)
            metrics.update(_metrics)
            batch.meta_info.pop("timing", None)

        # sync weights from actor to rollout
        with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
            self._fit_update_weights()

        # async next generation
        if not self.is_last_step:
            batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
            await asyncio.sleep(0)
        else:
            batch_data_future = None

        return batch, batch_data_future
