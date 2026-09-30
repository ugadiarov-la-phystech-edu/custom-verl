# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""verl/experimental/fully_async_policy/shell/vcpo/replay_buffer/*.sh: compose both replay / min-ESS arms.

Each script runs for real with ``--cfg job --resolve`` appended, so fully_async_main prints the composed
config instead of training: every override must exist in the config schema, and the bash logic (seeds,
tags, max_updates) must produce the intended values. The composed configs are then fed to the real
FullyAsyncTrainer / FullyAsyncRollouter __init__ and to the worker-side actor dataclass, whose
validation the scripts must pass.
"""

import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "verl" / "experimental" / "fully_async_policy" / "shell" / "vcpo" / "replay_buffer"
_ARM = "grpo_novcpo_8gpu_dapo17k_5+3_resp8k_megatron_offload_replay_tau=8_k=32"
QWEN = f"{_ARM}_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5.sh"
PANGU = f"{_ARM}_min-ess=1.07_ess-lr-scale=0.5_fresh=0.5_openpangu7b.sh"
ALL = [QWEN, PANGU]

# Script knobs that must not leak in from the caller's shell.
SCRIPT_ENV_KNOBS = {
    "VERL_GPU_MEM_CAP_GB",
    "SEED",
    "max_updates",
    "min_ess",
    "concurrency_ramp",
    "replay_min_fresh_ratio",
    "replay_reuse_halflife",
    "replay_requires_mini_batches",
    "log_dir",
    "CKPTS_DIR",
    "exp_name",
}


def _run(script, env, args, tmp):
    bindir = Path(tmp) / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    python = bindir / "python"
    if not python.exists():
        # a wrapper, not a symlink (a symlinked venv interpreter loses the venv); atomic for xdist
        wrapper = bindir / f".python.{os.getpid()}"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        wrapper.chmod(0o755)
        os.replace(wrapper, python)
    full_env = {k: v for k, v in os.environ.items() if k not in SCRIPT_ENV_KNOBS}
    full_env.update(
        {
            "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
            "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "HYDRA_FULL_ERROR": "1",
            "log_dir": str(Path(tmp) / "logs"),
        }
    )
    full_env.update(env)
    # the fully-async config's Hydra searchpath is relative: launch from the repo root, as the scripts say
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *args, "--cfg", "job", "--resolve"],
        cwd=REPO,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@cache
def _compose(script, env_items=(), args=()):
    tmp = Path(os.environ.get("PYTEST_REPLAY_SCRIPTS_TMP", "/tmp")) / f"replay_compose_{os.getpid()}"
    proc = _run(script, dict(env_items), args, tmp)
    assert proc.returncode == 0, f"{script} failed to compose:\n{proc.stderr[-4000:]}"
    return yaml.safe_load(proc.stdout)


def compose(script, args=(), **env):
    return _compose(script, tuple(sorted(env.items())), tuple(args))


def _cfg(script, **kw):
    return OmegaConf.create(compose(script, **kw))


# --------------------------------------------------------------------------- the shared recipe


@pytest.mark.parametrize("script", ALL)
class TestRecipe:
    def test_layout(self, script):
        c = compose(script)
        assert c["actor_rollout_ref"]["hybrid_engine"] is False
        assert c["trainer"]["n_gpus_per_node"] == 3 and c["rollout"]["n_gpus_per_node"] == 5
        m = c["actor_rollout_ref"]["actor"]["megatron"]
        assert (m["tensor_model_parallel_size"], m["pipeline_model_parallel_size"], m["context_parallel_size"]) == (
            1,
            1,
            1,
        )
        opt = c["actor_rollout_ref"]["actor"]["optim"]["override_optimizer_config"]
        assert opt["optimizer_cpu_offload"] is True and opt["main_params_dtype"] == "bfloat16"

    def test_batches(self, script):
        c = compose(script)
        actor = c["actor_rollout_ref"]["actor"]
        assert actor["ppo_mini_batch_size"] == 33 and c["actor_rollout_ref"]["rollout"]["n"] == 16
        assert actor["ppo_epochs"] == 1
        assert actor["loss_agg_mode"] == "seq-mean-token-mean"
        assert c["data"]["train_batch_size"] == 0 and c["data"]["gen_batch_size"] == 1
        assert c["async_training"]["concurrent_samples_per_replica"] == 33

    def test_objective(self, script):
        c = compose(script)
        corr = c["algorithm"]["rollout_correction"]
        assert corr["bypass_mode"] is True and corr["loss_type"] == "reinforce"
        assert corr["rollout_is"] == "token" and float(corr["rollout_is_threshold"]) == 2.0
        assert corr["rollout_rs"] is None
        assert corr["log_probs_pearson_corr"] is True
        policy_loss = c["actor_rollout_ref"]["actor"]["policy_loss"]
        assert policy_loss["loss_mode"] == "bypass_mode"
        assert policy_loss["rollout_correction"] == corr  # the interpolation reaches the worker config
        assert c["algorithm"]["adv_estimator"] == "grpo" and c["algorithm"]["use_kl_in_reward"] is False
        assert c["actor_rollout_ref"]["rollout"]["calculate_log_probs"] is True

    def test_async_and_replay(self, script):
        a = compose(script)["async_training"]
        assert a["trigger_parameter_sync_step"] == 1 and a["require_batches"] == 1
        assert a["partial_rollout"] is True
        assert a["staleness_threshold"] == 32.0
        assert a["serialize_validation"] is True and a["pause_generation_during_save"] is True
        r = a["replay_buffer"]
        assert r["enable"] is True
        assert (r["tau"], r["staleness_threshold"], r["requires_mini_batches"]) == (8, 32, 0.5)
        assert r["reuse_halflife"] == 1 and r["min_fresh_ratio"] == 0.5
        assert r["min_fresh_wait_timeout_s"] == 3600.0

    def test_checkpoints(self, script):
        c = compose(script)
        assert c["actor_rollout_ref"]["actor"]["checkpoint"]["save_contents"] == ["hf_model"]
        assert c["trainer"]["resume_mode"] == "disable"
        assert c["trainer"]["max_actor_ckpt_to_keep"] is None

    @pytest.mark.parametrize("seed", [1, 7])
    def test_one_seed_feeds_every_seed_knob(self, script, seed):
        c = compose(script, SEED=str(seed)) if seed != 1 else compose(script)
        assert c["data"]["seed"] == seed
        assert c["actor_rollout_ref"]["actor"]["megatron"]["seed"] == seed
        assert c["actor_rollout_ref"]["actor"]["data_loader_seed"] == seed
        assert c["actor_rollout_ref"]["rollout"]["seed"] == seed
        assert c["async_training"]["replay_buffer"]["sampling_seed"] == seed
        assert c["trainer"]["experiment_name"].endswith(f"seed-{seed}")

    def test_max_updates(self, script):
        assert compose(script)["trainer"]["total_training_steps"] is None
        assert compose(script, max_updates="12")["trainer"]["total_training_steps"] == 12

    def test_env_overrides_and_tags(self, script):
        c = compose(script, replay_min_fresh_ratio="0", replay_reuse_halflife="null", concurrency_ramp="null")
        name = c["trainer"]["experiment_name"]
        assert c["async_training"]["replay_buffer"]["min_fresh_ratio"] == 0
        assert c["async_training"]["replay_buffer"]["reuse_halflife"] is None
        assert c["async_training"]["concurrency_ramp"] is None
        assert "fresh-" not in name and " nu-" not in name and "ramp-" not in name
        default = compose(script)["trainer"]["experiment_name"]
        assert " nu-1" in default and " fresh-0.5" in default and " rmb-0.5" in default and " ramp-" in default

    def test_emulation_tag(self, script):
        name = compose(script, VERL_GPU_MEM_CAP_GB="80", gpu_memory_utilization="0.5")["trainer"]["experiment_name"]
        assert "h100-emu-80gb-gmu0.5" in name

    def test_logs_go_to_log_dir(self, script):
        c = compose(script)
        assert c["trainer"]["default_local_dir"].endswith("/logs")
        assert c["trainer"]["rollout_data_dir"] == c["trainer"]["default_local_dir"]

    def test_worker_side_actor_config(self, script):
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import ESSScalingConfig

        actor = omega_conf_to_dataclass(_cfg(script).actor_rollout_ref.actor)
        min_ess = 1.1 if script == QWEN else 1.07
        assert actor.ess_scaling == ESSScalingConfig(enable=True, min_ess=min_ess, lr_scale=0.5, use_clipped=False)
        assert actor.policy_loss.loss_mode == "bypass_mode"
        assert actor.policy_loss.rollout_correction["loss_type"] == "reinforce"

    def test_trainer_accepts_the_config(self, script):
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

        cfg = _cfg(script)
        cfg.trainer.logger = ["console"]
        trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
        t = trainer_cls(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
        assert t.replay_enable
        assert t.replay_first_mini_size == 18  # 0.5 x 33 -> 18 (18 x 16 splits over dp=3)
        assert t.replay_buffer.reuse_halflife == 1.0
        assert t.max_train_steps is None

    def test_rollouter_accepts_the_config(self, script, monkeypatch):
        from verl.experimental.fully_async_policy import fully_async_rollouter as mod

        rollouter_cls = mod.FullyAsyncRollouter.__ray_metadata__.modified_class
        monkeypatch.setattr(mod, "create_rl_dataset", lambda *a, **k: [0] * 8)
        monkeypatch.setattr(mod, "create_rl_sampler", lambda *a, **k: None)
        monkeypatch.setattr(
            rollouter_cls, "_create_dataloader", lambda self, *a, **k: setattr(self, "train_dataloader", [0] * 8)
        )
        monkeypatch.setattr(rollouter_cls, "_init_dump_executor", lambda self: None)
        r = rollouter_cls(config=_cfg(script), tokenizer=None)
        assert r.replay_mode and r.serialize_validation and r.pause_generation_during_save
        assert r.concurrency_ramp == ([5, 12, 20] if script == QWEN else [4, 10, 20])
        assert r.ramp_first_size == 18


# --------------------------------------------------------------------------- per-model differences


def test_qwen_model():
    c = compose(QWEN)
    assert c["actor_rollout_ref"]["model"]["path"] == "Qwen/Qwen3-8B"
    assert c["actor_rollout_ref"]["model"].get("external_lib") is None
    assert c["data"].get("add_bos_token_to_prompt", False) is False
    assert c["trainer"]["test_freq"] == 25 and c["trainer"]["save_freq"] == 25
    assert "Qwen3-8B" in c["trainer"]["experiment_name"]


def test_openpangu_model():
    c = compose(PANGU)
    model = c["actor_rollout_ref"]["model"]
    assert model["path"].endswith("openPangu-Embedded-7B-llama")
    assert model["trust_remote_code"] is True and c["data"]["trust_remote_code"] is True
    assert model["external_lib"] == "verl.models.mcore.llama_attention_bias_bridge"
    assert c["data"]["add_bos_token_to_prompt"] is True
    assert c["trainer"]["test_freq"] == 15 and c["trainer"]["save_freq"] == 15
    name = c["trainer"]["experiment_name"]
    assert "openPangu-7B" in name and " bos " in name


def test_the_arms_differ_only_where_intended():
    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{prefix}{k}."))
            else:
                out[f"{prefix}{k}"] = v
        return out

    q, p = flat(compose(QWEN)), flat(compose(PANGU))
    differing = {k for k in q.keys() | p.keys() if q.get(k) != p.get(k)}
    # keys interpolated from the experiment name or the model path follow those two
    derived = {k for k in differing if "experiment_name" in k or k.endswith("default_local_dir")}
    derived |= {"actor_rollout_ref.model.tokenizer_path", "actor_rollout_ref.rollout.prometheus.served_model_name"}
    assert differing - derived == {
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.model.trust_remote_code",
        "actor_rollout_ref.model.external_lib",
        "data.trust_remote_code",
        "data.add_bos_token_to_prompt",
        "actor_rollout_ref.actor.ess_scaling.min_ess",
        "async_training.concurrency_ramp",
        "trainer.test_freq",
        "trainer.save_freq",
    }


def test_unknown_override_fails():
    proc = _run(QWEN, {}, ("async_training.not_a_key=1",), Path("/tmp") / f"replay_compose_{os.getpid()}")
    assert proc.returncode != 0
