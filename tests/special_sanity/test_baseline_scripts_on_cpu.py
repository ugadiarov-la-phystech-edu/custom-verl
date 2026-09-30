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
"""examples/baselines/run_*_megatron.sh: compose each sync GRPO baseline through Hydra and pin the result.

Every script is run for real with ``--cfg job --resolve`` appended, so ``main_ppo`` prints the fully
composed config instead of training. That checks that every override exists in (or is legally
appended to) the verl config schema, and that the scripts' bash logic (seeds, max_updates, the H100
emulation tag) produces the intended values.
"""

import asyncio
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "examples" / "baselines"
QWEN = "run_qwen3-8b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh"
PANGU = "run_openpangu7b_dapo17k_grpo_sync_B128xn16_mini32_megatron.sh"
ORZ = "run_orz7b_orz72k_grpo_sync_B128xn16_mini32_megatron.sh"
ALL = [QWEN, PANGU, ORZ]


def _run(script, env=None, args=(), tmp=None):
    """Run a script with --cfg job --resolve; returns (returncode, stdout, stderr)."""
    workdir = Path(tmp or os.environ.get("PYTEST_BASELINE_TMP", "/tmp")) / "baseline_compose"
    bindir = workdir / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    # A wrapper, not a symlink: a symlinked venv interpreter outside the venv loses the venv.
    python3 = bindir / "python3"
    if not python3.exists():
        python3.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python3.chmod(0o755)
    full_env = {k: v for k, v in os.environ.items() if k not in {"VERL_GPU_MEM_CAP_GB", "SEED", "max_updates"}}
    full_env.update(
        {
            "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
            "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    full_env.update(env or {})
    proc = subprocess.run(
        ["bash", str(SCRIPTS / script), *args, "--cfg", "job", "--resolve"],
        cwd=workdir,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc.returncode, proc.stdout, proc.stderr


@cache
def _compose(script, env_items=(), args=()):
    rc, out, err = _run(script, dict(env_items), args)
    assert rc == 0, f"{script} failed to compose:\n{err[-4000:]}"
    return yaml.safe_load(out)


def compose(script, args=(), **env):
    return _compose(script, tuple(sorted(env.items())), tuple(args))


# trainer.experiment_name and every key interpolated from it
NAME_DERIVED_KEYS = {
    "trainer.experiment_name",
    "trainer.default_local_dir",
    "actor_rollout_ref.rollout.trace.experiment_name",
}


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


# --------------------------------------------------------------------------- shared recipe


@pytest.mark.parametrize("script", ALL)
class TestSharedRecipe:
    def test_trainer_path(self, script):
        c = compose(script)
        assert c["trainer"]["use_v1"] is True
        assert c["trainer"]["v1"]["trainer_mode"] == "sync"
        assert c["actor_rollout_ref"]["actor"]["strategy"] == "megatron"
        assert c["actor_rollout_ref"]["hybrid_engine"] is True
        assert c["critic"]["enable"] in (None, False)

    def test_geometry(self, script):
        c = compose(script)
        actor = c["actor_rollout_ref"]["actor"]
        assert c["data"]["train_batch_size"] == 128
        assert actor["ppo_mini_batch_size"] == 32
        assert actor["ppo_epochs"] == 1
        assert actor["ppo_micro_batch_size_per_gpu"] == 1
        assert actor["use_dynamic_bsz"] is False
        assert c["actor_rollout_ref"]["rollout"]["n"] == 16
        assert (c["data"]["max_prompt_length"], c["data"]["max_response_length"]) == (2048, 8192)

    def test_stock_ppo_loss_grpo_no_kl(self, script):
        c = compose(script)
        actor = c["actor_rollout_ref"]["actor"]
        assert c["algorithm"]["adv_estimator"] == "grpo"
        assert c["algorithm"]["use_kl_in_reward"] is False
        assert actor["policy_loss"]["loss_mode"] == "vanilla"
        assert (actor["clip_ratio"], actor["clip_ratio_low"], actor["clip_ratio_high"]) == (0.2, 0.2, 0.2)
        assert actor["clip_ratio_c"] == 3.0
        assert actor["loss_agg_mode"] == "token-mean"
        assert actor["use_kl_loss"] is False
        assert actor["entropy_coeff"] == 0
        assert actor["calculate_entropy"] is True
        assert c["algorithm"]["rollout_correction"]["bypass_mode"] is False
        assert c["algorithm"]["rollout_correction"]["rollout_is"] is None

    def test_optimizer(self, script):
        optim = compose(script)["actor_rollout_ref"]["actor"]["optim"]
        assert optim["lr"] == 1e-6
        assert optim["lr_warmup_steps"] == 0
        assert optim["lr_decay_style"] == "constant"
        assert optim["weight_decay"] == 0.01
        assert optim["clip_grad"] == 1.0
        assert optim["override_optimizer_config"] == {
            "optimizer_cpu_offload": True,
            "optimizer_offload_fraction": 1.0,
            "use_torch_optimizer_for_cpu_offload": True,
            "overlap_cpu_optimizer_d2h_h2d": False,
            "use_precision_aware_optimizer": True,
            "main_params_dtype": "bfloat16",
        }

    def test_megatron(self, script):
        c = compose(script)
        mg = c["actor_rollout_ref"]["actor"]["megatron"]
        assert (mg["tensor_model_parallel_size"], mg["pipeline_model_parallel_size"]) == (1, 1)
        assert mg["context_parallel_size"] == 1
        assert mg["param_offload"] is False and mg["optimizer_offload"] is False
        assert mg["dtype"] == "bfloat16"
        assert mg["override_ddp_config"] == {"grad_reduce_in_fp32": False}
        tf = mg["override_transformer_config"]
        assert (tf["recompute_granularity"], tf["recompute_method"], tf["recompute_num_layers"]) == (
            "full",
            "uniform",
            1,
        )
        assert c["actor_rollout_ref"]["model"]["use_remove_padding"] is True

    def test_rollout(self, script):
        r = compose(script)["actor_rollout_ref"]["rollout"]
        assert (r["name"], r["mode"]) == ("vllm", "async")
        assert r["gpu_memory_utilization"] == 0.5
        assert r["tensor_model_parallel_size"] == 1
        assert r["max_num_batched_tokens"] == 10240
        assert r["enable_chunked_prefill"] is True
        assert (r["temperature"], r["top_p"], r["top_k"]) == (1.0, 1.0, -1)
        assert r["calculate_log_probs"] is True
        assert r["val_kwargs"]["n"] == 1
        # v0.9.0 asserts these match the actor's dynamic-bsz setting
        assert r["log_prob_use_dynamic_bsz"] is False
        assert r["log_prob_micro_batch_size_per_gpu"] == 1

    def test_checkpoints(self, script):
        c = compose(script)
        assert c["actor_rollout_ref"]["actor"]["checkpoint"]["save_contents"] == ["hf_model"]
        assert c["trainer"]["resume_mode"] == "disable"
        assert c["trainer"]["max_actor_ckpt_to_keep"] is None
        assert c["trainer"]["default_local_dir"] == f"logs/{c['trainer']['experiment_name']}"

    def test_one_seed_feeds_every_knob_including_vllm(self, script):
        for seed in (1, 7):
            c = compose(script, SEED=str(seed)) if seed != 1 else compose(script)
            arr = c["actor_rollout_ref"]
            seeds = {
                "data.seed": c["data"]["seed"],
                "actor.megatron.seed": arr["actor"]["megatron"]["seed"],
                "actor.data_loader_seed": arr["actor"]["data_loader_seed"],
                "ref.megatron.seed": arr["ref"]["megatron"]["seed"],
                "critic.megatron.seed": c["critic"]["megatron"]["seed"],
                "rollout.seed": arr["rollout"]["seed"],
            }
            assert set(seeds.values()) == {seed}, seeds
            assert c["trainer"]["experiment_name"].endswith(f"seed-{seed}")

    def test_default_has_no_step_cap(self, script):
        assert compose(script)["trainer"]["total_training_steps"] is None

    @pytest.mark.parametrize(
        "max_updates, extra, expected",
        [
            ("200", {}, 50),  # 4 updates per rollout step
            ("201", {}, 51),  # rounded up to whole rollout steps
            ("4", {}, 1),
            ("1", {}, 1),
            ("200", {"train_prompt_mini_bsz": "64"}, 100),  # 2 updates per step
            ("200", {"ppo_epochs": "2"}, 25),  # 8 updates per step
        ],
    )
    def test_max_updates_becomes_total_training_steps(self, script, max_updates, extra, expected):
        c = compose(script, max_updates=max_updates, **extra)
        assert c["trainer"]["total_training_steps"] == expected

    def test_cli_total_training_steps_wins(self, script):
        c = compose(script, args=("trainer.total_training_steps=3",), max_updates="200")
        assert c["trainer"]["total_training_steps"] == 3

    def test_h100_emulation_tag(self, script):
        plain = compose(script)["trainer"]["experiment_name"]
        assert "h100-emu" not in plain
        emu = compose(script, VERL_GPU_MEM_CAP_GB="108", gpu_memory_utilization="0.283")
        name = emu["trainer"]["experiment_name"]
        assert name.endswith(" h100-emu-108gb-gmu0.283")
        assert emu["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.283
        # the knob is only read: nothing else in the config changes except the tagged names
        diff = {
            k for k, v in _flatten(emu).items() if _flatten(compose(script, gpu_memory_utilization="0.283")).get(k) != v
        }
        assert diff == NAME_DERIVED_KEYS


@pytest.mark.parametrize("script", ALL)
@pytest.mark.parametrize(
    "env",
    [
        {"max_updates": "abc"},
        {"max_updates": "0"},
        {"max_updates": "-5"},
        {"max_updates": "1.5"},
        {"train_prompt_mini_bsz": "256"},  # mini batch larger than the batch: 0 updates per step
    ],
)
def test_invalid_step_cap_or_geometry_fails_fast(script, env, tmp_path):
    rc, _, err = _run(script, env=env, tmp=tmp_path)
    assert rc == 2, err[-2000:]
    assert "must be" in err


# --------------------------------------------------------------------------- per-script differences


def test_qwen_arm():
    c = compose(QWEN)
    assert c["actor_rollout_ref"]["model"]["path"] == "Qwen/Qwen3-8B"
    assert c["data"]["add_bos_token_to_prompt"] is False
    assert c["actor_rollout_ref"]["model"]["external_lib"] is None
    assert c["reward"]["custom_reward_function"]["path"] is None
    val = c["actor_rollout_ref"]["rollout"]["val_kwargs"]
    assert (val["temperature"], val["top_p"]) == (0.8, 0.7)
    assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (2, 2)


def test_openpangu_arm():
    c = compose(PANGU)
    model = c["actor_rollout_ref"]["model"]
    assert model["external_lib"] == "verl.models.mcore.llama_attention_bias_bridge"
    assert model["trust_remote_code"] is True
    assert c["data"]["trust_remote_code"] is True
    assert c["data"]["add_bos_token_to_prompt"] is True
    assert c["reward"]["custom_reward_function"]["path"] is None
    val = c["actor_rollout_ref"]["rollout"]["val_kwargs"]
    assert (val["temperature"], val["top_p"]) == (0.8, 0.7)
    assert " bos seed-1" in c["trainer"]["experiment_name"]


def test_openpangu_exports_the_hf_modules_cache_for_ray_workers():
    text = (SCRIPTS / PANGU).read_text()
    assert 'export PYTHONPATH="${HF_MODULES_CACHE}' in text


def test_openpangu_is_the_qwen_recipe_with_only_model_specific_changes():
    qwen, pangu = _flatten(compose(QWEN)), _flatten(compose(PANGU))
    differing = {k for k in qwen.keys() | pangu.keys() if qwen.get(k) != pangu.get(k)}
    must_differ = {
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.model.trust_remote_code",
        "actor_rollout_ref.model.external_lib",
        "data.trust_remote_code",
        "data.add_bos_token_to_prompt",
    } | NAME_DERIVED_KEYS
    assert must_differ <= differing
    # Anything else that differs must be interpolated from the model path (tokenizer_path,
    # critic.model.path, served_model_name, ...), never an independent training/rollout setting.
    for key in differing - must_differ:
        assert "Qwen3-8B" in str(qwen.get(key)), (key, qwen.get(key), pangu.get(key))


def test_orz_arm():
    c = compose(ORZ)
    assert c["actor_rollout_ref"]["model"]["path"] == "Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
    assert c["data"]["add_bos_token_to_prompt"] is False
    assert c["actor_rollout_ref"]["model"]["external_lib"] is None
    crf = c["reward"]["custom_reward_function"]
    assert crf == {**crf, "path": "rewards/orz_tag_aware_math.py", "name": "compute_score"}
    # the legacy top-level key is ignored by main_ppo in 0.9.0 and must stay unset
    assert c["custom_reward_function"]["path"] is None
    val = c["actor_rollout_ref"]["rollout"]["val_kwargs"]
    assert (val["temperature"], val["top_p"]) == (1.0, 1.0)
    assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (5, 5)


def test_orz_reward_loads_through_the_trainer_loader_and_scores(monkeypatch):
    from omegaconf import OmegaConf

    from verl.trainer.ppo.reward import get_custom_reward_fn

    monkeypatch.chdir(REPO)  # scripts launch from the repo root; the path is relative to it
    config = OmegaConf.create({"reward": compose(ORZ)["reward"]})
    fn = get_custom_reward_fn(config)
    assert fn is not None

    async def score(solution):
        out = fn(data_source="math_dapo", solution_str=solution, ground_truth="42", extra_info={})
        return await out if asyncio.iscoroutine(out) else out

    assert asyncio.run(score("<think>...</think> <answer>\\boxed{42}</answer>"))["score"] == 1.0
    assert asyncio.run(score("<answer>\\boxed{41}</answer>"))["score"] == -1.0


@pytest.mark.parametrize("script", ALL)
def test_no_legacy_or_noop_overrides(script):
    text = (SCRIPTS / script).read_text()
    for stale in (
        "+actor_rollout_ref.actor.megatron.override_transformer_config",
        "actor_rollout_ref.actor.megatron.use_remove_padding=",
        "actor_rollout_ref.actor.megatron.grad_offload=",
        "\n    custom_reward_function.",
        "recipe/fully_async_policy/reward",
    ):
        assert stale not in text, stale
