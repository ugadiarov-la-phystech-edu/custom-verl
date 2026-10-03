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
QWEN_FP8 = f"{_ARM}_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8.sh"
QWEN_FP8_C8 = f"{_ARM}_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_tis-c8_token-mean.sh"
ALL = [QWEN, PANGU, QWEN_FP8]
QWEN_ARMS = (QWEN, QWEN_FP8)
DEEPGEMM_VARS = ("CUDA_HOME", "DG_JIT_CACHE_DIR", "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER")

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
    "ROLLOUT_QUANT",
    "gpu_memory_utilization",
    "lr_warmup_steps",
    "lr_decay_steps",
    "total_rollout_steps",
    "test_freq",
    "save_freq",
    "rollout_is_threshold",
    "loss_agg_mode",
    "DYNAMIC_BSZ",
    "DYNAMIC_BSZ_MAX_TOKENS",
    "DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS",
    "DEEPGEMM_CUDA_HOME",
    *DEEPGEMM_VARS,
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
    for _ in range(2):
        proc = subprocess.run(
            ["bash", str(SCRIPTS / script), *args, "--cfg", "job", "--resolve"],
            cwd=REPO,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        # Retry once only if the Python interpreter itself crashed (bash reports 128 + SIGSEGV), as in
        # test_baseline_scripts_on_cpu.py: seen sporadically under 8 pytest-xdist workers. Real errors are not
        # retried.
        if proc.returncode != 128 + 11:
            break
    return proc


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
        assert c["trainer"]["experiment_name"].endswith(f" seed-{seed}")

    def test_max_updates(self, script):
        assert compose(script)["trainer"]["total_training_steps"] is None
        assert compose(script, max_updates="13")["trainer"]["total_training_steps"] == 13

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
        min_ess = 1.1 if script in QWEN_ARMS else 1.07
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
        assert r.concurrency_ramp == ([5, 12, 20] if script in QWEN_ARMS else [4, 10, 20])
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


# --------------------------------------------------------------------------- fp8 rollout (QWEN_FP8 wrapper)


class TestFp8Rollout:
    def test_default_rollout_is_bf16(self):
        c = compose(QWEN)
        assert c["actor_rollout_ref"]["rollout"]["quantization"] is None
        assert "rollout-fp8" not in c["trainer"]["experiment_name"]

    def test_wrapper_sets_fp8(self):
        c = compose(QWEN_FP8)
        assert c["actor_rollout_ref"]["rollout"]["quantization"] == "fp8"
        assert c["trainer"]["experiment_name"].endswith(" 0.1-wd warmup-12 rollout-fp8 seed-1")
        # the objective that absorbs the FP8 mismatch is unchanged: token TIS against rollout log-probs
        corr = c["algorithm"]["rollout_correction"]
        assert (corr["rollout_is"], float(corr["rollout_is_threshold"]), corr["loss_type"]) == (
            "token",
            2.0,
            "reinforce",
        )
        assert c["actor_rollout_ref"]["rollout"]["calculate_log_probs"] is True

    def test_wrapper_differs_from_the_arm_only_by_rollout_precision(self):
        def flat(d, prefix=""):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out.update(flat(v, f"{prefix}{k}."))
                else:
                    out[f"{prefix}{k}"] = v
            return out

        q, f = flat(compose(QWEN)), flat(compose(QWEN_FP8))
        differing = {k for k in q.keys() | f.keys() if q.get(k) != f.get(k)}
        name_derived = {k for k in differing if "experiment_name" in k or k.endswith("default_local_dir")}
        # rollout precision, the H100 emulation's vLLM budget (the trainer cap is an env var, not config) and
        # the wrappers' schedule
        assert differing - name_derived == {
            "actor_rollout_ref.rollout.quantization",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.actor.optim.lr_warmup_steps",
            "trainer.test_freq",
            "trainer.save_freq",
        }

    def test_toggle_on_the_arm_equals_the_wrapper(self):
        emu = dict(VERL_GPU_MEM_CAP_GB="78", gpu_memory_utilization="0.5")
        schedule = dict(lr_warmup_steps="12", test_freq="12", save_freq="12", SEED="1")
        assert compose(QWEN, ROLLOUT_QUANT="fp8", **emu, **schedule) == compose(QWEN_FP8)

    def test_wrapper_can_be_switched_back_to_bf16(self):
        real_h100 = dict(VERL_GPU_MEM_CAP_GB="", gpu_memory_utilization="0.9")
        arm_schedule = dict(lr_warmup_steps="0", test_freq="25", save_freq="25")
        assert compose(QWEN_FP8, ROLLOUT_QUANT="bf16", **real_h100, **arm_schedule) == compose(QWEN)

    @pytest.mark.parametrize("script", QWEN_ARMS)
    def test_invalid_rollout_quant_is_rejected(self, script, tmp_path):
        proc = _run(script, {"ROLLOUT_QUANT": "int8"}, (), tmp_path)
        assert proc.returncode == 2
        assert "ROLLOUT_QUANT must be bf16 or fp8" in proc.stderr

    def test_wrapper_forwards_hydra_overrides(self):
        c = compose(QWEN_FP8, args=("trainer.total_training_steps=3",), SEED="7")
        assert c["trainer"]["total_training_steps"] == 3
        assert c["actor_rollout_ref"]["rollout"]["seed"] == 7
        assert c["trainer"]["experiment_name"].endswith(" warmup-12 rollout-fp8 seed-7")


def _env_seen_by_python(tmp_path, env, script=QWEN_FP8, names=DEEPGEMM_VARS):
    """Run a wrapper with a stand-in python that prints the given env vars (default: the DeepGEMM ones)."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "python"
    fake.write_text("#!/bin/sh\n" + "".join(f'echo "{v}=${{{v}-<unset>}}"\n' for v in names))
    fake.chmod(0o755)
    base = {k: v for k, v in os.environ.items() if k not in SCRIPT_ENV_KNOBS}
    base.update({"PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}", "log_dir": str(tmp_path / "logs")})
    base.update(env)
    proc = subprocess.run(
        ["bash", str(SCRIPTS / script)], cwd=REPO, env=base, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def _fake_toolkit(tmp_path):
    toolkit = tmp_path / "cuda-12.9"
    (toolkit / "bin").mkdir(parents=True)
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)
    return toolkit


class TestFp8DeepGemmEnv:
    def test_toolkit_found_exports_all_three(self, tmp_path):
        toolkit = _fake_toolkit(tmp_path)
        seen = _env_seen_by_python(tmp_path, {"DEEPGEMM_CUDA_HOME": str(toolkit)})
        assert seen["CUDA_HOME"] == str(toolkit)
        assert seen["DG_JIT_CACHE_DIR"] == "/home/jovyan/ugadiarov/cache/deep_gemm"
        assert seen["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] == "0"

    def test_missing_toolkit_exports_nothing(self, tmp_path):
        seen = _env_seen_by_python(tmp_path, {"DEEPGEMM_CUDA_HOME": str(tmp_path / "missing")})
        assert set(seen.values()) == {"<unset>"}

    def test_existing_cuda_home_is_left_alone(self, tmp_path):
        # e.g. remote_h200, whose activate.sh sets all three itself
        toolkit = _fake_toolkit(tmp_path)
        env = {"DEEPGEMM_CUDA_HOME": str(toolkit), "CUDA_HOME": "/opt/cuda", "DG_JIT_CACHE_DIR": "/data/dg"}
        seen = _env_seen_by_python(tmp_path, env)
        assert seen["CUDA_HOME"] == "/opt/cuda"
        assert seen["DG_JIT_CACHE_DIR"] == "/data/dg"
        assert seen["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] == "<unset>"

    def test_explicit_values_win_over_defaults(self, tmp_path):
        toolkit = _fake_toolkit(tmp_path)
        env = {
            "DEEPGEMM_CUDA_HOME": str(toolkit),
            "DG_JIT_CACHE_DIR": "/data/dg",
            "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "1",
        }
        seen = _env_seen_by_python(tmp_path, env)
        assert seen["CUDA_HOME"] == str(toolkit)
        assert seen["DG_JIT_CACHE_DIR"] == "/data/dg"
        assert seen["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] == "1"


# --------------------------------------------------------------------------- fp8 + sync-baseline TIS cap / aggregation


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _differing(a, b):
    fa, fb = _flat(a), _flat(b)
    diff = {k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)}
    return {k for k in diff if "experiment_name" not in k and not k.endswith("default_local_dir")}


# Keys DYNAMIC_BSZ=True changes. The ref.* and critic.* ones are interpolated from the actor/rollout keys and
# inert here (no reference policy, no critic).
DYNBSZ_KEYS = {
    "actor_rollout_ref.actor.use_dynamic_bsz",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz",
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
    "critic.use_dynamic_bsz",
}


class TestFp8TisC8TokenMean:
    def test_cap_and_aggregation(self):
        c = compose(QWEN_FP8_C8)
        assert float(c["algorithm"]["rollout_correction"]["rollout_is_threshold"]) == 8.0
        # the worker's loss reads only policy_loss.rollout_correction: the cap must reach it too
        assert (
            float(c["actor_rollout_ref"]["actor"]["policy_loss"]["rollout_correction"]["rollout_is_threshold"]) == 8.0
        )
        assert c["actor_rollout_ref"]["actor"]["loss_agg_mode"] == "token-mean"
        assert c["actor_rollout_ref"]["rollout"]["quantization"] == "fp8"
        corr = c["algorithm"]["rollout_correction"]
        assert (corr["rollout_is"], corr["loss_type"], corr["bypass_mode"]) == ("token", "reinforce", True)

    def test_name(self):
        name = compose(QWEN_FP8_C8)["trainer"]["experiment_name"]
        assert " token-mean " in name
        assert name.endswith(" 0.1-wd warmup-12 dynbsz-10240 tis-C8 rollout-fp8 seed-1")

    def test_differs_from_the_fp8_arm_only_by_cap_aggregation_and_dynbsz(self):
        assert _differing(compose(QWEN_FP8), compose(QWEN_FP8_C8)) == {
            "algorithm.rollout_correction.rollout_is_threshold",
            "actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold",
            "actor_rollout_ref.actor.loss_agg_mode",
            "critic.loss_agg_mode",  # interpolated from the actor's; inert (GRPO has no critic)
            *DYNBSZ_KEYS,
        }

    def test_equals_the_fp8_arm_with_the_three_knobs(self):
        assert compose(QWEN_FP8_C8) == compose(
            QWEN_FP8, rollout_is_threshold="8", loss_agg_mode="token-mean", DYNAMIC_BSZ="True"
        )

    def test_knobs_can_be_set_back(self):
        assert compose(
            QWEN_FP8_C8, rollout_is_threshold="2.0", loss_agg_mode="seq-mean-token-mean", DYNAMIC_BSZ="False"
        ) == compose(QWEN_FP8)

    def test_base_default_cap_is_untagged(self):
        c = compose(QWEN)
        assert float(c["algorithm"]["rollout_correction"]["rollout_is_threshold"]) == 2.0
        assert "tis-C" not in c["trainer"]["experiment_name"]
        assert compose(QWEN, rollout_is_threshold="2") == c  # 2 == 2.0: no tag, same config

    @pytest.mark.parametrize("bad", ["0", "-1", "abc", "8x"])
    def test_invalid_cap_is_rejected(self, bad, tmp_path):
        proc = _run(QWEN, {"rollout_is_threshold": bad}, (), tmp_path)
        assert proc.returncode == 2
        assert "rollout_is_threshold must be a positive number" in proc.stderr

    def test_trainer_and_worker_accept_the_config(self):
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
        from verl.utils.config import omega_conf_to_dataclass

        cfg = _cfg(QWEN_FP8_C8)
        actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
        assert actor.loss_agg_mode == "token-mean"
        assert float(actor.policy_loss.rollout_correction["rollout_is_threshold"]) == 8.0
        cfg.trainer.logger = ["console"]
        trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
        t = trainer_cls(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
        assert t.replay_enable

    def test_deepgemm_env_is_inherited_from_the_fp8_wrapper(self, tmp_path):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        fake = bindir / "python"
        fake.write_text("#!/bin/sh\n" + "".join(f'echo "{v}=${{{v}-<unset>}}"\n' for v in DEEPGEMM_VARS))
        fake.chmod(0o755)
        toolkit = _fake_toolkit(tmp_path)
        env = {k: v for k, v in os.environ.items() if k not in SCRIPT_ENV_KNOBS}
        env.update({"PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}", "log_dir": str(tmp_path / "logs")})
        env["DEEPGEMM_CUDA_HOME"] = str(toolkit)
        proc = subprocess.run(
            ["bash", str(SCRIPTS / QWEN_FP8_C8)], cwd=REPO, env=env, capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        seen = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        assert seen["CUDA_HOME"] == str(toolkit)
        assert seen["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] == "0"


# --------------------------------------------------------------------------- H100 emulation defaults (fp8 wrappers)

FP8_WRAPPERS = (QWEN_FP8, QWEN_FP8_C8)


@pytest.mark.parametrize("script", FP8_WRAPPERS)
class TestFp8H100Emulation:
    def test_vllm_budget_and_tag(self, script):
        c = compose(script)
        assert c["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.5
        assert " h100-emu-78gb-gmu0.5 " in c["trainer"]["experiment_name"]

    def test_cap_reaches_the_trainer_env(self, script, tmp_path):
        seen = _env_seen_by_python(tmp_path, {}, script=script, names=("VERL_GPU_MEM_CAP_GB",))
        assert seen["VERL_GPU_MEM_CAP_GB"] == "78"

    def test_real_h100_override(self, script, tmp_path):
        c = compose(script, VERL_GPU_MEM_CAP_GB="", gpu_memory_utilization="0.88")
        assert c["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.88
        assert "h100-emu" not in c["trainer"]["experiment_name"]
        env = {"VERL_GPU_MEM_CAP_GB": ""}
        assert _env_seen_by_python(tmp_path, env, script=script, names=("VERL_GPU_MEM_CAP_GB",)) == {
            "VERL_GPU_MEM_CAP_GB": ""
        }

    def test_explicit_values_win(self, script):
        c = compose(script, VERL_GPU_MEM_CAP_GB="70", gpu_memory_utilization="0.45")
        assert c["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.45
        assert " h100-emu-70gb-gmu0.45 " in c["trainer"]["experiment_name"]


def test_bf16_arm_does_not_emulate():
    c = compose(QWEN)
    assert c["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.9
    assert "h100-emu" not in c["trainer"]["experiment_name"]


# --------------------------------------------------------------------------- schedule (fp8 wrappers) and warmup knob


@pytest.mark.parametrize("script", FP8_WRAPPERS)
class TestFp8Schedule:
    def test_defaults(self, script):
        c = compose(script)
        assert c["actor_rollout_ref"]["actor"]["optim"]["lr_warmup_steps"] == 12
        assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (12, 12)
        assert c["data"]["seed"] == 1 and c["actor_rollout_ref"]["rollout"]["seed"] == 1
        assert " warmup-12 " in c["trainer"]["experiment_name"]

    def test_overridable(self, script):
        c = compose(script, lr_warmup_steps="0", test_freq="5", save_freq="10", SEED="3")
        assert c["actor_rollout_ref"]["actor"]["optim"]["lr_warmup_steps"] == 0
        assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (5, 10)
        assert c["data"]["seed"] == 3
        assert "warmup-" not in c["trainer"]["experiment_name"]

    def test_warmup_must_be_shorter_than_the_lr_decay_horizon(self, script, tmp_path):
        proc = _run(script, {"total_rollout_steps": "12"}, (), tmp_path)  # lr_decay_steps defaults to it
        assert proc.returncode == 2
        assert "lr_warmup_steps=12 must be < lr_decay_steps=12" in proc.stderr
        # the update cap is not the scheduler horizon: a short capped run with the 12-update warmup is fine
        assert compose(script, max_updates="3")["trainer"]["total_training_steps"] == 3


class TestWarmupKnob:
    def test_arm_default_is_untagged_zero(self):
        c = compose(QWEN)
        assert c["actor_rollout_ref"]["actor"]["optim"]["lr_warmup_steps"] == 0
        assert "warmup-" not in c["trainer"]["experiment_name"]

    @pytest.mark.parametrize("bad", ["-1", "1.5", "abc"])
    def test_invalid_is_rejected(self, bad, tmp_path):
        proc = _run(QWEN, {"lr_warmup_steps": bad}, (), tmp_path)
        assert proc.returncode == 2
        assert "lr_warmup_steps must be a non-negative integer" in proc.stderr


# --------------------------------------------------------------------------- LR schedule horizon (all arms)


@pytest.mark.parametrize("script", ALL)
class TestLrDecayHorizon:
    """The fully-async trainer builds Megatron's LR scheduler before it learns the run length
    (optim.total_training_steps is still -1), so lr_decay_steps must be passed explicitly; Megatron asserts
    lr_decay_steps > 0 and lr_warmup_steps < lr_decay_steps. Without it every launch dies at trainer init."""

    def test_passed_and_valid(self, script):
        optim = compose(script)["actor_rollout_ref"]["actor"]["optim"]
        assert optim["lr_decay_steps"] == 66000  # = total_rollout_steps, as in the custom_vcpo source
        assert 0 <= optim["lr_warmup_steps"] < optim["lr_decay_steps"]

    def test_follows_total_rollout_steps_and_is_overridable(self, script):
        assert compose(script, total_rollout_steps="64")["actor_rollout_ref"]["actor"]["optim"]["lr_decay_steps"] == 64
        c = compose(script, lr_decay_steps="500")
        assert c["actor_rollout_ref"]["actor"]["optim"]["lr_decay_steps"] == 500

    @pytest.mark.parametrize("bad", ["0", "-5", "1.5"])
    def test_invalid_is_rejected(self, script, bad, tmp_path):
        proc = _run(script, {"lr_decay_steps": bad}, (), tmp_path)
        assert proc.returncode == 2
        assert "lr_decay_steps must be a positive integer" in proc.stderr


# --------------------------------------------------------------------------- dynamic batch size toggle


class TestDynamicBatchSize:
    def test_default_is_off_and_untagged(self):
        for script in (QWEN, QWEN_FP8, PANGU):
            c = compose(script)
            assert c["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
            assert c["actor_rollout_ref"]["rollout"]["log_prob_use_dynamic_bsz"] is False
            assert "dynbsz" not in c["trainer"]["experiment_name"]
        assert compose(QWEN, DYNAMIC_BSZ="False") == compose(QWEN)

    def test_on_sets_flags_and_caps_consistently(self):
        c = compose(QWEN, DYNAMIC_BSZ="True")
        actor, rollout = c["actor_rollout_ref"]["actor"], c["actor_rollout_ref"]["rollout"]
        # engine_workers.py asserts these two flags are equal and both caps set
        assert actor["use_dynamic_bsz"] is rollout["log_prob_use_dynamic_bsz"] is True
        # default cap: one full-length sequence, the micro-batch-1 memory worst case
        assert actor["ppo_max_token_len_per_gpu"] == c["data"]["max_prompt_length"] + c["data"]["max_response_length"]
        assert actor["ppo_max_token_len_per_gpu"] == rollout["log_prob_max_token_len_per_gpu"] == 10240
        assert _differing(compose(QWEN), c) == DYNBSZ_KEYS
        assert c["trainer"]["experiment_name"].endswith(" 0.1-wd dynbsz-10240 seed-1")

    def test_caps_are_configurable(self):
        c = compose(QWEN, DYNAMIC_BSZ="True", DYNAMIC_BSZ_MAX_TOKENS="16384", DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS="20480")
        assert c["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == 16384
        assert c["actor_rollout_ref"]["rollout"]["log_prob_max_token_len_per_gpu"] == 20480
        assert " dynbsz-16384 " in c["trainer"]["experiment_name"]

    def test_log_prob_cap_follows_training_cap(self):
        c = compose(QWEN, DYNAMIC_BSZ="True", DYNAMIC_BSZ_MAX_TOKENS="12288")
        assert c["actor_rollout_ref"]["rollout"]["log_prob_max_token_len_per_gpu"] == 12288

    def test_tis_c8_wrapper_enables_it_and_can_switch_it_off(self):
        assert compose(QWEN_FP8_C8)["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is True
        off = compose(QWEN_FP8_C8, DYNAMIC_BSZ="False")
        assert off["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
        assert "dynbsz" not in off["trainer"]["experiment_name"]

    def test_trainer_and_worker_accept_the_config(self):
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
        from verl.utils.config import omega_conf_to_dataclass

        cfg = _cfg(QWEN_FP8_C8)
        actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
        assert actor.use_dynamic_bsz is True and actor.ppo_max_token_len_per_gpu == 10240
        cfg.trainer.logger = ["console"]
        trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
        t = trainer_cls(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
        assert t.replay_enable and t.replay_first_mini_size == 18


@pytest.mark.parametrize("script", [QWEN, QWEN_FP8_C8])
@pytest.mark.parametrize(
    "env, message",
    [
        ({"DYNAMIC_BSZ": "maybe"}, "DYNAMIC_BSZ must be"),
        ({"DYNAMIC_BSZ": "True", "DYNAMIC_BSZ_MAX_TOKENS": "8192"}, "must be >= max_prompt_length"),
        ({"DYNAMIC_BSZ": "True", "DYNAMIC_BSZ_MAX_TOKENS": "abc"}, "must be positive integers"),
        ({"DYNAMIC_BSZ": "True", "DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS": "4096"}, "must be >= max_prompt_length"),
    ],
)
def test_invalid_dynbsz_settings_are_rejected(script, env, message, tmp_path):
    proc = _run(script, env, (), tmp_path)
    assert proc.returncode == 2
    assert message in proc.stderr
