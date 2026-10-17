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
QWEN25_FP8_C8 = (
    "grpo_novcpo_8gpu_dapo17k_3+5_resp8k_megatron_offload_replay_tau=8_k=32"
    "_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.5_fp8_tis-c8_token-mean_qwen2.5-7b.sh"
)
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
    "replay_pull_cap",
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
    "MODEL_PATH",
    "model_tag",
    "n_gpus_rollout",
    "train_prompt_mini_bsz",
    "concurrent_samples_per_replica",
    "KV_CACHE_DTYPE",
    "ATTN_BACKEND",
    "VLLM_LOG_STATS",
    "ROLLOUT_MAX_MODEL_LEN",
    "ROLLOUT_MAX_NUM_SEQS",
    "VERL_QWEN2_K_BIAS_CENTER",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
    "val_temperature",
    "val_top_p",
    "replay_tau",
    "replay_staleness_threshold",
    "staleness_threshold",
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
        assert r["pull_cap"] is False

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
        assert "pull-cap" not in default and "pull-cap" not in name

    def test_emulation_tag(self, script):
        name = compose(script, VERL_GPU_MEM_CAP_GB="80", gpu_memory_utilization="0.5")["trainer"]["experiment_name"]
        assert "h100-emu-80gb-gmu0.5" in name

    def test_logs_go_to_log_dir(self, script):
        c = compose(script)
        assert c["trainer"]["default_local_dir"].endswith("/logs")
        # per-update rollout dumps (<step>.jsonl) are off
        assert c["trainer"]["rollout_data_dir"] is None

    def test_hydra_record_goes_to_log_dir(self, script, tmp_path):
        # hydra/config.yaml, hydra/overrides.yaml and the driver log land next to tensorboard/ and global_step_N/
        argv = _argv_seen_by_python(tmp_path, {}, script=script)
        log_dir = str(tmp_path / "logs")
        assert f"hydra.run.dir={log_dir}" in argv
        assert "hydra.output_subdir=hydra" in argv
        assert f"trainer.default_local_dir={log_dir}" in argv
        # the user's own overrides still come last and win
        assert argv.index("hydra.run.dir=" + log_dir) < argv.index("trainer.total_training_steps=5")

    def test_hydra_record_defaults_to_logs_exp_name(self, script, tmp_path):
        exp_name = "a/b with spaces"
        argv = _argv_seen_by_python(tmp_path, {"exp_name": exp_name}, script=script, log_dir=None, cwd=tmp_path)
        assert "hydra.run.dir=logs/a_b with spaces" in argv
        assert f"trainer.experiment_name={exp_name}" in argv
        assert (tmp_path / "logs" / "a_b with spaces").is_dir()

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


def _argv_seen_by_python(tmp_path, env, script=QWEN, log_dir="<tmp>/logs", cwd=REPO):
    """Run a wrapper with a stand-in python that prints its argv, one element per line."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "python"
    fake.write_text('#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    fake.chmod(0o755)
    base = {k: v for k, v in os.environ.items() if k not in SCRIPT_ENV_KNOBS and k != "log_dir"}
    base["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    if log_dir is not None:
        base["log_dir"] = log_dir.replace("<tmp>", str(tmp_path))
    base.update(env)
    proc = subprocess.run(
        ["bash", str(SCRIPTS / script), "trainer.total_training_steps=5"],
        cwd=cwd,
        env=base,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.splitlines()


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


# --------------------------------------------------------------------------- Qwen2.5-7B base, 3+5 layout


QWEN25_OWN = {
    "MODEL_PATH": "Qwen/Qwen2.5-7B",
    "model_tag": "Qwen2.5-7B",
    "n_gpus_rollout": "3",
    "train_prompt_mini_bsz": "35",
    "test_freq": "20",
    "save_freq": "20",
}


class TestQwen25Base3p5:
    def test_model_layout_and_cadence(self):
        c = compose(QWEN25_FP8_C8)
        assert c["actor_rollout_ref"]["model"]["path"] == "Qwen/Qwen2.5-7B"
        assert (c["rollout"]["n_gpus_per_node"], c["trainer"]["n_gpus_per_node"]) == (3, 5)
        assert c["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 35
        assert c["async_training"]["concurrent_samples_per_replica"] == 35
        assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (20, 20)
        assert c["actor_rollout_ref"]["model"].get("external_lib") is None  # Qwen2Bridge maps the QKV biases

    def test_name(self):
        name = compose(QWEN25_FP8_C8)["trainer"]["experiment_name"]
        assert " DAPO17K-AIME24 Qwen2.5-7B 3-5 tp1dp5 hdo B-35 token-mean " in name and "Qwen3" not in name
        assert name.endswith(" 0.1-wd warmup-12 dynbsz-10240 tis-C8 rollout-fp8 seed-1")

    def test_is_the_tis_c8_arm_with_only_the_requested_changes(self):
        assert compose(QWEN25_FP8_C8) == compose(QWEN_FP8_C8, **QWEN25_OWN)

    def test_exports_the_same_parameters_as_the_tis_c8_arm(self):
        def exports(script):
            return {ln.strip() for ln in (SCRIPTS / script).read_text().splitlines() if ln.startswith("export ")}

        own = tuple(f"export {k}=" for k in QWEN25_OWN)
        qwen25, qwen3 = exports(QWEN25_FP8_C8), exports(QWEN_FP8_C8)
        own_lines = {ln for ln in qwen25 if ln.startswith(own)}
        assert len(own_lines) == len(QWEN25_OWN)
        assert qwen25 - own_lines == {ln for ln in qwen3 if not ln.startswith(own)}

    def test_runs_the_fp8_wrapper_like_the_tis_c8_arm(self):
        def execs(script):
            return [ln for ln in (SCRIPTS / script).read_text().splitlines() if ln.lstrip().startswith("exec ")]

        assert execs(QWEN25_FP8_C8) == execs(QWEN_FP8_C8)
        assert QWEN_FP8 in execs(QWEN25_FP8_C8)[0]

    def test_knobs_still_apply(self):
        c = compose(QWEN25_FP8_C8, MODEL_PATH="/models/qwen2.5-7b", test_freq="10", DYNAMIC_BSZ="False")
        assert c["actor_rollout_ref"]["model"]["path"] == "/models/qwen2.5-7b"
        assert c["trainer"]["test_freq"] == 10 and c["trainer"]["save_freq"] == 20
        assert c["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
        assert " Qwen2.5-7B 3-5 tp1dp5 " in c["trainer"]["experiment_name"]

    def test_qwen3_arms_keep_their_name(self):
        for script in (QWEN, QWEN_FP8, QWEN_FP8_C8):
            assert " DAPO17K-AIME24 Qwen3-8B 5-3 tp1dp3 hdo B-33 " in compose(script)["trainer"]["experiment_name"]

    def test_trainer_rollouter_and_worker_accept_the_config(self, monkeypatch):
        from verl.experimental.fully_async_policy import fully_async_rollouter as mod
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
        from verl.utils.config import omega_conf_to_dataclass

        cfg = _cfg(QWEN25_FP8_C8)
        actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
        assert actor.loss_agg_mode == "token-mean" and actor.use_dynamic_bsz is True
        cfg.trainer.logger = ["console"]
        trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
        t = trainer_cls(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
        assert t.replay_enable
        assert t.replay_first_mini_size == 20  # 0.5 x 35 -> 18, raised to 20 (20 x 16 splits over dp=5)

        rollouter_cls = mod.FullyAsyncRollouter.__ray_metadata__.modified_class
        monkeypatch.setattr(mod, "create_rl_dataset", lambda *a, **k: [0] * 8)
        monkeypatch.setattr(mod, "create_rl_sampler", lambda *a, **k: None)
        monkeypatch.setattr(
            rollouter_cls, "_create_dataloader", lambda self, *a, **k: setattr(self, "train_dataloader", [0] * 8)
        )
        monkeypatch.setattr(rollouter_cls, "_init_dump_executor", lambda self: None)
        r = rollouter_cls(config=_cfg(QWEN25_FP8_C8), tokenizer=None)
        assert r.concurrency_ramp == [5, 12, 20] and r.ramp_first_size == 20


# --------------------------------------------------------------------------- H100 emulation defaults (fp8 wrappers)

FP8_WRAPPERS = (QWEN_FP8, QWEN_FP8_C8, QWEN25_FP8_C8)


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
        freq = 20 if script == QWEN25_FP8_C8 else 12
        assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (freq, freq)
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


# --------------------------------------------------------------------------- rollout engine knobs (vLLM)

# The explicit defaults of the "Rollout engine knobs" block: setting them must compose to the plain arm.
ENGINE_KNOB_DEFAULTS = {
    "KV_CACHE_DTYPE": "auto",
    "ATTN_BACKEND": "default",
    "VLLM_LOG_STATS": "False",
    "ROLLOUT_MAX_MODEL_LEN": "null",
}
KV_KEY = "actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype"
ATTN_KEY = "actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend"
ALL_ENGINE_KNOBS = {
    "KV_CACHE_DTYPE": "fp8_e4m3",
    "ATTN_BACKEND": "FLASHINFER",
    "VLLM_LOG_STATS": "True",
    "ROLLOUT_MAX_MODEL_LEN": "10240",
}


class TestRolloutEngineKnobs:
    @pytest.mark.parametrize("script", [QWEN, QWEN_FP8, QWEN_FP8_C8, PANGU])
    def test_defaults_leave_the_arm_untouched(self, script):
        c = compose(script)
        r = c["actor_rollout_ref"]["rollout"]
        assert r["engine_kwargs"]["vllm"] == {}
        assert r["disable_log_stats"] is True and r["max_model_len"] is None
        name = c["trainer"]["experiment_name"]
        assert " kv-" not in name and " attn-" not in name
        assert compose(script, **ENGINE_KNOB_DEFAULTS) == c

    @pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "fp8_e5m2"])
    def test_fp8_kv_cache_reaches_the_engine_kwargs_and_the_name(self, dtype):
        c = compose(QWEN_FP8_C8, KV_CACHE_DTYPE=dtype)
        assert c["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"] == {"kv_cache_dtype": dtype}
        assert c["trainer"]["experiment_name"].endswith(f" dynbsz-10240 tis-C8 rollout-fp8 kv-{dtype} seed-1")
        # the KV dtype is the only change: FP8 weights, TIS C=8, token-mean and the H100 budget stay as they were
        assert _differing(compose(QWEN_FP8_C8), c) == {KV_KEY}

    def test_attention_backend_pin(self):
        c = compose(QWEN, ATTN_BACKEND="FLASHINFER")
        assert c["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"] == {"attention_backend": "FLASHINFER"}
        assert c["trainer"]["experiment_name"].endswith(" 0.1-wd attn-flashinfer seed-1")
        assert _differing(compose(QWEN), c) == {ATTN_KEY}

    def test_kv_and_backend_tags_sit_between_quant_and_seed(self):
        c = compose(QWEN_FP8, KV_CACHE_DTYPE="fp8", ATTN_BACKEND="FLASH_ATTN")
        assert c["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"] == {
            "kv_cache_dtype": "fp8",
            "attention_backend": "FLASH_ATTN",
        }
        assert c["trainer"]["experiment_name"].endswith(" warmup-12 rollout-fp8 kv-fp8 attn-flash-attn seed-1")

    def test_log_stats_is_metrics_only_and_untagged(self):
        c = compose(QWEN, VLLM_LOG_STATS="True")
        assert c["actor_rollout_ref"]["rollout"]["disable_log_stats"] is False
        assert _differing(compose(QWEN), c) == {"actor_rollout_ref.rollout.disable_log_stats"}
        assert c["trainer"]["experiment_name"] == compose(QWEN)["trainer"]["experiment_name"]

    def test_max_model_len_is_untagged(self):
        c = compose(QWEN, ROLLOUT_MAX_MODEL_LEN="10240")
        assert c["actor_rollout_ref"]["rollout"]["max_model_len"] == 10240
        assert _differing(compose(QWEN), c) == {"actor_rollout_ref.rollout.max_model_len"}
        assert c["trainer"]["experiment_name"] == compose(QWEN)["trainer"]["experiment_name"]

    def test_trainer_rollouter_and_worker_accept_all_knobs(self, monkeypatch):
        from verl.experimental.fully_async_policy import fully_async_rollouter as mod
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import RolloutConfig

        cfg = _cfg(QWEN_FP8_C8, **ALL_ENGINE_KNOBS)
        rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout, dataclass_type=RolloutConfig)
        assert dict(rollout.engine_kwargs["vllm"]) == {"kv_cache_dtype": "fp8_e4m3", "attention_backend": "FLASHINFER"}
        assert rollout.disable_log_stats is False and rollout.max_model_len == 10240
        assert rollout.quantization == "fp8"
        cfg.trainer.logger = ["console"]
        trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
        t = trainer_cls(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
        assert t.replay_enable

        rollouter_cls = mod.FullyAsyncRollouter.__ray_metadata__.modified_class
        monkeypatch.setattr(mod, "create_rl_dataset", lambda *a, **k: [0] * 8)
        monkeypatch.setattr(mod, "create_rl_sampler", lambda *a, **k: None)
        monkeypatch.setattr(
            rollouter_cls, "_create_dataloader", lambda self, *a, **k: setattr(self, "train_dataloader", [0] * 8)
        )
        monkeypatch.setattr(rollouter_cls, "_init_dump_executor", lambda self: None)
        r = rollouter_cls(config=_cfg(QWEN_FP8_C8, **ALL_ENGINE_KNOBS), tokenizer=None)
        assert r.replay_mode and r.concurrency_ramp == [5, 12, 20]


@pytest.mark.parametrize("script", [QWEN, QWEN_FP8_C8])
@pytest.mark.parametrize(
    "env, message",
    [
        ({"KV_CACHE_DTYPE": "int8"}, "KV_CACHE_DTYPE must be"),
        ({"KV_CACHE_DTYPE": "FP8"}, "KV_CACHE_DTYPE must be"),
        ({"ATTN_BACKEND": "flashinfer"}, "ATTN_BACKEND must be"),
        ({"VLLM_LOG_STATS": "yes"}, "VLLM_LOG_STATS must be"),
        ({"ROLLOUT_MAX_MODEL_LEN": "8192"}, "must be >= max_prompt_length"),
        ({"ROLLOUT_MAX_MODEL_LEN": "abc"}, "positive integer"),
    ],
)
def test_invalid_engine_knobs_are_rejected(script, env, message, tmp_path):
    proc = _run(script, env, (), tmp_path)
    assert proc.returncode == 2
    assert message in proc.stderr


# --------------------------------------------------------------------------- Qwen2.5-7B base, 3+5, bf16 rollout, no warmup, tau 4 / k 16 / stale 1

QWEN25_BF16_C8 = (
    "grpo_novcpo_8gpu_dapo17k_3+5_resp8k_megatron_offload_replay_tau=4_k=16"
    "_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.9_bf16_tis-c8_token-mean_nowarmup_stale=1_qwen2.5-7b.sh"
)
QWEN25_FP8KV_PULLCAP = (
    "grpo_novcpo_8gpu_dapo17k_3+5_resp8k_megatron_offload_replay_tau=4_k=16"
    "_min-ess=1.1_ess-lr-scale=0.5_nu=1_fresh=0.9_fp8_kv-fp8_tis-c8_token-mean_nowarmup_stale=1_pull-cap_qwen2.5-7b.sh"
)
QWEN25_BF16_OWN = {
    "ROLLOUT_QUANT": "bf16",
    "lr_warmup_steps": "0",
    "replay_tau": "4",
    "replay_staleness_threshold": "16",
    "staleness_threshold": "1",
    "replay_min_fresh_ratio": "0.9",
    "concurrency_ramp": "null",
    "test_freq": "60",
    "save_freq": "60",
}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


class TestQwen25Bf16NoWarmup3p5:
    def test_bf16_rollout_and_no_warmup(self):
        c = compose(QWEN25_BF16_C8)
        rollout = c["actor_rollout_ref"]["rollout"]
        assert rollout["quantization"] is None and rollout["dtype"] == "bfloat16"
        assert rollout.get("engine_kwargs", {}).get("vllm", {}).get("kv_cache_dtype") in (None, "auto")
        assert c["actor_rollout_ref"]["actor"]["optim"]["lr_warmup_steps"] == 0
        assert c["algorithm"]["rollout_correction"]["rollout_is_threshold"] == 8
        assert c["actor_rollout_ref"]["actor"]["loss_agg_mode"] == "token-mean"
        assert c["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is True

    def test_staleness_settings(self):
        a = compose(QWEN25_BF16_C8)["async_training"]
        assert a["staleness_threshold"] == 1
        assert (a["replay_buffer"]["tau"], a["replay_buffer"]["staleness_threshold"]) == (4, 16)
        assert a["replay_buffer"]["requires_mini_batches"] == 0.5 and a["replay_buffer"]["min_fresh_ratio"] == 0.9
        assert a["concurrency_ramp"] is None

    def test_model_layout_and_cadence_are_the_fp8_arms(self):
        c = compose(QWEN25_BF16_C8)
        assert c["actor_rollout_ref"]["model"]["path"] == "Qwen/Qwen2.5-7B"
        assert (c["rollout"]["n_gpus_per_node"], c["trainer"]["n_gpus_per_node"]) == (3, 5)
        assert c["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 35
        assert c["async_training"]["concurrent_samples_per_replica"] == 35
        assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (60, 60)
        assert c["data"]["seed"] == 1 and c["actor_rollout_ref"]["rollout"]["seed"] == 1
        val = c["actor_rollout_ref"]["rollout"]["val_kwargs"]
        assert (val["temperature"], val["top_p"]) == (1.0, 1.0)

    def test_name(self):
        name = compose(QWEN25_BF16_C8)["trainer"]["experiment_name"]
        assert " DAPO17K-AIME24 Qwen2.5-7B 3-5 tp1dp5 hdo B-35 token-mean " in name and "Qwen3" not in name
        assert name.startswith("GRPO-noVCPO replay tau-4 k-16 stale-1 rmb-0.5 nu-1 fresh-0.9 ess-")
        assert " h100-emu-78gb-gmu0.5 DAPO17K-AIME24 " in name and " ramp-" not in name
        assert name.endswith(" 0.1-wd dynbsz-10240 tis-C8 val-t1.0-p1.0 seed-1")
        assert "warmup" not in name and "rollout-fp8" not in name and " kv-" not in name

    def test_is_the_fp8_arm_with_only_the_requested_changes(self):
        assert compose(QWEN25_BF16_C8) == compose(QWEN25_FP8_C8, **QWEN25_BF16_OWN)

    def test_differs_from_the_fp8_arm_only_there(self):
        fp8, bf16 = _flat(compose(QWEN25_FP8_C8)), _flat(compose(QWEN25_BF16_C8))
        assert {k for k in fp8 if fp8[k] != bf16.get(k)} | {k for k in bf16 if k not in fp8} == {
            "actor_rollout_ref.rollout.quantization",
            "actor_rollout_ref.actor.optim.lr_warmup_steps",
            "async_training.staleness_threshold",
            "async_training.replay_buffer.tau",
            "async_training.replay_buffer.staleness_threshold",
            "async_training.replay_buffer.min_fresh_ratio",
            "async_training.concurrency_ramp",
            "trainer.test_freq",
            "trainer.save_freq",
            "trainer.experiment_name",
            "actor_rollout_ref.rollout.trace.experiment_name",  # mirrors trainer.experiment_name
        }

    def test_exports_are_the_fp8_arms_with_the_two_changes_and_no_deepgemm(self):
        def exports(script):
            return {ln.strip() for ln in (SCRIPTS / script).read_text().splitlines() if ln.startswith("export ")}

        fp8, bf16 = exports(QWEN25_FP8_C8), exports(QWEN25_BF16_C8)
        assert bf16 - fp8 == {f"export {k}=${{{k}:-{v}}}" for k, v in QWEN25_BF16_OWN.items()}
        assert fp8 - bf16 == {
            "export ROLLOUT_QUANT=${ROLLOUT_QUANT:-fp8}",
            "export lr_warmup_steps=${lr_warmup_steps:-12}",
            "export test_freq=${test_freq:-20}",
            "export save_freq=${save_freq:-20}",
        }
        text = (SCRIPTS / QWEN25_BF16_C8).read_text()
        assert "DEEPGEMM_CUDA_HOME" not in text and not any(f"export {v}" in text for v in DEEPGEMM_VARS)

    def test_execs_the_base_arm_directly(self):
        execs = [ln for ln in (SCRIPTS / QWEN25_BF16_C8).read_text().splitlines() if ln.lstrip().startswith("exec ")]
        assert len(execs) == 1 and QWEN in execs[0] and "_fp8" not in execs[0]

    def test_emulation_cap_reaches_the_trainer_env_without_deepgemm(self, tmp_path):
        seen = _env_seen_by_python(
            tmp_path, {}, script=QWEN25_BF16_C8, names=("VERL_GPU_MEM_CAP_GB", "ROLLOUT_QUANT", *DEEPGEMM_VARS)
        )
        assert seen["VERL_GPU_MEM_CAP_GB"] == "78" and seen["ROLLOUT_QUANT"] == "bf16"
        assert all(seen[v] == "<unset>" for v in DEEPGEMM_VARS)

    def test_knobs_still_apply(self):
        c = compose(
            QWEN25_BF16_C8,
            ROLLOUT_QUANT="fp8",
            lr_warmup_steps="5",
            test_freq="10",
            gpu_memory_utilization="0.7",
            replay_tau="8",
            replay_staleness_threshold="32",
            staleness_threshold="32.0",
            replay_min_fresh_ratio="0.5",
            concurrency_ramp="[5, 12, 20]",
        )
        assert c["actor_rollout_ref"]["rollout"]["quantization"] == "fp8"
        a = c["async_training"]
        assert a["staleness_threshold"] == 32.0 and (a["replay_buffer"]["tau"], a["replay_buffer"]["staleness_threshold"]) == (8, 32)
        assert c["trainer"]["experiment_name"].startswith("GRPO-noVCPO replay tau-8 k-32 rmb-0.5 nu-1 fresh-0.5 ess-")
        assert c["actor_rollout_ref"]["actor"]["optim"]["lr_warmup_steps"] == 5
        assert c["trainer"]["test_freq"] == 10 and c["trainer"]["save_freq"] == 60
        assert c["async_training"]["replay_buffer"]["min_fresh_ratio"] == 0.5
        assert c["async_training"]["concurrency_ramp"] == [5, 12, 20] and " ramp-5-12-20 " in c["trainer"]["experiment_name"]
        assert c["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.7
        name = c["trainer"]["experiment_name"]
        assert " warmup-5 " in name and " rollout-fp8 " in name and " h100-emu-78gb-gmu0.7 " in name

    def test_trailing_overrides_win(self):
        c = compose(QWEN25_BF16_C8, args=("trainer.val_before_train=False", "actor_rollout_ref.actor.optim.lr=2e-6"))
        assert c["trainer"]["val_before_train"] is False
        assert c["actor_rollout_ref"]["actor"]["optim"]["lr"] == 2e-6

    def test_trainer_and_rollouter_accept_the_config(self, monkeypatch):
        from verl.experimental.fully_async_policy import fully_async_rollouter as mod
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
        from verl.utils.config import omega_conf_to_dataclass

        cfg = _cfg(QWEN25_BF16_C8)
        actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
        assert actor.loss_agg_mode == "token-mean" and actor.optim.lr_warmup_steps == 0
        cfg.trainer.logger = ["console"]
        trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
        t = trainer_cls(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
        assert t.replay_enable and t.replay_first_mini_size == 20

        rollouter_cls = mod.FullyAsyncRollouter.__ray_metadata__.modified_class
        monkeypatch.setattr(mod, "create_rl_dataset", lambda *a, **k: [0] * 8)
        monkeypatch.setattr(mod, "create_rl_sampler", lambda *a, **k: None)
        monkeypatch.setattr(
            rollouter_cls, "_create_dataloader", lambda self, *a, **k: setattr(self, "train_dataloader", [0] * 8)
        )
        monkeypatch.setattr(rollouter_cls, "_init_dump_executor", lambda self: None)
        r = rollouter_cls(config=_cfg(QWEN25_BF16_C8), tokenizer=None)
        assert not r.concurrency_ramp and r.ramp_first_size == 20
        assert r.staleness_threshold == 1 and r.required_samples == 35


class TestQwen25Fp8KvPullCap3p5:
    """The bf16 stale-1 arm with FP8 weights + FP8 KV (k-bias centring), the pull cap, max_num_seqs 512, max_model_len
    10240 and vLLM log stats; everything else identical."""

    def test_fp8_rollout_fp8_kv_and_pull_cap(self):
        c = compose(QWEN25_FP8KV_PULLCAP)
        rollout = c["actor_rollout_ref"]["rollout"]
        assert rollout["quantization"] == "fp8" and rollout["dtype"] == "bfloat16"
        assert rollout["engine_kwargs"]["vllm"]["kv_cache_dtype"] == "fp8_e4m3"
        assert rollout["max_num_seqs"] == 512 and rollout["max_model_len"] == 10240
        assert rollout["disable_log_stats"] is False
        a = c["async_training"]
        assert a["replay_buffer"]["pull_cap"] is True and a["replay_buffer"]["requires_mini_batches"] == 0.5
        assert a["staleness_threshold"] == 1 and a["replay_buffer"]["min_fresh_ratio"] == 0.9
        assert (a["replay_buffer"]["tau"], a["replay_buffer"]["staleness_threshold"]) == (4, 16)
        assert a["concurrency_ramp"] is None and a["concurrent_samples_per_replica"] == 35
        assert c["actor_rollout_ref"]["actor"]["optim"]["lr_warmup_steps"] == 0
        assert c["algorithm"]["rollout_correction"]["rollout_is_threshold"] == 8
        assert c["actor_rollout_ref"]["actor"]["loss_agg_mode"] == "token-mean"
        assert c["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is True

    def test_model_layout_and_cadence(self):
        c = compose(QWEN25_FP8KV_PULLCAP)
        assert c["actor_rollout_ref"]["model"]["path"] == "Qwen/Qwen2.5-7B"
        assert (c["rollout"]["n_gpus_per_node"], c["trainer"]["n_gpus_per_node"]) == (3, 5)
        assert c["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 35
        assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (60, 60)
        assert c["data"]["seed"] == 1 and c["actor_rollout_ref"]["rollout"]["seed"] == 1
        val = c["actor_rollout_ref"]["rollout"]["val_kwargs"]
        assert (val["temperature"], val["top_p"]) == (1.0, 1.0)

    def test_name(self):
        name = compose(QWEN25_FP8KV_PULLCAP)["trainer"]["experiment_name"]
        assert name.startswith("GRPO-noVCPO replay tau-4 k-16 stale-1 rmb-0.5 nu-1 fresh-0.9 pull-cap ess-")
        assert " h100-emu-78gb-gmu0.5 DAPO17K-AIME24 Qwen2.5-7B 3-5 tp1dp5 hdo B-35 token-mean " in name
        assert name.endswith(" 0.1-wd dynbsz-10240 tis-C8 rollout-fp8 kv-fp8_e4m3 val-t1.0-p1.0 seed-1")
        assert " ramp-" not in name and "warmup" not in name and " attn-" not in name and "Qwen3" not in name

    def test_differs_from_the_bf16_stale1_arm_only_there(self):
        bf16, fp8 = _flat(compose(QWEN25_BF16_C8)), _flat(compose(QWEN25_FP8KV_PULLCAP))
        assert {k for k in fp8 if fp8[k] != bf16.get(k)} | {k for k in bf16 if k not in fp8} == {
            "actor_rollout_ref.rollout.quantization",
            "actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype",
            "actor_rollout_ref.rollout.max_num_seqs",
            "actor_rollout_ref.rollout.max_model_len",
            "actor_rollout_ref.rollout.disable_log_stats",
            "async_training.replay_buffer.pull_cap",
            "trainer.experiment_name",
            "actor_rollout_ref.rollout.trace.experiment_name",  # mirrors trainer.experiment_name
        }

    def test_execs_the_fp8_tis_c8_wrapper_with_the_max_num_seqs_override_first(self):
        text = (SCRIPTS / QWEN25_FP8KV_PULLCAP).read_text()
        execs = [ln for ln in text.splitlines() if ln.lstrip().startswith("exec ")]
        assert len(execs) == 1 and QWEN_FP8_C8 in execs[0]
        assert 'actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" "$@"' in text

    def test_env_reaches_python_with_k_bias_centring_and_deepgemm(self, tmp_path):
        toolkit = _fake_toolkit(tmp_path)
        names = ("ROLLOUT_QUANT", "KV_CACHE_DTYPE", "VERL_QWEN2_K_BIAS_CENTER", "VERL_GPU_MEM_CAP_GB", *DEEPGEMM_VARS)
        seen = _env_seen_by_python(tmp_path, {"DEEPGEMM_CUDA_HOME": str(toolkit)}, script=QWEN25_FP8KV_PULLCAP, names=names)
        assert seen["ROLLOUT_QUANT"] == "fp8" and seen["KV_CACHE_DTYPE"] == "fp8_e4m3"
        assert seen["VERL_QWEN2_K_BIAS_CENTER"] == "1" and seen["VERL_GPU_MEM_CAP_GB"] == "78"
        assert seen["CUDA_HOME"] == str(toolkit) and seen["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] == "0"

    def test_knobs_still_apply(self):
        c = compose(
            QWEN25_FP8KV_PULLCAP,
            replay_pull_cap="False",
            KV_CACHE_DTYPE="auto",
            ROLLOUT_MAX_NUM_SEQS="256",
            staleness_threshold="2",
            VLLM_LOG_STATS="False",
        )
        rollout = c["actor_rollout_ref"]["rollout"]
        assert rollout["max_num_seqs"] == 256 and rollout["disable_log_stats"] is True
        assert rollout.get("engine_kwargs", {}).get("vllm", {}).get("kv_cache_dtype") in (None, "auto")
        assert c["async_training"]["replay_buffer"]["pull_cap"] is False and c["async_training"]["staleness_threshold"] == 2
        name = c["trainer"]["experiment_name"]
        assert " stale-2 " in name and "pull-cap" not in name and " kv-" not in name and " rollout-fp8 " in name

    def test_trailing_override_wins_over_max_num_seqs(self):
        c = compose(QWEN25_FP8KV_PULLCAP, args=("actor_rollout_ref.rollout.max_num_seqs=768",))
        assert c["actor_rollout_ref"]["rollout"]["max_num_seqs"] == 768

    def test_trainer_and_rollouter_accept_the_config(self, monkeypatch):
        from verl.experimental.fully_async_policy import fully_async_rollouter as mod
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

        cfg = _cfg(QWEN25_FP8KV_PULLCAP)
        cfg.trainer.logger = ["console"]
        trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
        t = trainer_cls(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
        assert t.replay_enable and t.replay_pull_cap is True and t.replay_first_mini_size == 20

        rollouter_cls = mod.FullyAsyncRollouter.__ray_metadata__.modified_class
        monkeypatch.setattr(mod, "create_rl_dataset", lambda *a, **k: [0] * 8)
        monkeypatch.setattr(mod, "create_rl_sampler", lambda *a, **k: None)
        monkeypatch.setattr(
            rollouter_cls, "_create_dataloader", lambda self, *a, **k: setattr(self, "train_dataloader", [0] * 8)
        )
        monkeypatch.setattr(rollouter_cls, "_init_dump_executor", lambda self: None)
        r = rollouter_cls(config=_cfg(QWEN25_FP8KV_PULLCAP), tokenizer=None)
        assert not r.concurrency_ramp and r.staleness_threshold == 1 and r.required_samples == 35


@pytest.mark.parametrize("script", [QWEN, QWEN25_FP8_C8])
def test_base_arm_tags_a_non_default_staleness_threshold(script):
    assert " stale-" not in compose(script)["trainer"]["experiment_name"]
    name = compose(script, staleness_threshold="4")["trainer"]["experiment_name"]
    assert " k-32 stale-4 rmb-0.5 " in name


# --------------------------------------------------------------------------- pull cap knob


class TestPullCapKnob:
    """replay_pull_cap=True: the trainer pulls at most one mini-batch per update (default off, tagged pull-cap)."""

    def test_on(self):
        c = compose(QWEN25_BF16_C8, replay_pull_cap="True")
        assert c["async_training"]["replay_buffer"]["pull_cap"] is True
        assert " fresh-0.9 pull-cap ess-" in c["trainer"]["experiment_name"]

    def test_changes_only_that_key(self):
        assert _differing(compose(QWEN25_BF16_C8), compose(QWEN25_BF16_C8, replay_pull_cap="True")) == {
            "async_training.replay_buffer.pull_cap"
        }

    def test_explicit_false_is_the_default(self):
        assert compose(QWEN25_BF16_C8, replay_pull_cap="False") == compose(QWEN25_BF16_C8)

    def test_composes_with_a_wider_quota_and_the_4p4_layout(self):
        c = compose(
            QWEN25_BF16_C8,
            replay_pull_cap="True",
            staleness_threshold="2",
            train_prompt_mini_bsz="32",
            n_gpus_rollout="4",
        )
        a = c["async_training"]
        assert a["replay_buffer"]["pull_cap"] is True and a["staleness_threshold"] == 2
        assert c["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 32
        assert (c["rollout"]["n_gpus_per_node"], c["trainer"]["n_gpus_per_node"]) == (4, 4)
        name = c["trainer"]["experiment_name"]
        assert " stale-2 " in name and " pull-cap " in name and " 4-4 " in name

    def test_trainer_accepts_the_composed_config(self):
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as actor_cls

        cls = actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls
        cfg = _cfg(QWEN25_BF16_C8, replay_pull_cap="True")
        cfg.trainer.logger = ["console"]  # no tensorboard_log/ in the cwd
        t = cls(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
        assert t.replay_pull_cap is True and t.replay_requires_mini_batches == 0.5

    def test_trainer_rejects_the_cap_with_a_multi_minibatch_watermark(self):
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as actor_cls

        cls = actor_cls.__ray_metadata__.modified_class if hasattr(actor_cls, "__ray_metadata__") else actor_cls
        cfg = _cfg(QWEN25_BF16_C8, replay_pull_cap="True", replay_requires_mini_batches="2")
        cfg.trainer.logger = ["console"]
        with pytest.raises(AssertionError, match="pull_cap needs requires_mini_batches <= 1"):
            cls(config=cfg, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
