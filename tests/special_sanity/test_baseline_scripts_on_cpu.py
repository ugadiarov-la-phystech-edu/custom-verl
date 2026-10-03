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
QWEN_TIS = "run_qwen3-8b_dapo17k_grpo_sync_tis_B128xn16_mini32_megatron.sh"
ALL = [QWEN, PANGU, ORZ]


# Script knobs that must not leak in from the caller's shell.
SCRIPT_ENV_KNOBS = {
    "VERL_GPU_MEM_CAP_GB",
    "SEED",
    "max_updates",
    "ROLLOUT_QUANT",
    "TIS",
    "TIS_THRESHOLD",
    "clip_ratio_low",
    "clip_ratio_high",
    "clip_ratio_c",
    "lr_warmup_steps",
    "test_freq",
    "save_freq",
    "gpu_memory_utilization",
    "weight_decay",
    "DYNAMIC_BSZ",
    "DYNAMIC_BSZ_MAX_TOKENS",
    "DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS",
    "DEEPGEMM_CUDA_HOME",
}


def _run(script, env=None, args=(), tmp=None):
    """Run a script with --cfg job --resolve; returns (returncode, stdout, stderr)."""
    workdir = Path(tmp or os.environ.get("PYTEST_BASELINE_TMP", "/tmp")) / "baseline_compose"
    bindir = workdir / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    # A wrapper, not a symlink: a symlinked venv interpreter outside the venv loses the venv.
    python3 = bindir / "python3"
    if not python3.exists():
        # Atomic: pytest-xdist workers share this directory.
        tmp_wrapper = bindir / f".python3.{os.getpid()}"
        tmp_wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        tmp_wrapper.chmod(0o755)
        os.replace(tmp_wrapper, python3)
    full_env = {k: v for k, v in os.environ.items() if k not in SCRIPT_ENV_KNOBS}
    full_env.update(
        {
            "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
            "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    full_env.update(env or {})
    for _ in range(2):
        proc = subprocess.run(
            ["bash", str(SCRIPTS / script), *args, "--cfg", "job", "--resolve"],
            cwd=workdir,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        # Retry once only if the Python interpreter itself crashed (bash reports 128 + SIGSEGV):
        # observed once under 8 pytest-xdist workers, never reproduced. Real errors are not retried.
        if proc.returncode != 128 + 11:
            break
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


# --------------------------------------------------------------------------- Qwen3-8B precision / TIS variants

# Keys dynamic batching changes: the actor/rollout flags and caps, plus the ref/critic keys that
# follow the actor through oc.select (inert: no ref model and no critic under GRPO without KL).
DYNBSZ_KEYS = {
    "actor_rollout_ref.actor.use_dynamic_bsz",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz",
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
    "critic.use_dynamic_bsz",
}

TOGGLE_KEYS = {
    "actor_rollout_ref.rollout.quantization",
    "algorithm.rollout_correction.rollout_is",
    "algorithm.rollout_correction.rollout_is_threshold",
}


def _diff(a, b):
    fa, fb = _flatten(a), _flatten(b)
    return {k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)}


class TestQwenPrecisionAndTis:
    def test_baseline_defaults_are_bf16_without_tis(self):
        c = compose(QWEN)
        assert c["actor_rollout_ref"]["rollout"]["quantization"] is None
        assert c["algorithm"]["rollout_correction"]["rollout_is"] is None
        name = c["trainer"]["experiment_name"]
        assert "tis-" not in name and "rollout-fp8" not in name and "rollout-int8" not in name

    def test_explicit_defaults_compose_identically_to_the_baseline(self):
        assert _diff(compose(QWEN), compose(QWEN, ROLLOUT_QUANT="bf16", TIS="False")) == set()

    def test_bf16_tis(self):
        c = compose(QWEN_TIS)
        rc = c["algorithm"]["rollout_correction"]
        assert rc["rollout_is"] == "token"
        assert rc["rollout_is_threshold"] == 8.0
        assert rc["bypass_mode"] is False
        assert rc["rollout_rs"] is None
        assert c["actor_rollout_ref"]["rollout"]["quantization"] is None
        assert c["actor_rollout_ref"]["rollout"]["calculate_log_probs"] is True
        assert c["actor_rollout_ref"]["rollout"]["logprobs_mode"] == "processed_logprobs"
        assert c["trainer"]["experiment_name"].endswith(
            " 0.1-wd clip-0.2-0.28-c10.0 warmup-3 dynbsz-10240 tis-C8 seed-1 h100-emu-76gb-gmu0.283"
        )

    def test_fp8_tis(self):
        c = compose(QWEN_TIS, ROLLOUT_QUANT="fp8")
        assert c["actor_rollout_ref"]["rollout"]["quantization"] == "fp8"
        assert c["algorithm"]["rollout_correction"]["rollout_is"] == "token"
        assert c["trainer"]["experiment_name"].endswith(" tis-C8 rollout-fp8 seed-1 h100-emu-76gb-gmu0.283")

    def test_int8_tis(self):
        c = compose(QWEN_TIS, ROLLOUT_QUANT="int8")
        assert c["actor_rollout_ref"]["rollout"]["quantization"] == "int8"
        assert c["algorithm"]["rollout_correction"]["rollout_is"] == "token"
        assert c["trainer"]["experiment_name"].endswith(" tis-C8 rollout-int8 seed-1 h100-emu-76gb-gmu0.283")

    def test_int8_is_accepted_by_the_rollout_config(self):
        from omegaconf import OmegaConf

        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import RolloutConfig

        rollout = OmegaConf.create(compose(QWEN_TIS, ROLLOUT_QUANT="int8")["actor_rollout_ref"]["rollout"])
        assert omega_conf_to_dataclass(rollout, dataclass_type=RolloutConfig).quantization == "int8"

    def test_tis_script_uses_flashrl_parameters(self):
        c = compose(QWEN_TIS)
        actor = c["actor_rollout_ref"]["actor"]
        assert c["algorithm"]["rollout_correction"]["rollout_is_threshold"] == 8.0
        assert (actor["clip_ratio"], actor["clip_ratio_low"], actor["clip_ratio_high"]) == (0.2, 0.2, 0.28)
        assert actor["clip_ratio_c"] == 10.0
        assert actor["optim"]["lr_warmup_steps"] == 3  # FlashRL: 10
        assert actor["optim"]["weight_decay"] == 0.1
        assert actor["optim"]["lr_decay_style"] == "constant"
        assert actor["optim"]["lr"] == 1e-6

    def test_bf16_tis_differs_from_baseline_only_by_the_flashrl_knobs_and_dynbsz(self):
        expected = (
            DYNBSZ_KEYS
            | {
                "algorithm.rollout_correction.rollout_is",
                "algorithm.rollout_correction.rollout_is_threshold",
                "actor_rollout_ref.actor.clip_ratio_high",
                "actor_rollout_ref.actor.clip_ratio_c",
                "actor_rollout_ref.actor.optim.lr_warmup_steps",
                "actor_rollout_ref.actor.optim.weight_decay",
                "trainer.test_freq",
                "trainer.save_freq",
                "actor_rollout_ref.rollout.gpu_memory_utilization",
            }
            | NAME_DERIVED_KEYS
        )
        assert _diff(compose(QWEN), compose(QWEN_TIS)) == expected

    def test_tis_script_with_baseline_knobs_is_a_tis_only_ablation(self):
        ablation = compose(
            QWEN_TIS,
            TIS_THRESHOLD="2.0",
            clip_ratio_high="0.2",
            clip_ratio_c="3.0",
            lr_warmup_steps="0",
            weight_decay="0.01",
            DYNAMIC_BSZ="False",
            test_freq="2",
            save_freq="2",
            VERL_GPU_MEM_CAP_GB="",
            gpu_memory_utilization="0.5",
        )
        assert _diff(compose(QWEN), ablation) == {"algorithm.rollout_correction.rollout_is"} | NAME_DERIVED_KEYS
        assert ablation["trainer"]["experiment_name"].endswith(" 0.01-wd tis-C2.0 seed-1")

    def test_baseline_name_unchanged_by_new_knobs(self):
        name = compose(QWEN)["trainer"]["experiment_name"]
        assert "clip-" not in name and "warmup-" not in name
        assert name.endswith(" 0.01-wd seed-1")

    @pytest.mark.parametrize(
        "env, tag",
        [
            ({"clip_ratio_high": "0.28"}, " clip-0.2-0.28-c3.0"),
            ({"clip_ratio_c": "10.0"}, " clip-0.2-0.2-c10.0"),
            ({"lr_warmup_steps": "5"}, " warmup-5"),
        ],
    )
    def test_nondefault_knobs_are_tagged_on_the_baseline(self, env, tag):
        assert tag in compose(QWEN, **env)["trainer"]["experiment_name"]

    @pytest.mark.parametrize("quant", ["fp8", "int8"])
    def test_quantized_tis_differs_from_bf16_tis_only_by_quantization(self, quant):
        diff = _diff(compose(QWEN_TIS), compose(QWEN_TIS, ROLLOUT_QUANT=quant))
        assert diff == {"actor_rollout_ref.rollout.quantization"} | NAME_DERIVED_KEYS

    def test_int8_and_fp8_differ_only_by_quantization(self):
        diff = _diff(compose(QWEN_TIS, ROLLOUT_QUANT="fp8"), compose(QWEN_TIS, ROLLOUT_QUANT="int8"))
        assert diff == {"actor_rollout_ref.rollout.quantization"} | NAME_DERIVED_KEYS

    def test_tis_script_is_the_baseline_with_tis_and_flashrl_knobs(self):
        flashrl = dict(
            TIS="True",
            TIS_THRESHOLD="8",
            clip_ratio_high="0.28",
            clip_ratio_c="10.0",
            lr_warmup_steps="3",
            weight_decay="0.1",
            DYNAMIC_BSZ="True",
            test_freq="3",
            save_freq="3",
            VERL_GPU_MEM_CAP_GB="76",
            gpu_memory_utilization="0.283",
        )
        assert _diff(compose(QWEN_TIS), compose(QWEN, **flashrl)) == set()

    def test_tis_script_cadence(self):
        c = compose(QWEN_TIS)
        assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (3, 3)
        c = compose(QWEN_TIS, test_freq="5", save_freq="4")
        assert (c["trainer"]["test_freq"], c["trainer"]["save_freq"]) == (5, 4)

    def test_tis_script_honours_explicit_env_values(self):
        # the wrapper only sets defaults; explicit env values win
        c = compose(QWEN_TIS, TIS="False", weight_decay="0.05")
        assert c["algorithm"]["rollout_correction"]["rollout_is"] is None
        assert c["actor_rollout_ref"]["actor"]["optim"]["weight_decay"] == 0.05

    def test_threshold_flows_to_config_and_name(self):
        c = compose(QWEN_TIS, TIS_THRESHOLD="2")
        assert c["algorithm"]["rollout_correction"]["rollout_is_threshold"] == 2.0
        assert " tis-C2 seed-1" in c["trainer"]["experiment_name"]

    def test_wrapper_forwards_env_knobs_and_cli_overrides(self):
        c = compose(QWEN_TIS, args=("trainer.total_training_steps=3",), SEED="7", ROLLOUT_QUANT="fp8")
        assert c["trainer"]["total_training_steps"] == 3
        assert c["actor_rollout_ref"]["rollout"]["seed"] == 7
        assert c["trainer"]["experiment_name"].endswith("rollout-fp8 seed-7 h100-emu-76gb-gmu0.283")

    def test_tis_script_emulates_an_h100_by_default(self):
        c = compose(QWEN_TIS)
        assert c["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.283
        assert c["trainer"]["experiment_name"].endswith(" seed-1 h100-emu-76gb-gmu0.283")

    def test_tis_cap_reaches_the_trainer_env(self, tmp_path):
        assert _env_seen_by_python(QWEN_TIS, tmp_path, {}, ["VERL_GPU_MEM_CAP_GB"])["VERL_GPU_MEM_CAP_GB"] == "76"

    def test_tis_script_on_a_real_h100(self, tmp_path):
        # an empty VERL_GPU_MEM_CAP_GB switches the cap off; 0.5 restores the H100 vLLM budget
        c = compose(QWEN_TIS, VERL_GPU_MEM_CAP_GB="", gpu_memory_utilization="0.5")
        assert c["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.5
        assert "h100-emu" not in c["trainer"]["experiment_name"]
        seen = _env_seen_by_python(QWEN_TIS, tmp_path, {"VERL_GPU_MEM_CAP_GB": ""}, ["VERL_GPU_MEM_CAP_GB"])
        assert seen["VERL_GPU_MEM_CAP_GB"] == ""

    def test_emulation_tag_still_last(self):
        c = compose(QWEN_TIS, ROLLOUT_QUANT="fp8", VERL_GPU_MEM_CAP_GB="108", gpu_memory_utilization="0.283")
        assert c["trainer"]["experiment_name"].endswith("tis-C8 rollout-fp8 seed-1 h100-emu-108gb-gmu0.283")

    def test_toggles_do_not_leak_into_other_arms(self):
        for script in (PANGU, ORZ):
            c = compose(script)
            assert c["actor_rollout_ref"]["rollout"]["quantization"] is None
            assert c["algorithm"]["rollout_correction"]["rollout_is"] is None

    @pytest.mark.parametrize("quant", ["fp8", "int8"])
    def test_quantized_without_tis_is_allowed_with_a_warning(self, quant, tmp_path):
        rc, out, err = _run(QWEN, env={"ROLLOUT_QUANT": quant}, tmp=tmp_path)
        assert rc == 0, err[-2000:]
        assert f"WARNING: ROLLOUT_QUANT={quant} without TIS" in err
        assert yaml.safe_load(out)["actor_rollout_ref"]["rollout"]["quantization"] == quant

    @pytest.mark.parametrize("quant", ["fp8", "int8"])
    def test_no_warning_with_tis(self, quant, tmp_path):
        rc, _, err = _run(QWEN_TIS, env={"ROLLOUT_QUANT": quant}, tmp=tmp_path)
        assert rc == 0 and "WARNING: ROLLOUT_QUANT" not in err

    def test_no_warning_for_bf16_without_tis(self, tmp_path):
        rc, _, err = _run(QWEN, env={}, tmp=tmp_path)
        assert rc == 0 and "WARNING: ROLLOUT_QUANT" not in err


@pytest.mark.parametrize("script", [QWEN, QWEN_TIS])
@pytest.mark.parametrize(
    "env, message",
    [
        ({"ROLLOUT_QUANT": "int4"}, "ROLLOUT_QUANT must be"),
        ({"ROLLOUT_QUANT": "FP8"}, "ROLLOUT_QUANT must be"),
        ({"ROLLOUT_QUANT": "INT8"}, "ROLLOUT_QUANT must be"),
        ({"TIS": "yes"}, "TIS must be"),
        ({"TIS_THRESHOLD": "abc"}, "TIS_THRESHOLD must be"),
        ({"TIS_THRESHOLD": "0"}, "TIS_THRESHOLD must be"),
        ({"TIS_THRESHOLD": "-1"}, "TIS_THRESHOLD must be"),
    ],
)
def test_invalid_toggles_fail_fast(script, env, message, tmp_path):
    rc, _, err = _run(script, env=env, tmp=tmp_path)
    assert rc == 2, err[-2000:]
    assert message in err


# --------------------------------------------------------------------------- dynamic batch size


class TestDynamicBatchSize:
    def test_baseline_default_is_off_and_unchanged(self):
        c = compose(QWEN)
        assert c["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
        assert c["actor_rollout_ref"]["rollout"]["log_prob_use_dynamic_bsz"] is False
        assert "dynbsz" not in c["trainer"]["experiment_name"]
        assert _diff(c, compose(QWEN, DYNAMIC_BSZ="False")) == set()

    def test_on_sets_flags_and_caps_consistently(self):
        c = compose(QWEN, DYNAMIC_BSZ="True")
        actor, rollout = c["actor_rollout_ref"]["actor"], c["actor_rollout_ref"]["rollout"]
        # engine_workers.py asserts these two flags are equal and both caps set
        assert actor["use_dynamic_bsz"] is rollout["log_prob_use_dynamic_bsz"] is True
        assert actor["ppo_max_token_len_per_gpu"] == 10240
        assert rollout["log_prob_max_token_len_per_gpu"] == 10240
        assert _diff(compose(QWEN), c) == DYNBSZ_KEYS | NAME_DERIVED_KEYS
        assert c["trainer"]["experiment_name"].endswith(" 0.01-wd dynbsz-10240 seed-1")

    def test_default_cap_is_one_full_sequence(self):
        c = compose(QWEN, DYNAMIC_BSZ="True")
        assert c["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == (
            c["data"]["max_prompt_length"] + c["data"]["max_response_length"]
        )

    def test_caps_are_configurable(self):
        c = compose(QWEN, DYNAMIC_BSZ="True", DYNAMIC_BSZ_MAX_TOKENS="16384", DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS="20480")
        assert c["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == 16384
        assert c["actor_rollout_ref"]["rollout"]["log_prob_max_token_len_per_gpu"] == 20480
        assert " dynbsz-16384 " in c["trainer"]["experiment_name"]

    def test_log_prob_cap_follows_training_cap(self):
        c = compose(QWEN, DYNAMIC_BSZ="True", DYNAMIC_BSZ_MAX_TOKENS="12288")
        assert c["actor_rollout_ref"]["rollout"]["log_prob_max_token_len_per_gpu"] == 12288

    def test_tis_script_enables_it(self):
        c = compose(QWEN_TIS)
        assert c["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is True
        assert c["actor_rollout_ref"]["rollout"]["log_prob_use_dynamic_bsz"] is True
        assert compose(QWEN_TIS, DYNAMIC_BSZ="False")["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False

    def test_passes_verl_config_validation(self):
        from omegaconf import OmegaConf

        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config

        for c in (compose(QWEN_TIS), compose(QWEN_TIS, ROLLOUT_QUANT="fp8"), compose(QWEN_TIS, ROLLOUT_QUANT="int8")):
            cfg = OmegaConf.create(c)
            validate_config(config=cfg, use_reference_policy=need_reference_policy(cfg), use_critic=need_critic(cfg))

    def test_toggle_does_not_leak_into_other_arms(self):
        for script in (PANGU, ORZ):
            assert compose(script)["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False


@pytest.mark.parametrize("script", [QWEN, QWEN_TIS])
@pytest.mark.parametrize(
    "env, message",
    [
        ({"DYNAMIC_BSZ": "maybe"}, "DYNAMIC_BSZ must be"),
        ({"DYNAMIC_BSZ": "True", "DYNAMIC_BSZ_MAX_TOKENS": "8192"}, "must be >= max_prompt_length"),
        ({"DYNAMIC_BSZ": "True", "DYNAMIC_BSZ_LOG_PROB_MAX_TOKENS": "4096"}, "must be >= max_prompt_length"),
        ({"DYNAMIC_BSZ": "True", "DYNAMIC_BSZ_MAX_TOKENS": "10k"}, "positive integers"),
        ({"DYNAMIC_BSZ": "True", "DYNAMIC_BSZ_MAX_TOKENS": "0"}, "positive integers"),
    ],
)
def test_invalid_dynamic_bsz_settings_fail_fast(script, env, message, tmp_path):
    rc, _, err = _run(script, env=env, tmp=tmp_path)
    assert rc == 2, err[-2000:]
    assert message in err


@pytest.mark.parametrize("script", [QWEN, QWEN_TIS])
def test_warmup_longer_than_a_short_run_fails_fast(script, tmp_path):
    # 40 updates = 10 rollout steps; a 10-step warmup -> Megatron would assert at init
    env = {"max_updates": "40", "lr_warmup_steps": "10"}
    rc, _, err = _run(script, env=env, tmp=tmp_path)
    assert rc == 2, err[-2000:]
    assert "lr_warmup_steps=10 must be < the 10 rollout steps" in err


def test_warmup_shorter_than_the_run_is_fine():
    c = compose(QWEN_TIS, max_updates="16")  # 4 rollout steps > 3 warmup steps
    assert c["trainer"]["total_training_steps"] == 4
    assert c["actor_rollout_ref"]["actor"]["optim"]["lr_warmup_steps"] == 3


def test_tis_default_warmup_needs_more_than_three_rollout_steps(tmp_path):
    rc, _, err = _run(QWEN_TIS, env={"max_updates": "12"}, tmp=tmp_path)  # 3 rollout steps = 3 warmup steps
    assert rc == 2, err[-2000:]
    assert "lr_warmup_steps=3 must be < the 3 rollout steps" in err


# --------------------------------------------------------------------------- DeepGEMM env (TIS script)

DEEPGEMM_VARS = ("CUDA_HOME", "DG_JIT_CACHE_DIR", "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER")


def _env_seen_by_python(script, tmp_path, env=None, names=None):
    """Run a script with a stand-in python3 that prints the given env vars (default: the DeepGEMM ones)."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "python3"
    lines = "".join(f'echo "{v}=${{{v}-<unset>}}"\n' for v in (names or DEEPGEMM_VARS))
    fake.write_text("#!/bin/sh\n" + lines)
    fake.chmod(0o755)
    base = {k: v for k, v in os.environ.items() if k not in SCRIPT_ENV_KNOBS and k not in DEEPGEMM_VARS}
    base.pop("DEEPGEMM_CUDA_HOME", None)
    base["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    base.update(env or {})
    proc = subprocess.run(
        ["bash", str(SCRIPTS / script)], cwd=tmp_path, env=base, capture_output=True, text=True, timeout=120
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


class TestDeepGemmEnv:
    def test_exports_all_three_when_the_toolkit_exists(self, tmp_path):
        toolkit = _fake_toolkit(tmp_path)
        seen = _env_seen_by_python(QWEN_TIS, tmp_path, {"DEEPGEMM_CUDA_HOME": str(toolkit)})
        assert seen == {
            "CUDA_HOME": str(toolkit),
            "DG_JIT_CACHE_DIR": "/home/jovyan/ugadiarov/cache/deep_gemm",
            "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0",
        }

    def test_exports_nothing_where_the_toolkit_is_absent(self, tmp_path):
        seen = _env_seen_by_python(QWEN_TIS, tmp_path, {"DEEPGEMM_CUDA_HOME": str(tmp_path / "missing")})
        assert set(seen.values()) == {"<unset>"}

    def test_a_toolkit_dir_without_nvcc_is_ignored(self, tmp_path):
        (tmp_path / "cuda-no-nvcc" / "bin").mkdir(parents=True)
        seen = _env_seen_by_python(QWEN_TIS, tmp_path, {"DEEPGEMM_CUDA_HOME": str(tmp_path / "cuda-no-nvcc")})
        assert set(seen.values()) == {"<unset>"}

    def test_an_existing_cuda_home_is_left_alone(self, tmp_path):
        # a full system toolkit: keep it, and keep vLLM's FlashInfer small-batch path enabled
        toolkit = _fake_toolkit(tmp_path)
        seen = _env_seen_by_python(
            QWEN_TIS, tmp_path, {"DEEPGEMM_CUDA_HOME": str(toolkit), "CUDA_HOME": "/usr/local/cuda-12.9"}
        )
        assert seen == {
            "CUDA_HOME": "/usr/local/cuda-12.9",
            "DG_JIT_CACHE_DIR": "<unset>",
            "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "<unset>",
        }

    def test_explicit_cache_dir_and_flashinfer_flag_win(self, tmp_path):
        toolkit = _fake_toolkit(tmp_path)
        seen = _env_seen_by_python(
            QWEN_TIS,
            tmp_path,
            {
                "DEEPGEMM_CUDA_HOME": str(toolkit),
                "DG_JIT_CACHE_DIR": "/scratch/dg",
                "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "1",
            },
        )
        assert seen["DG_JIT_CACHE_DIR"] == "/scratch/dg"
        assert seen["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] == "1"
        assert seen["CUDA_HOME"] == str(toolkit)

    def test_baseline_script_exports_nothing(self, tmp_path):
        toolkit = _fake_toolkit(tmp_path)
        seen = _env_seen_by_python(QWEN, tmp_path, {"DEEPGEMM_CUDA_HOME": str(toolkit)})
        assert set(seen.values()) == {"<unset>"}

    def test_config_is_unaffected(self, tmp_path):
        toolkit = _fake_toolkit(tmp_path)
        assert _diff(compose(QWEN_TIS), compose(QWEN_TIS, DEEPGEMM_CUDA_HOME=str(toolkit))) == set()
