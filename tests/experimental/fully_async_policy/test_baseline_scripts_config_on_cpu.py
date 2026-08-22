# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Every ``run/baselines`` launch script must compose against the fully-async recipe.

These scripts pass ~120 Hydra overrides each. The failure mode that actually bites is a single
override that no longer resolves -- a renamed key, a struct error, or a quoting bug that only
manifests in one script -- and it is not discovered until a multi-GPU job dies at startup. So each
script is executed here under a stub ``python`` that records its argv, and that argv is composed
against the real recipe yaml.

CPU-only: no Ray, no GPU, no model. The scripts are run with ``cwd`` inside ``tmp_path`` so their
``mkdir -p logs/...`` lands in the temporary directory.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "run" / "baselines"
RECIPE_CONFIG_DIR = REPO_ROOT / "verl" / "experimental" / "fully_async_policy" / "config"

SCRIPTS = sorted(SCRIPT_DIR.glob("*.sh")) if SCRIPT_DIR.is_dir() else []

pytestmark = pytest.mark.skipif(
    not SCRIPTS or shutil.which("bash") is None,
    reason="run/baselines scripts or bash unavailable",
)


def _capture_overrides(script: Path, tmp_path: Path) -> list[str]:
    """Run the script with a stub ``python`` on PATH and return the Hydra overrides it passes."""
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "python"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)

    # SMOKE_VERIFY=0: the smoke script's post-run artifact checks must not run against a
    # stub `python` that never produced a checkpoint.
    env = dict(os.environ, PATH=f"{stub_dir}:{os.environ.get('PATH', '')}", SMOKE_VERIFY="0")
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{script.name} exited {proc.returncode}\n{proc.stderr[-2000:]}"

    argv = [line for line in proc.stdout.splitlines() if line.strip()]
    assert argv[0] == "-m", f"{script.name}: expected `python -m <module>`, got {argv[:2]}"
    assert argv[1] == "verl.experimental.fully_async_policy.fully_async_main", argv[1]
    assert argv[2].startswith("--config-name="), argv[2]
    # The recipe differs by backend -- megatron scripts select the megatron yaml, fsdp2 scripts the
    # plain one -- so compose against whichever the script itself asked for, never a fixed name.
    config_name = argv[2].split("=", 1)[1]
    return config_name, argv[3:]


def _compose(config_name: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(RECIPE_CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    """{script name: composed config} -- each script is executed exactly once."""
    out = {}
    for script in SCRIPTS:
        tmp = tmp_path_factory.mktemp(script.stem[:24].replace("=", "_"))
        out[script.name] = _compose(*_capture_overrides(script, tmp))
    return out


def test_scripts_are_discovered():
    assert SCRIPTS, "no scripts found in run/baselines"


@pytest.mark.parametrize("name", [s.name for s in SCRIPTS])
def test_script_composes(composed, name):
    """The whole override list resolves against the recipe -- no struct errors, no stale keys."""
    assert composed[name] is not None


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_stdout_is_unbuffered(script):
    """The driver's own prints are small and sit in an unflushed block buffer otherwise.

    Verified on the remote: with fd 1 redirected to a file, a process' stdout only reaches disk once
    8 KB accumulates, so the driver's startup prints are lost for the life of the run. This does not
    on its own restore the Ray-forwarded actor output -- that is what the file backend above is for.
    """
    assert "export PYTHONUNBUFFERED=1" in script.read_text(), f"{script.name} must export PYTHONUNBUFFERED=1"


# Which save_contents yields a weights-only checkpoint depends on the training backend, and the
# two are inverses of each other -- see test_export_only_checkpoint_policy.
EXPORT_ONLY_CONTENTS = {"megatron": ["model", "hf_model"], "fsdp2": ["hf_model"]}


@pytest.mark.parametrize("name", [s.name for s in SCRIPTS])
def test_export_only_checkpoint_policy(composed, name):
    """Weights-only checkpoints, kept forever, never resumed from.

    The token that produces the HF export flips with the backend:

    * megatron/mbridge -- ``should_save_model`` drives ``bridge.save_hf_weights``
      (``megatron_checkpoint_manager.py:753``) while the ``hf_model`` branch at ``:820`` is
      gated on ``not use_hf_checkpoint`` and never runs. So ``model`` is mandatory and
      ``hf_model`` alone would write no weights at all.
    * fsdp2 -- ``should_save_hf_model`` is an independent branch
      (``fsdp_checkpoint_manager.py:341-391``) that gathers its own full state dict and calls
      ``save_pretrained``. ``model`` there writes per-rank sharded ``model_world_size_*.pt``
      files instead, which ``resume_mode=disable`` never reads.
    """
    cfg = composed[name]
    ckpt = cfg.actor_rollout_ref.actor.checkpoint
    strategy = cfg.actor_rollout_ref.actor.strategy

    assert cfg.trainer.max_actor_ckpt_to_keep is None, "null disables both retention trims"
    assert cfg.trainer.resume_mode == "disable"
    assert strategy in EXPORT_ONLY_CONTENTS, f"unhandled backend {strategy!r}"
    assert list(ckpt.save_contents) == EXPORT_ONLY_CONTENTS[strategy]
    assert "optimizer" not in ckpt.save_contents
    assert "extra" not in ckpt.save_contents
    # load_contents follows save_contents by interpolation in actor.yaml.
    assert list(ckpt.load_contents) == list(ckpt.save_contents)


@pytest.mark.parametrize("name", [s.name for s in SCRIPTS])
def test_fsdp_strategy_is_set_on_both_keys(composed, name):
    """``actor.strategy`` is the load-bearing one, and it silently overwrites the other.

    ``FSDPActorConfig.__post_init__`` (``verl/workers/config/actor.py:310-317``) aliases
    ``self.engine`` to ``self.fsdp_config`` and then copies ``self.strategy`` onto it, so a script
    that sets only ``fsdp_config.strategy=fsdp2`` runs FSDP1 with no error. Several in-tree
    example scripts are wrong in exactly this way; the baselines must set both.
    """
    cfg = composed[name]
    strategy = cfg.actor_rollout_ref.actor.strategy
    if not strategy.startswith("fsdp"):
        pytest.skip(f"{strategy} backend has no fsdp_config")
    assert cfg.actor_rollout_ref.actor.fsdp_config.strategy == strategy


@pytest.mark.parametrize("name", [s.name for s in SCRIPTS])
def test_stop_the_world_accounting_enabled(composed, name):
    """Validation and saves must be pure time translations for cumulative_training_time."""
    at = composed[name].async_training
    assert at.serialize_validation is True
    assert at.pause_generation_during_save is True
    assert composed[name].actor_rollout_ref.rollout.calculate_log_probs is True


@pytest.mark.parametrize("name", [s.name for s in SCRIPTS])
def test_batch_shape_invariants(composed, name):
    """The B-33x1 layout: 33 prompts x n=16 must divide evenly across the trainer GPUs."""
    cfg = composed[name]
    mini_bsz = cfg.actor_rollout_ref.actor.ppo_mini_batch_size
    n = cfg.actor_rollout_ref.rollout.n
    trainer_gpus = cfg.trainer.nnodes * cfg.trainer.n_gpus_per_node
    assert (mini_bsz * n) % trainer_gpus == 0, f"{mini_bsz}*{n} not divisible by {trainer_gpus}"
    assert cfg.data.train_batch_size == 0, "asserted by the rollouter"
    assert cfg.data.gen_batch_size == 1, "asserted by the rollouter"
    assert cfg.trainer.test_freq != 0, "test_freq=0 raises ZeroDivisionError in the trainer"


@pytest.mark.parametrize("name", [s.name for s in SCRIPTS])
def test_loss_mode_is_internally_consistent(composed, name):
    """bypass_mode=True only selects the bypass loss if the worker-side loss_mode says so --
    the driver-side injection never reaches the workers."""
    cfg = composed[name]
    rc = cfg.algorithm.rollout_correction
    loss_mode = cfg.actor_rollout_ref.actor.policy_loss.loss_mode

    if loss_mode == "bypass_mode":
        assert rc.bypass_mode is True
        # the worker needs its own copy of the correction settings
        worker_rc = cfg.actor_rollout_ref.actor.policy_loss.rollout_correction
        assert worker_rc.loss_type == rc.loss_type
        assert worker_rc.rollout_is == rc.rollout_is
    elif rc.bypass_mode is False:
        # decoupled: driver-side IS weights ride the batch, vanilla loss applies them
        assert loss_mode == "vanilla"
        assert rc.rollout_is is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
