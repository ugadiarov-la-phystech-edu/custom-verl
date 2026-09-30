# Agent Instructions for verl

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

### Lint & tests

```bash
pre-commit run --all-files   # ruff + ruff-format + mypy + repo sanity checks
# CPU tests, as CI runs them (needs the `test` extra for pytest-asyncio):
pytest -s -x --asyncio-mode=auto -o python_files='*_on_cpu.py' tests/
pytest -s --asyncio-mode=auto tests/<area>/test_foo_on_cpu.py::test_name
```

- Config edits under `verl/trainer/config/` require regenerating `_generated_*.yaml`
  via `scripts/generate_trainer_config.sh` (also a pre-commit hook); never hand-edit them.
- `tests/<area>` mirrors `verl/<area>`. Only `test_*_on_cpu.py` runs in CPU CI; other files
  assume a GPU. `special_*` folders (distributed, e2e, npu, sanity, standalone) have their own workflows.
- A `*_on_cpu.py` test needing vllm/sglang must `pytest.importorskip` the backend
  (megatron-core is in the `cpu` extra and may be imported directly).
- Sanity hooks enforce license headers, docstring coverage, `verl`/`SGLang` spelling, and no
  raw CUDA device calls in `verl/` — use `verl.utils.device` helpers.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

---

## 3. Architecture Map

- Entry point: `verl/trainer/main_ppo.py` (Hydra, `ppo_trainer.yaml`) → Ray `TaskRunnerV1`.
  The v0 path (`trainer.use_v1=False` → `main_ppo_v0.py` / `ppo/ray_trainer.py`) is deprecated.
- V1 trainers in `verl/trainer/ppo/v1/` are chosen by `trainer.v1.trainer_mode`
  (`sync`, `colocate_async`, `separate_async`) via `register_trainer` / `get_trainer_cls`.
  Rollout is async-only: the agent loop manager writes samples to TransferQueue, and the trainer consumes them.
- Training backends are pluggable engines in `verl/workers/engine/` (fsdp, megatron, veomni,
  torchtitan, ...), selected by the `model_engine` Hydra default. Rollout servers (vllm/sglang/trtllm)
  are in `verl/workers/rollout/`. Agent loops are in `verl/experimental/agent_loop/`.
- `verl/single_controller/` provides the Ray WorkerGroup/RPC layer; `verl/protocol.py` defines `DataProto`.
- Config = YAML groups in `verl/trainer/config/` + typed dataclasses in `verl/workers/config/`.
- `recipe/` is a git submodule (`verl-recipe`).

Traps that fail silently:

- Custom reward functions go under `reward.custom_reward_function.{path,name}`. The top-level
  `custom_reward_function.*` still composes but `main_ppo` ignores it.
- Megatron models are built only through Megatron-Bridge; the per-model converters in
  `verl/models/mcore/` are not on the training path. Unmapped parameters only log a warning, so
  weights missing from a bridge (e.g. Llama attention biases) are silently dropped. Patch the
  bridge from a module passed via `actor_rollout_ref.model.external_lib`
  (see `llama_attention_bias_bridge.py`).
- In the v1 `fit` loop, `save_checkpoint` is timed inside `step` while `testing` is outside it;
  derive training time as `step - save_checkpoint`.
- Megatron HF checkpoints land in `global_step_N/actor/model/huggingface/`.

## 4. Fork Conventions

- `examples/baselines/`: sync GRPO baseline scripts ported from the `custom_vcpo` fork; launch from
  the repo root (they reference `rewards/` by relative path). Each header's "PORT NOTES" lists the
  deviations from the verl 0.7 originals.
- `rewards/`: custom reward functions loaded by path, not importable package code.
- Example scripts under `examples/` must be named `run_<model>_<train-backend>.sh`
  (pre-commit `check-example-naming`).

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).
