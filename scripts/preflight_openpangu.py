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

"""CPU-only preflight for running a custom-code model (openPangu-Embedded-7B) through the
fully-async GRPO recipe.

Motivation: several ways this can go wrong are *silent* -- they yield a running multi-day job with
meaningless numbers rather than an error. This script settles each of them before a GPU is booked:

* verl passes no ``stop`` / ``stop_token_ids`` / ``eos_token_id`` to vLLM
  (``agent_loop.py:496-502``, ``vllm_async_server.py:503-505``); termination depends entirely on the
  model's own eos. Get it wrong and every rollout runs to the length cap and scores -1.0.
* ``monkey_patch_compute_logits`` (``vllm_rollout/utils.py:91-103``) masks
  ``logits[..., len(tokenizer):] = -inf``. An eos id at or above ``len(tokenizer)`` can never be
  sampled.
* The overlong-prompt filter swallows every chat-template exception and returns
  ``max_prompt_length + 1`` (``rl_dataset.py:238-263``), so a broken template empties the dataset.
* ``math_dapo.compute_score`` is format-agnostic: it takes ``solution_str[-300:]`` and the LAST
  ``Answer:`` match. Reward correctness therefore hinges on the model's output shape, and every
  reward manager decodes with ``skip_special_tokens=True`` (``reward_manager/naive.py:54-56``), which
  deletes special tokens before any parser could split on them.

Run on the machine that has the model:

    source /home/jovyan/ugadiarov/custom-verl/activate.sh
    cd /home/jovyan/ugadiarov/custom-verl && python scripts/preflight_openpangu.py

Exits non-zero if any check FAILs. Checks that cannot run (missing dependency, missing parquet) are
reported SKIP and do not fail the run.
"""

import argparse
import json
import os
import sys
import traceback

MODEL_DEFAULT = "FreedomIntelligence/openPangu-Embedded-7B"
DATA_DEFAULT = "/home/jovyan/datasets/math_datasets/dapo"

# The markers the openPangu notes describe. Verified here rather than assumed: [unused9]/[unused10]
# delimit chat roles, [unused16]/[unused17] open and close the 7B's thinking block.
ROLE_MARKERS = ["[unused9]", "[unused10]"]
THINK_MARKERS = ["[unused16]", "[unused17]"]

_results = []


def record(name, ok, detail=""):
    """ok: True -> PASS, False -> FAIL, None -> SKIP."""
    tag = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    _results.append((name, ok))
    print(f"  [{tag}] {name}")
    for line in str(detail).splitlines():
        print(f"         {line}")
    return ok


def section(title):
    print(f"\n=== {title} ===")


def show(text, limit=1400):
    """Render a string with escapes visible -- whitespace and specials are the whole point here."""
    body = text if len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"
    return repr(body)


# --------------------------------------------------------------------------- 1. tokenizer


def check_tokenizer(model):
    section("1. Tokenizer loads with trust_remote_code")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # verl never passes use_fast for a tokenizer (no call site does), so whether we get the slow
    # class is decided by the checkpoint's auto_map, not by us. Report what actually happened.
    record(
        "tokenizer constructed",
        True,
        f"class={type(tok).__name__}  is_fast={getattr(tok, 'is_fast', 'n/a')}  len={len(tok)}",
    )
    return tok


# --------------------------------------------------------------------------- 2. chat template


def check_chat_template(tok, sample_prompt):
    section("2. Chat template (slow-think must be the default, no suffix)")
    if getattr(tok, "chat_template", None) is None:
        return record(
            "tokenizer.chat_template is set",
            False,
            "None -> apply_chat_template raises, the overlong filter swallows it "
            "(rl_dataset.py:238-263) and silently drops every row.",
        )
    record("tokenizer.chat_template is set", True, f"{len(tok.chat_template)} chars")

    messages = [{"role": "user", "content": sample_prompt}]
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("\n  --- rendered prompt, verbatim ---")
    print("  " + show(rendered))
    print("  --- end ---\n")

    found = [m for m in ROLE_MARKERS if m in rendered]
    record(
        "role markers present",
        len(found) == len(ROLE_MARKERS),
        f"found {found}, expected {ROLE_MARKERS} (informational: a different scheme is fine "
        f"so long as the template renders)",
    )
    # The decision recorded in the plan: train in slow-think mode, which must be the default.
    ok = "/no_think" not in rendered and "/auto_think" not in rendered
    record(
        "no thinking-mode suffix injected by the template",
        ok,
        "slow think is the default -- nothing to pass explicitly"
        if ok
        else "the template injects a mode suffix; the arm would not be training slow-think",
    )
    record(
        "generation prompt is a suffix of the rendered text",
        rendered.startswith(tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)),
        "add_generation_prompt=True only appends -- what verl's agent loop assumes",
    )
    return rendered


# --------------------------------------------------------------------------- 3. eos / logit mask


def check_eos(tok, model):
    section("3. EOS ids and the vLLM logit mask")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    gen_eos = None
    try:
        from transformers import GenerationConfig

        gen_eos = GenerationConfig.from_pretrained(model, trust_remote_code=True).eos_token_id
    except Exception as e:  # noqa: BLE001 - generation_config.json is optional
        gen_eos = f"<unavailable: {type(e).__name__}>"

    n_tok, vocab = len(tok), getattr(cfg, "vocab_size", None)
    record(
        "eos / pad / vocab",
        True,
        f"tokenizer.eos_token={tok.eos_token!r} id={tok.eos_token_id}\n"
        f"config.eos_token_id={getattr(cfg, 'eos_token_id', None)}\n"
        f"generation_config.eos_token_id={gen_eos}\n"
        f"pad_token={tok.pad_token!r} id={tok.pad_token_id}\n"
        f"len(tokenizer)={n_tok}  config.vocab_size={vocab}",
    )

    # THE mask hazard: vllm_async_server.py:402-404 sends vocab_size=len(tokenizer) to
    # monkey_patch_compute_logits, which sets logits[..., len(tokenizer):] = -inf.
    ids = [i for i in [tok.eos_token_id, getattr(cfg, "eos_token_id", None)] if isinstance(i, int)]
    if isinstance(gen_eos, int):
        ids.append(gen_eos)
    elif isinstance(gen_eos, list):
        ids += [i for i in gen_eos if isinstance(i, int)]
    bad = sorted({i for i in ids if i >= n_tok})
    record(
        "every eos id is < len(tokenizer)",
        not bad,
        "otherwise verl's logit mask makes EOS unsamplable and every rollout hits the length cap"
        if bad
        else f"checked ids {sorted(set(ids))} against len(tokenizer)={n_tok}",
    )
    record(
        "tokenizer eos agrees with config eos",
        getattr(cfg, "eos_token_id", None) in (None, tok.eos_token_id),
        "HFModelConfig overwrites hf_config eos from the tokenizer (workers/config/model.py:192-199), "
        "and vLLM reads the checkpoint's own -- a mismatch means the two disagree at runtime",
    )
    return cfg


# --------------------------------------------------------------------------- 4. delimiter survival


def check_delimiters(tok):
    section("4. Do the thinking delimiters survive decoding?")
    survives = {}
    for marker in THINK_MARKERS + ROLE_MARKERS:
        ids = tok.encode(marker, add_special_tokens=False)
        special = any(i in set(tok.all_special_ids) for i in ids)
        kept = tok.decode(ids, skip_special_tokens=True)
        raw = tok.decode(ids, skip_special_tokens=False)
        survives[marker] = marker in kept
        record(
            f"{marker}",
            None,
            f"ids={ids} single_token={len(ids) == 1} is_special={special}\n"
            f"decode(skip_special_tokens=True) ={kept!r}\n"
            f"decode(skip_special_tokens=False)={raw!r}",
        )
    close = THINK_MARKERS[1]
    record(
        f"{close} survives the reward decode",
        survives.get(close, False),
        "reward managers decode with skip_special_tokens=True (reward_manager/naive.py:54-56). "
        "If this is FAIL, a delimiter-aware custom reward function is IMPOSSIBLE without also "
        "changing that decode, and scoring must rely on the [-300:] window instead.",
    )
    return survives


# --------------------------------------------------------------------------- 5. prompt lengths


def check_prompt_lengths(tok, data_dir, max_prompt_length):
    section(f"5. Prompt lengths under the real template (cap={max_prompt_length})")
    try:
        import pandas as pd
    except ImportError:
        return record("prompt length survey", None, "pandas unavailable")

    files = ["dapo-math-17k.parquet", "aime-2024.parquet", "aime-2025.parquet"]
    all_ok = True
    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            record(fname, None, f"not found at {path}")
            continue
        df = pd.read_parquet(path)
        lengths = []
        for msgs in df["prompt"]:
            rendered = tok.apply_chat_template(list(msgs), add_generation_prompt=True, tokenize=True)
            lengths.append(len(rendered))
        lengths.sort()
        over = sum(1 for n in lengths if n > max_prompt_length)
        ok = over == 0
        all_ok &= ok
        p99 = lengths[min(len(lengths) - 1, int(0.99 * len(lengths)))]
        record(
            f"{fname}: {len(lengths)} rows fit the cap",
            ok,
            f"max={lengths[-1]}  p99={p99}  median={lengths[len(lengths) // 2]}  over_cap={over}\n"
            f"data_source={sorted(set(df['data_source']))}",
        )
    return all_ok


# --------------------------------------------------------------------------- 6. reward round-trip


def check_reward_roundtrip(tok):
    section("6. Reward round-trip through the real scorer")
    sys.path.insert(0, os.getcwd())
    try:
        from verl.utils.reward_score import math_dapo
    except Exception as e:  # noqa: BLE001
        return record("import math_dapo", None, f"{type(e).__name__}: {e}")

    open_m, close_m = THINK_MARKERS

    def roundtrip(text):
        """Exactly what the reward manager sees: ids -> decode(skip_special_tokens=True)."""
        return tok.decode(tok.encode(text, add_special_tokens=False), skip_special_tokens=True)

    # (a) the ordinary case: a decoy answer inside the thinking block, real answer on the last line.
    decoy = (
        f"{open_m}\nLet me try. Suppose the total is 7.\nAnswer: 7\n"
        "Hmm, that double-counts. Redo it properly.\n" + ("Filler reasoning step. " * 40) + f"\n{close_m}\n"
        "Combining the cases gives 42.\nAnswer: 42"
    )
    res = math_dapo.compute_score(roundtrip(decoy), "42")
    record(
        "final answer wins over a decoy inside the thinking block",
        res["score"] == 1.0,
        f"score={res['score']} pred={res['pred']!r} (the [-300:] window is what saves this)",
    )

    # (b) unparseable -> -1.0, indistinguishable from wrong except via pred.
    res_boxed = math_dapo.compute_score(roundtrip(f"{open_m}\nwork\n{close_m}\nThe answer is \\boxed{{42}}"), "42")
    record(
        "\\boxed{} WITHOUT an 'Answer:' line scores -1.0",
        res_boxed["score"] == -1.0 and res_boxed["pred"] == "[INVALID]",
        f"score={res_boxed['score']} pred={res_boxed['pred']!r}\n"
        "This is expected, not a bug: this dispatch path requires the Answer: line "
        "(reward_score/math_dapo.py:166,180). It is why the DAPO prompt instructs that format.",
    )

    # (c) the real exposure: a long tail after the answer pushes it out of the 300-char window.
    tail = decoy + "\n" + ("Let me double-check the arithmetic once more. " * 12)
    res_tail = math_dapo.compute_score(roundtrip(tail), "42")
    record(
        "long trailing text after the answer",
        None,
        f"score={res_tail['score']} pred={res_tail['pred']!r}\n"
        "-1.0 here means a verbose epilogue silently costs reward -- watch the [INVALID] rate "
        "and pred values in the first validation.",
    )
    return True


# --------------------------------------------------------------------------- 7. architecture


def check_architecture(model, cfg):
    section("7. Architecture support (FSDP2 actor + vLLM rollout)")
    arch = (getattr(cfg, "architectures", None) or ["<unknown>"])[0]
    record("architecture", None, f"{arch}  model_type={getattr(cfg, 'model_type', None)}")

    # verl's megatron registry rejects unknown architectures outright; the FSDP2 path does not,
    # which is why this arm must be FSDP2.
    try:
        from verl.models.mcore.registry import SupportedModel

        known = {m.value for m in SupportedModel}
        record(
            "mcore registry status",
            None,
            f"{arch} in mcore registry: {arch in known}\n"
            + (
                "The stock PanguEmbedded arch is absent, which is why this arm must be FSDP2."
                if arch not in known
                else "Re-aliased to Llama, so the registry accepts it -- but megatron still CANNOT "
                "represent this model: bias sits on q/k/v and o_proj but not the MLP, and "
                "megatron-core governs o_proj and both MLP linears with the single add_bias_linear "
                "flag (attention.py:355, mlp.py:128/147). verl hardcodes it False "
                "(config_converter.py:178) and mbridge maps linear_proj weight-only, so o_proj.bias "
                "would be dropped. FSDP2 remains the only viable backend."
            ),
        )
    except Exception as e:  # noqa: BLE001
        record("mcore registry lookup", None, f"{type(e).__name__}: {e}")

    # vLLM must have its own implementation: trust_remote_code supplies the config, not the
    # modeling code used for inference (vllm_rollout/utils.py:287 -> model.load_weights).
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        supported = ModelRegistry.get_supported_archs()
        ok = arch in supported
        record(
            "vLLM has a registered implementation",
            ok,
            f"{arch} registered: {ok}"
            + ("" if ok else "\nFALLBACK: add +actor_rollout_ref.rollout.engine_kwargs.vllm.model_impl=transformers"),
        )
    except Exception as e:  # noqa: BLE001
        record("vLLM registry lookup", None, f"{type(e).__name__}: {e}")

    # FSDP2 auto-wrap needs _no_split_modules; verl asserts on it (utils/fsdp_utils.py:559-568).
    try:
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        with init_empty_weights():
            m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        nsm = getattr(m, "_no_split_modules", None)
        record(
            "_no_split_modules is set (FSDP2 auto-wrap)",
            bool(nsm),
            f"_no_split_modules={nsm}  gradient_checkpointing={getattr(m, 'supports_gradient_checkpointing', None)}",
        )
        names = {mod.__class__.__name__ for mod in m.modules()}
        record(
            "the named decoder-layer class actually exists",
            bool(nsm) and all(n in names for n in nsm),
            f"present: {[n for n in (nsm or []) if n in names]}",
        )
        # verl forces attn_implementation=flash_attention_2 by default (workers/config/model.py:185-188).
        record(
            "declares flash-attention support",
            bool(getattr(m, "_supports_flash_attn", None) or getattr(m, "_supports_flash_attn_2", None)),
            "otherwise set +actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        )
        # apply_monkey_patch silently skips unknown model_type and then patches
        # _flash_attention_forward only if that symbol is reachable (monkey_patch.py:529-538).
        # apply_monkey_patch (monkey_patch.py:529-538) patches the model module's own
        # _flash_attention_forward if it has one, else the shared
        # transformers.integrations.flash_attention. Either target is fine; having NEITHER is what
        # would make use_remove_padding=True a silent no-op.
        mod = sys.modules.get(type(m).__module__)
        local_target = hasattr(mod, "_flash_attention_forward")
        try:
            import transformers.integrations.flash_attention as _fa

            global_target = hasattr(_fa, "_flash_attention_forward")
        except Exception:  # noqa: BLE001
            global_target = False
        record(
            "remove-padding patch has a reachable target",
            local_target or global_target,
            f"module={type(m).__module__}\n"
            f"module-local _flash_attention_forward: {local_target}\n"
            f"transformers.integrations.flash_attention._flash_attention_forward: {global_target}\n"
            "Native Llama routes attention through ALL_ATTENTION_FUNCTIONS, so the global target is "
            "the correct one and the module-local miss is expected.",
        )
    except Exception as e:  # noqa: BLE001
        record("model class introspection", None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")


# --------------------------------------------------------------------------- 8/9. Llama re-alias


def _shim_loss_kwargs():
    """Make the openPangu remote code importable on transformers >= 4.54, as a TEST FIXTURE only.

    ``modeling_openpangu_dense.py:59`` imports ``LossKwargs``, which 4.54 folded into
    ``TransformersKwargs``; it is used exactly once, at ``:493``, as a typing marker. We alias it so
    the parity check below can instantiate the ORIGINAL implementation and compare it against native
    Llama. The training path does not rely on this -- that is the whole point of the re-alias.
    """
    import transformers.utils as U

    if not hasattr(U, "LossKwargs"):
        from transformers.utils import TransformersKwargs

        U.LossKwargs = TransformersKwargs
    return hasattr(U, "LossKwargs")


def check_llama_key_map(model_path):
    section("8. Llama loads the checkpoint with no missing or unexpected keys")
    import glob

    cfg_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
    if cfg_path is None or not os.path.exists(cfg_path):
        return record("key map", None, f"{model_path} is not a local re-aliased directory")
    cfg = json.load(open(cfg_path))
    if cfg.get("model_type") != "llama":
        return record(
            "key map",
            None,
            f"model_type={cfg.get('model_type')!r}; run scripts/realias_openpangu_to_llama.py first",
        )

    # Compare the checkpoint's own tensor names against the names a Llama of this config expects.
    # Reading the safetensors index (or headers) avoids materialising 16 GB.
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index):
        ckpt_keys = set(json.load(open(index))["weight_map"])
    else:
        from safetensors import safe_open

        ckpt_keys = set()
        shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not shards:
            return record("key map", None, "no safetensors found (weights not downloaded yet)")
        for shard in shards:
            with safe_open(shard, framework="pt") as f:
                ckpt_keys |= set(f.keys())

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    llama_cfg = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        llama = AutoModelForCausalLM.from_config(llama_cfg)
    expected = set(llama.state_dict())
    if llama_cfg.tie_word_embeddings:
        expected.discard("lm_head.weight")

    missing, unexpected = sorted(expected - ckpt_keys), sorted(ckpt_keys - expected)
    record(
        "every Llama parameter is present in the checkpoint",
        not missing,
        f"{len(expected)} expected, {len(ckpt_keys)} in checkpoint\nmissing: {missing[:8]}",
    )
    # rotary_emb.inv_freq is a NON-persistent buffer in both implementations -- recomputed from
    # rope_theta at init. Some exports save it anyway; transformers ignores it. Anything else being
    # unexpected would mean a genuine name mismatch.
    benign = [k for k in unexpected if k.endswith("rotary_emb.inv_freq")]
    real = [k for k in unexpected if k not in benign]
    record(
        "the checkpoint carries no parameter Llama would silently ignore",
        not real,
        f"unexpected: {len(unexpected)} total, {len(benign)} benign rotary_emb.inv_freq buffers "
        f"(derived from rope_theta, recomputed at init)\nreal mismatches: {real[:8]}",
    )
    # Names matching is not enough: a shape mismatch would pass the key check and only fail when the
    # weights are actually loaded, i.e. on the GPU box after the job starts.
    shapes = {}
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        from safetensors import safe_open

        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                shapes[k] = tuple(f.get_slice(k).get_shape())
    expected_shapes = {k: tuple(v.shape) for k, v in llama.state_dict().items()}
    bad = [
        f"{k}: ckpt{shapes[k]} vs llama{expected_shapes[k]}"
        for k in sorted(expected_shapes)
        if k in shapes and shapes[k] != expected_shapes[k]
    ]
    record(
        "every shared parameter has the shape Llama expects",
        not bad,
        f"compared {len(set(shapes) & set(expected_shapes))} tensors\nmismatches: {bad[:6]}"
        if shapes
        else "no safetensors shards found to inspect",
    )
    record(
        "attention bias tensors present on all four projections",
        all(
            any(k.endswith(f"self_attn.{p}.bias") for k in ckpt_keys) for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
        "Pangu's bias=true maps to Llama's attention_bias, which covers o_proj too",
    )


def check_numerical_parity(hub_model):
    section("9. Numerical parity: original openPangu code vs native Llama")
    if not _shim_loss_kwargs():
        return record("parity", None, "could not shim LossKwargs; original code not importable")
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    except Exception as e:  # noqa: BLE001
        return record("parity", None, f"{type(e).__name__}: {e}")

    # A miniature model: this compares the two implementations' MATHEMATICS (RMSNorm, rotary,
    # attention scaling, residual order), which is independent of size. Seconds on CPU, no weights.
    try:
        pcfg = AutoConfig.from_pretrained(hub_model, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        return record("parity", None, f"cannot load PanguEmbeddedConfig: {type(e).__name__}: {e}")

    small = dict(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    for k, v in small.items():
        setattr(pcfg, k, v)
    # The hub config carries torch_dtype=bfloat16, and from_config honours it -- so without this the
    # Pangu side is built in bf16 and the Llama side in fp32, and the comparison measures bf16
    # rounding (~5e-3 relative) instead of the implementations.
    pcfg.torch_dtype = torch.float32
    # Keep every value that differs from a LlamaConfig default -- those are what parity must cover.
    pcfg.rms_norm_eps = 1e-5
    pcfg.rope_theta = 16000000.0
    pcfg.bias = True
    pcfg.tie_word_embeddings = False
    pcfg.attention_dropout = 0.0
    pcfg._attn_implementation = "eager"

    lcfg = LlamaConfig(
        **small,
        rms_norm_eps=pcfg.rms_norm_eps,
        rope_theta=pcfg.rope_theta,
        attention_bias=True,
        mlp_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=getattr(pcfg, "pad_token_id", 0),
        attn_implementation="eager",
    )

    torch.manual_seed(0)
    pangu = AutoModelForCausalLM.from_config(pcfg, trust_remote_code=True).to(torch.float32).eval()
    llama = LlamaForCausalLM(lcfg).to(torch.float32).eval()
    record(
        "both implementations built at the same dtype",
        next(pangu.parameters()).dtype == next(llama.parameters()).dtype == torch.float32,
        f"pangu={next(pangu.parameters()).dtype} llama={next(llama.parameters()).dtype}",
    )

    # Names are 1:1, so a strict load is itself an assertion about the parameter map.
    missing, unexpected = llama.load_state_dict(pangu.state_dict(), strict=False)
    record(
        "miniature Llama accepts the Pangu state dict verbatim",
        not missing and not unexpected,
        f"missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}",
    )

    ids = torch.randint(0, small["vocab_size"], (2, 17))
    with torch.no_grad():
        a = pangu(input_ids=ids).logits
        b = llama(input_ids=ids).logits
    delta = (a - b).abs().max().item()
    scale = a.abs().max().item()
    record(
        "logits match between the two implementations",
        delta <= 1e-5 * max(1.0, scale),
        f"max|delta|={delta:.3e}  max|logit|={scale:.3e}\n"
        "This is the gate for the re-alias: it compares RMSNorm, rotary embedding, attention "
        "scaling and residual order directly, not by reading the source.",
    )


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", MODEL_DEFAULT))
    ap.add_argument(
        "--model-path",
        default=None,
        help="local re-aliased checkpoint dir; enables the Llama key-map and parity gates",
    )
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", DATA_DEFAULT))
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    args = ap.parse_args()

    print(f"model    : {args.model}")
    print(f"model dir: {args.model_path or '<none - key-map gate skipped>'}")
    print(f"data dir : {args.data_dir}")

    # Every check except the parity gate runs against the artifact we will actually train on; the
    # parity gate deliberately uses the hub id, because it needs the ORIGINAL Pangu implementation.
    target = args.model_path or args.model
    tok = check_tokenizer(target)

    sample = (
        "Solve the following math problem step by step. The last line of your response should be of "
        "the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        'What is $1+1$?\n\nRemember to put your answer on its own line after "Answer:".'
    )
    check_chat_template(tok, sample)
    cfg = check_eos(tok, target)
    check_delimiters(tok)
    check_prompt_lengths(tok, args.data_dir, args.max_prompt_length)
    check_reward_roundtrip(tok)
    check_architecture(target, cfg)
    if args.model_path:
        check_llama_key_map(args.model_path)
    check_numerical_parity(args.model)

    section("Summary")
    failed = [n for n, ok in _results if ok is False]
    skipped = [n for n, ok in _results if ok is None]
    passed = [n for n, ok in _results if ok is True]
    print(f"  {len(passed)} passed, {len(failed)} failed, {len(skipped)} informational/skipped")
    for n in failed:
        print(f"  FAILED: {n}")
    print(json.dumps({"passed": len(passed), "failed": failed, "skipped": len(skipped)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
