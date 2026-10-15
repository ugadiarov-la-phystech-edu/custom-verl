# Copyright 2026 custom-verl authors
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
"""CPU tests for verl/utils/vllm/qwen2_k_bias_center.py (fix A: Qwen2 key-bias centring for FP8 KV caches)."""

import inspect
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm.model_executor.models import qwen2  # noqa: E402

import verl.utils.vllm.qwen2_k_bias_center as kbc  # noqa: E402

HEAD_DIM = 128
ROPE_THETA = 1e6
# Lowest-frequency RoPE pairs, where Qwen2.5-7B keeps its large k/q biases (pairs 58-63 -> dims 58-63 and 122-127).
LOW_FREQ_DIMS = list(range(58, 64)) + list(range(122, 128))


@pytest.fixture(autouse=True)
def _pristine_qwen2(monkeypatch):
    """Every test starts and ends with vLLM's original Qwen2Attention.forward and no VERL_QWEN2_K_BIAS_CENTER."""
    kbc.remove_qwen2_k_bias_center()
    original = qwen2.Qwen2Attention.forward
    monkeypatch.delenv(kbc.K_BIAS_CENTER_ENV, raising=False)
    yield original
    kbc.remove_qwen2_k_bias_center()
    assert qwen2.Qwen2Attention.forward is original


# ----------------------------------------------------------------------------- env parsing / install


@pytest.mark.parametrize("value, expected", [(None, False), ("", False), ("0", False), (" 0 ", False), ("1", True)])
def test_env_parsing(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(kbc.K_BIAS_CENTER_ENV, value)
    assert kbc.k_bias_center_enabled() is expected


@pytest.mark.parametrize("value", ["true", "2", "yes", "on"])
def test_env_parsing_rejects_typos(monkeypatch, value):
    monkeypatch.setenv(kbc.K_BIAS_CENTER_ENV, value)
    with pytest.raises(ValueError, match="VERL_QWEN2_K_BIAS_CENTER"):
        kbc.k_bias_center_enabled()


def test_apply_is_idempotent_and_remove_restores(_pristine_qwen2):
    original = _pristine_qwen2
    kbc.apply_qwen2_k_bias_center()
    assert qwen2.Qwen2Attention.forward is kbc._k_bias_centered_forward
    kbc.apply_qwen2_k_bias_center()  # second call must not wrap the patched forward as "original"
    assert kbc._ORIGINAL_FORWARD is original
    kbc.remove_qwen2_k_bias_center()
    assert qwen2.Qwen2Attention.forward is original
    kbc.remove_qwen2_k_bias_center()  # no-op when not installed
    assert qwen2.Qwen2Attention.forward is original


def test_maybe_apply_follows_env(monkeypatch, _pristine_qwen2):
    assert kbc.maybe_apply_qwen2_k_bias_center() is False
    assert qwen2.Qwen2Attention.forward is _pristine_qwen2
    monkeypatch.setenv(kbc.K_BIAS_CENTER_ENV, "0")
    assert kbc.maybe_apply_qwen2_k_bias_center() is False
    assert qwen2.Qwen2Attention.forward is _pristine_qwen2
    monkeypatch.setenv(kbc.K_BIAS_CENTER_ENV, "1")
    assert kbc.maybe_apply_qwen2_k_bias_center() is True
    assert qwen2.Qwen2Attention.forward is kbc._k_bias_centered_forward


# ----------------------------------------------------------------------------- the patched forward on a fake module


class _QKV:
    """Fused qkv projection like QKVParallelLinear: returns (x @ W^T + b, None) and exposes a live ``bias``."""

    def __init__(self, hidden, out, bias=True, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(out, hidden, generator=g)
        self.bias = torch.randn(out, generator=g) * 10 if bias else None

    def __call__(self, x):
        y = x @ self.weight.T
        return (y + self.bias if self.bias is not None else y), None


def _fake_attention(q_size, kv_size, hidden=16, bias=True, qk_norm=False):
    seen = {}

    def rotary_emb(positions, q, k):
        # any position-dependent transform will do; record what RoPE produced
        q, k = q + positions[:, None], k * (1.0 + positions[:, None])
        seen["rope_q"], seen["rope_k"] = q.clone(), k.clone()
        return q, k

    def attn(q, k, v):
        seen["q"], seen["k"], seen["v"] = q, k, v
        return torch.cat([q, k, v], dim=-1)

    def o_proj(x):
        return x * 2.0, None

    mod = SimpleNamespace(
        qkv_proj=_QKV(hidden, q_size + 2 * kv_size, bias=bias),
        q_size=q_size,
        kv_size=kv_size,
        qk_norm=qk_norm,
        rotary_emb=rotary_emb,
        attn=attn,
        o_proj=o_proj,
    )
    return mod, seen


@pytest.mark.parametrize("q_size, kv_size", [(8, 4), (12, 4), (7, 3), (28 * 128, 4 * 128)])
def test_patched_forward_subtracts_k_bias_after_rope(q_size, kv_size):
    """k reaching attention == RoPE(k) - k-slice of the fused bias; q and v untouched; TP-local / GQA layouts."""
    kbc.apply_qwen2_k_bias_center()
    mod, seen = _fake_attention(q_size, kv_size, hidden=32)
    positions = torch.arange(5)
    hidden = torch.randn(5, 32)
    kbc._k_bias_centered_forward(mod, positions, hidden)
    k_bias = mod.qkv_proj.bias[q_size : q_size + kv_size]
    torch.testing.assert_close(seen["k"], seen["rope_k"] - k_bias)
    torch.testing.assert_close(seen["q"], seen["rope_q"])
    qkv, _ = mod.qkv_proj(hidden)
    torch.testing.assert_close(seen["v"], qkv[:, q_size + kv_size :])


def test_patched_forward_reads_bias_live():
    """A refit updates the bias in place; the next forward must use the new value (never an init-time copy)."""
    kbc.apply_qwen2_k_bias_center()
    mod, seen = _fake_attention(8, 4)
    positions, hidden = torch.arange(3), torch.randn(3, 16)
    kbc._k_bias_centered_forward(mod, positions, hidden)
    mod.qkv_proj.bias.copy_(torch.randn_like(mod.qkv_proj.bias) * 100)  # in-place, like load_weights
    kbc._k_bias_centered_forward(mod, positions, hidden)
    torch.testing.assert_close(seen["k"], seen["rope_k"] - mod.qkv_proj.bias[8:12])


def test_patched_forward_output_path():
    kbc.apply_qwen2_k_bias_center()
    mod, seen = _fake_attention(8, 4)
    out = kbc._k_bias_centered_forward(mod, torch.arange(3), torch.randn(3, 16))
    torch.testing.assert_close(out, 2.0 * torch.cat([seen["q"], seen["k"], seen["v"]], dim=-1))


@pytest.mark.parametrize("qk_norm, bias", [(True, True), (False, False)])
def test_patched_forward_delegates_without_constant_key_offset(monkeypatch, qk_norm, bias):
    """QK-norm variants and bias-free projections go through vLLM's original forward unchanged."""
    calls = []
    kbc.apply_qwen2_k_bias_center()
    mod, _ = _fake_attention(8, 4, bias=bias, qk_norm=qk_norm)
    positions, hidden = torch.arange(2), torch.randn(2, 16)
    with monkeypatch.context() as m:
        m.setattr(kbc, "_ORIGINAL_FORWARD", lambda self, p, h: calls.append((self, p, h)) or "orig")
        assert kbc._k_bias_centered_forward(mod, positions, hidden) == "orig"
    assert calls == [(mod, positions, hidden)]


def test_patched_forward_delegates_to_real_original_for_qk_norm():
    """With the real original installed as fallback, a QK-norm module takes vLLM's QK-norm branch."""
    kbc.apply_qwen2_k_bias_center()
    q_size, kv_size, nh, nkv, hd = 8, 4, 2, 1, 4
    mod, seen = _fake_attention(q_size, kv_size, qk_norm=True)
    mod.num_heads, mod.num_kv_heads, mod.head_dim = nh, nkv, hd
    mod.q_norm = lambda x: x * 0 + 1.0
    mod.k_norm = lambda x: x * 0 + 2.0
    kbc._k_bias_centered_forward(mod, torch.arange(3), torch.randn(3, 16))
    # QK-norm ran (constant 2.0 keys before RoPE), and no bias was subtracted afterwards
    torch.testing.assert_close(seen["k"], seen["rope_k"])
    torch.testing.assert_close(seen["rope_k"], 2.0 * (1.0 + torch.arange(3.0))[:, None].expand(3, kv_size))


# ----------------------------------------------------------------------------- source-drift guard

# Non-comment, whitespace-normalised body of vLLM 0.24's Qwen2Attention.forward. The patch re-implements the
# non-QK-norm path of exactly this code; if vLLM changes it, re-check _k_bias_centered_forward and update this snapshot.
_EXPECTED_ORIGINAL_FORWARD = [
    "def forward(",
    "self,",
    "positions: torch.Tensor,",
    "hidden_states: torch.Tensor,",
    ") -> torch.Tensor:",
    "qkv, _ = self.qkv_proj(hidden_states)",
    "q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)",
    "if self.qk_norm:",
    "total_tokens = q.shape[0]",
    "q = q.view(total_tokens, self.num_heads, self.head_dim)",
    "k = k.view(total_tokens, self.num_kv_heads, self.head_dim)",
    "q = self.q_norm(q)",
    "k = self.k_norm(k)",
    "q = q.view(total_tokens, self.q_size)",
    "k = k.view(total_tokens, self.kv_size)",
    "q, k = self.rotary_emb(positions, q, k)",
    "attn_output = self.attn(q, k, v)",
    "output, _ = self.o_proj(attn_output)",
    "return output",
]


def _normalised_source(fn):
    lines = []
    for line in inspect.getsource(fn).splitlines():
        line = re.sub(r"#.*$", "", line).strip()
        if line:
            lines.append(line)
    return lines


def test_vllm_qwen2_forward_matches_the_reimplemented_code(_pristine_qwen2):
    assert _normalised_source(_pristine_qwen2) == _EXPECTED_ORIGINAL_FORWARD


def test_qwen2_attention_constructs_a_biased_fused_qkv():
    """The patch's slice assumes QKVParallelLinear(bias=True) with [q, k, v] output order."""
    src = inspect.getsource(qwen2.Qwen2Attention.__init__)
    assert re.search(r"QKVParallelLinear\(.*?bias=True", src, re.S)


# ----------------------------------------------------------------------------- numerics with a reference attention


def _rope(x, positions, num_heads, theta=ROPE_THETA):
    """Neox-style RoPE (Qwen2), x: (T, num_heads * HEAD_DIM)."""
    t = x.shape[0]
    x = x.view(t, num_heads, HEAD_DIM)
    half = HEAD_DIM // 2
    inv_freq = theta ** (-torch.arange(half, dtype=torch.float64) / half)
    ang = positions.double()[:, None] * inv_freq[None, :]
    cos, sin = ang.cos().to(x.dtype)[:, None, :], ang.sin().to(x.dtype)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).reshape(t, num_heads * HEAD_DIM)


def _causal_gqa_attention(q, k, v, num_heads, num_kv_heads):
    t = q.shape[0]
    q = q.view(t, num_heads, HEAD_DIM).transpose(0, 1)
    k = k.view(t, num_kv_heads, HEAD_DIM).transpose(0, 1).repeat_interleave(num_heads // num_kv_heads, dim=0)
    v = v.view(t, num_kv_heads, HEAD_DIM).transpose(0, 1).repeat_interleave(num_heads // num_kv_heads, dim=0)
    scores = q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM)
    scores = scores.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
    return (scores.softmax(-1) @ v).transpose(0, 1).reshape(t, num_heads * HEAD_DIM)


def _fp8(x):
    """vLLM's fp8_e4m3 KV cache with scale 1.0: saturating cast and back."""
    return x.float().clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype)


def _qwen25_like(seed, t=64, num_heads=4, num_kv_heads=2, k_offset=300.0, q_offset=5.0):
    """Activations shaped like Qwen2.5's bad layers: a large k bias and a q bias in the low-frequency RoPE dims.

    q_offset is chosen so the bias-free logit spread per row stays O(3), as measured in Qwen2.5-7B (2.5-3.2); the
    k offset (300) is in the 256-448 e4m3 band where the grid step is 32.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(t, num_heads * HEAD_DIM, generator=g, dtype=torch.float64)
    k = torch.randn(t, num_kv_heads * HEAD_DIM, generator=g, dtype=torch.float64) * 2
    v = torch.randn(t, num_kv_heads * HEAD_DIM, generator=g, dtype=torch.float64)
    q_bias = torch.zeros(num_heads * HEAD_DIM, dtype=torch.float64)
    k_bias = torch.zeros(num_kv_heads * HEAD_DIM, dtype=torch.float64)
    for h in range(num_heads):
        q_bias[h * HEAD_DIM + torch.tensor(LOW_FREQ_DIMS)] = q_offset
    for h in range(num_kv_heads):
        k_bias[h * HEAD_DIM + torch.tensor(LOW_FREQ_DIMS)] = k_offset
    return q + q_bias, k + k_bias, v, k_bias, num_heads, num_kv_heads


@pytest.mark.parametrize("pos_offset", [0, 30_000])
def test_centring_is_exact_with_rope_and_gqa(pos_offset):
    """Softmax shift-invariance: subtracting b_k after RoPE leaves causal GQA attention unchanged, at any position."""
    q, k, v, k_bias, nh, nkv = _qwen25_like(seed=1)
    positions = torch.arange(q.shape[0]) + pos_offset
    qr, kr = _rope(q, positions, nh), _rope(k, positions, nkv)
    ref = _causal_gqa_attention(qr, kr, v, nh, nkv)
    out = _causal_gqa_attention(qr, kr - k_bias, v, nh, nkv)
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)


def test_subtracting_the_bias_before_rope_is_not_exact():
    """Counter-check: removing b_k before RoPE (e.g. editing k_proj.bias in the weights) changes the output."""
    q, k, v, k_bias, nh, nkv = _qwen25_like(seed=2, k_offset=5.0, q_offset=5.0)
    k_bias = k_bias.clone()
    k_bias[: HEAD_DIM // 2] = 5.0  # some bias in high-frequency pairs too, as in Qwen2.5's middle layers
    positions = torch.arange(q.shape[0])
    ref = _causal_gqa_attention(_rope(q, positions, nh), _rope(k, positions, nkv), v, nh, nkv)
    pre = _causal_gqa_attention(_rope(q, positions, nh), _rope(k - k_bias, positions, nkv), v, nh, nkv)
    assert (pre - ref).abs().max() > 1e-3


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_centring_rescues_fp8_kv_cache(seed):
    """With a Qwen2.5-sized key offset, FP8 KV wrecks attention; caching RoPE(k) - b_k brings it back."""
    q, k, v, k_bias, nh, nkv = _qwen25_like(seed=seed)
    positions = torch.arange(q.shape[0])
    qr, kr = _rope(q, positions, nh), _rope(k, positions, nkv)
    ref = _causal_gqa_attention(qr, kr, v, nh, nkv)
    plain = _causal_gqa_attention(qr, _fp8(kr), _fp8(v), nh, nkv)
    centred = _causal_gqa_attention(qr, _fp8(kr - k_bias), _fp8(v), nh, nkv)
    err_plain = (plain - ref).norm() / ref.norm()
    err_centred = (centred - ref).norm() / ref.norm()
    assert err_plain > 0.2, f"baseline should be badly broken, rel err {err_plain:.3f}"
    assert err_centred < err_plain / 10, f"centred {err_centred:.4f} vs plain {err_plain:.4f}"


def test_centring_is_harmless_without_offset():
    """Zero bias (e.g. after a refit that zeroes it): centring is a no-op, FP8 results identical."""
    q, k, v, k_bias, nh, nkv = _qwen25_like(seed=6, k_offset=0.0, q_offset=0.0)
    positions = torch.arange(q.shape[0])
    qr, kr = _rope(q, positions, nh), _rope(k, positions, nkv)
    plain = _causal_gqa_attention(qr, _fp8(kr), _fp8(v), nh, nkv)
    centred = _causal_gqa_attention(qr, _fp8(kr - k_bias), _fp8(v), nh, nkv)
    torch.testing.assert_close(centred, plain, rtol=0, atol=0)


def test_patched_forward_with_real_vllm_rope_matches_unpatched():
    """End to end through the patched forward with vLLM's own RoPE op and a reference attention."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    try:
        with set_current_vllm_config(VllmConfig()):
            rope_parameters = {"rope_type": "default", "rope_theta": ROPE_THETA}
            rope = get_rope(HEAD_DIM, max_position=40_000, rope_parameters=rope_parameters)
    except Exception as exc:  # pragma: no cover - depends on the CPU vLLM build
        pytest.skip(f"vLLM RoPE cannot be built on this host: {type(exc).__name__}: {exc}")

    nh, nkv, hidden, t = 4, 2, 64, 24
    q_size, kv_size = nh * HEAD_DIM, nkv * HEAD_DIM
    qkv_proj = _QKV(hidden, q_size + 2 * kv_size, seed=7)
    qkv_proj.weight = qkv_proj.weight.to(torch.float32) * 0.1
    bias = torch.zeros(q_size + 2 * kv_size)
    for h in range(nh):
        bias[h * HEAD_DIM + torch.tensor(LOW_FREQ_DIMS)] = 50.0
    for h in range(nkv):
        bias[q_size + h * HEAD_DIM + torch.tensor(LOW_FREQ_DIMS)] = 300.0
    qkv_proj.bias = bias

    def rotary_emb(positions, q, k):
        return rope.forward_native(positions, q.clone(), k.clone())

    mod = SimpleNamespace(
        qkv_proj=qkv_proj,
        q_size=q_size,
        kv_size=kv_size,
        qk_norm=False,
        rotary_emb=rotary_emb,
        attn=lambda q, k, v: _causal_gqa_attention(q.double(), k.double(), v.double(), nh, nkv),
        o_proj=lambda x: (x, None),
    )
    positions = torch.arange(t) + 12_345
    hidden_states = torch.randn(t, hidden, generator=torch.Generator().manual_seed(8))

    unpatched = qwen2.Qwen2Attention.forward(mod, positions, hidden_states)
    kbc.apply_qwen2_k_bias_center()
    patched = qwen2.Qwen2Attention.forward(mod, positions, hidden_states)
    torch.testing.assert_close(patched, unpatched, rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------------------- wiring: worker hook, Ray env, script


@pytest.fixture
def worker_utils(monkeypatch):
    utils = pytest.importorskip("verl.workers.rollout.vllm_rollout.utils")
    monkeypatch.setattr(utils, "set_death_signal", lambda: None)
    monkeypatch.setattr(utils.VLLMHijack, "hijack", staticmethod(lambda: None))
    monkeypatch.setattr(utils, "apply_vllm_quant_patches", lambda: None)
    monkeypatch.setattr(utils, "is_refit_quant_model", lambda cfg: False)
    for var in ("VERL_FULL_DETERMINISM", "VERL_VLLM_FP8_QUANT_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    calls = []
    monkeypatch.setattr(kbc, "apply_qwen2_k_bias_center", lambda: calls.append(1))
    return utils, calls


@pytest.mark.parametrize("env, expected", [(None, []), ("0", []), ("1", [1])])
def test_worker_extension_applies_patch_iff_env(monkeypatch, worker_utils, env, expected):
    utils, calls = worker_utils
    if env is not None:
        monkeypatch.setenv(kbc.K_BIAS_CENTER_ENV, env)
    instance = utils.vLLMColocateWorkerExtension.__new__(utils.vLLMColocateWorkerExtension, vllm_config=None)
    assert isinstance(instance, utils.vLLMColocateWorkerExtension)
    assert calls == expected


def test_worker_extension_rejects_bad_env(monkeypatch, worker_utils):
    utils, _ = worker_utils
    monkeypatch.setenv(kbc.K_BIAS_CENTER_ENV, "true")
    with pytest.raises(ValueError):
        utils.vLLMColocateWorkerExtension.__new__(utils.vLLMColocateWorkerExtension, vllm_config=None)


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_ray_runtime_env_forwards_flag_only_when_set(monkeypatch, value):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    if value is None:
        monkeypatch.delenv(kbc.K_BIAS_CENTER_ENV, raising=False)
    else:
        monkeypatch.setenv(kbc.K_BIAS_CENTER_ENV, value)
    env_vars = get_ppo_ray_runtime_env()["env_vars"]
    if value is None:
        assert kbc.K_BIAS_CENTER_ENV not in env_vars
    else:
        assert env_vars[kbc.K_BIAS_CENTER_ENV] == value


@pytest.fixture
def gos():
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import scripts.group_outcome_stats as module

    return module


def test_script_flag_defaults_off(gos):
    assert gos.parse_args(["--num-prompts", "4"]).qwen2_k_bias_center is False
    assert gos.parse_args(["--num-prompts", "4", "--qwen2-k-bias-center"]).qwen2_k_bias_center is True


@pytest.mark.parametrize("flag", [False, True])
def test_script_engine_env_is_authoritative(monkeypatch, gos, flag):
    env = {kbc.K_BIAS_CENTER_ENV: "0" if flag else "1"}  # a stray shell value must be overridden
    monkeypatch.setattr(os, "environ", env)
    argv = ["--num-prompts", "4", "--precision", "fp8+fp8kv"] + (["--qwen2-k-bias-center"] if flag else [])
    gos.configure_engine_env(gos.parse_args(argv))
    assert env[kbc.K_BIAS_CENTER_ENV] == ("1" if flag else "0")
    assert (env.get("VLLM_DISABLE_COMPILE_CACHE") == "1") is flag
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] == "0"
