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
"""CPU tests for the online W8A8-INT8 rollout (``verl/utils/vllm/vllm_int8_utils.py`` and its hooks in
``vllm_quant_utils.py`` / ``vllm_async_server.py``).

The refit tests build real vLLM ``QKVParallelLinear`` / ``MergedColumnParallelLinear`` / ``RowParallelLinear``
layers with the compressed-tensors W8A8-INT8 config (world size 1, gloo), so the real weight loaders,
``CompressedTensorsW8A8Int8`` and the CUTLASS kernel's post-load transform run; only the GEMM itself is not
exercised (it needs a GPU).
"""

import os
import socket
from types import SimpleNamespace

import pytest
import torch

vllm = pytest.importorskip("vllm")

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from vllm.model_executor.layers.linear import (  # noqa: E402
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E402
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import CompressedTensorsW8A8Int8  # noqa: E402

from verl.utils.vllm import vllm_int8_utils as i8  # noqa: E402
from verl.utils.vllm import vllm_quant_utils as qu  # noqa: E402

HIDDEN = 64
HEAD = 16
HEADS = 4
KV_HEADS = 2
INTER = 96


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def vllm_ctx():
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield


@pytest.fixture
def int8_patches():
    patchers = i8.build_int8_method_patchers()
    assert patchers, "vLLM must provide CompressedTensorsW8A8Int8"
    for p in patchers:
        p.start()
    yield
    for p in patchers:
        p.stop()


def _quant_config(num_layers=1):
    return CompressedTensorsConfig.from_config(
        i8.build_int8_w8a8_quant_config(SimpleNamespace(num_hidden_layers=num_layers))
    )


def _rand(*shape, seed=0, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


def _dequant(layer) -> torch.Tensor:
    """[N, K] float weights a layer computes with (live layout: weight is the [K, N] transpose view)."""
    return layer.weight.data.t().float() * layer.weight_scale.data.float()


def _rtn(w: torch.Tensor) -> torch.Tensor:
    q, s = i8.quantize_int8_per_channel(w)
    return q.float() * s


# --------------------------------------------------------------------------- quantizer


class TestQuantizer:
    def test_error_is_within_half_a_step(self):
        w = _rand(48, 64, seed=1).float()
        q, s = i8.quantize_int8_per_channel(w)
        assert q.dtype == torch.int8 and q.shape == w.shape
        assert s.dtype == torch.float32 and s.shape == (48, 1)
        assert torch.all((q.float() * s - w).abs() <= s / 2 + 1e-7)

    def test_row_max_maps_to_the_grid_end_and_codes_are_symmetric(self):
        w = _rand(16, 32, seed=2).float()
        q, _ = i8.quantize_int8_per_channel(w)
        assert int(q.abs().amax(dim=1).min()) == 127
        assert int(q.min()) >= -127  # -128 is never emitted (symmetric grid)
        q_neg, _ = i8.quantize_int8_per_channel(-w)
        assert torch.equal(q_neg, -q)

    def test_scale_is_row_absmax_over_127(self):
        w = torch.tensor([[0.5, -1.27, 0.0], [2.54, 0.1, -0.2]])
        _, s = i8.quantize_int8_per_channel(w)
        torch.testing.assert_close(s, torch.tensor([[0.01], [0.02]]))

    def test_zero_row_gets_unit_scale_and_zero_codes(self):
        w = torch.zeros(3, 8)
        w[1] = torch.linspace(-1, 1, 8)
        q, s = i8.quantize_int8_per_channel(w)
        assert s[0].item() == 1.0 and s[2].item() == 1.0
        assert torch.all(q[0] == 0) and torch.all(q[2] == 0)
        assert torch.isfinite(s).all()

    def test_accepts_bf16_and_fp16(self):
        for dtype in (torch.bfloat16, torch.float16):
            w = _rand(8, 16, seed=3).to(dtype)
            q, s = i8.quantize_int8_per_channel(w)
            assert q.dtype == torch.int8 and s.dtype == torch.float32

    @pytest.mark.parametrize("shape", [(8,), (2, 3, 4)])
    def test_rejects_non_matrices(self, shape):
        with pytest.raises(ValueError, match="2-D"):
            i8.quantize_int8_per_channel(torch.ones(shape))

    def test_deterministic(self):
        w = _rand(8, 16, seed=4)
        a = i8.quantize_int8_per_channel(w)
        b = i8.quantize_int8_per_channel(w)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# --------------------------------------------------------------------------- config / detection


class TestConfig:
    def test_builds_a_dynamic_token_w8a8_int8_scheme(self):
        qc = _quant_config()
        scheme = qc.target_scheme_map["Linear"]
        w, a = scheme["weights"], scheme["input_activations"]
        assert (w.num_bits, str(w.type), w.symmetric, str(w.strategy), w.dynamic) == (8, "int", True, "channel", False)
        assert (a.num_bits, str(a.type), a.symmetric, str(a.strategy), a.dynamic) == (8, "int", True, "token", True)
        built = qc._get_scheme_from_parts(w, a)
        assert isinstance(built, CompressedTensorsW8A8Int8)
        assert not built.is_static_input_scheme and built.input_symmetric

    def test_ignores_lm_head_and_moe_routers(self):
        cfg = i8.build_int8_w8a8_quant_config(SimpleNamespace(num_hidden_layers=3))
        assert cfg["ignore"] == ["lm_head"] + [f"model.layers.{i}.mlp.gate" for i in range(3)]
        assert cfg["format"] == "int-quantized" and cfg["quant_method"] == "compressed-tensors"

    def test_missing_layer_count_ignores_only_lm_head(self):
        assert i8.build_int8_w8a8_quant_config(SimpleNamespace())["ignore"] == ["lm_head"]

    def test_is_int8_model(self):
        assert qu.is_int8_model(SimpleNamespace(quant_config=_quant_config()))
        assert qu.is_refit_quant_model(SimpleNamespace(quant_config=_quant_config()))

    def test_other_configs_are_not_int8(self):
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        fp8 = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128])
        assert not qu.is_int8_model(SimpleNamespace(quant_config=fp8))
        assert qu.is_refit_quant_model(SimpleNamespace(quant_config=fp8))
        assert not qu.is_int8_model(SimpleNamespace(quant_config=None))
        assert not qu.is_int8_model(SimpleNamespace())
        w4a16 = i8.build_int8_w8a8_quant_config(SimpleNamespace())
        w4a16["format"] = "pack-quantized"
        w4a16["config_groups"]["group_0"]["weights"].update(num_bits=4, strategy="group", group_size=128)
        del w4a16["config_groups"]["group_0"]["input_activations"]
        assert not qu.is_int8_model(SimpleNamespace(quant_config=CompressedTensorsConfig.from_config(w4a16)))


# --------------------------------------------------------------------------- refit on real vLLM layers


def _qkv(qc):
    return QKVParallelLinear(HIDDEN, HEAD, HEADS, KV_HEADS, bias=False, quant_config=qc, prefix="m.qkv_proj")


def _gate_up(qc):
    return MergedColumnParallelLinear(HIDDEN, [INTER, INTER], bias=False, quant_config=qc, prefix="m.gate_up_proj")


def _down(qc):
    return RowParallelLinear(INTER, HIDDEN, bias=False, quant_config=qc, prefix="m.down_proj")


def _o_proj(qc):
    # square: the live [K, N] transpose view has the checkpoint shape too, only the strides tell them apart
    return RowParallelLinear(HIDDEN, HIDDEN, bias=False, quant_config=qc, prefix="m.o_proj")


def _shards(kind, seed):
    if kind == "qkv":
        return {
            "q": _rand(HEADS * HEAD, HIDDEN, seed=seed),
            "k": _rand(KV_HEADS * HEAD, HIDDEN, seed=seed + 1),
            "v": _rand(KV_HEADS * HEAD, HIDDEN, seed=seed + 2),
        }
    if kind == "gate_up":
        return {0: _rand(INTER, HIDDEN, seed=seed), 1: _rand(INTER, HIDDEN, seed=seed + 1)}
    if kind == "o_proj":
        return {None: _rand(HIDDEN, HIDDEN, seed=seed)}
    return {None: _rand(HIDDEN, INTER, seed=seed)}


def _load(layer, shards):
    """What ``load_quanted_weights`` + vLLM's ``model.load_weights`` do per checkpoint tensor: give staged params
    their vLLM parameter class back, quantize, call the param loaders, restore the class."""
    swapped = []
    for param in layer.parameters(recurse=False):
        if hasattr(param, "subclass_type"):
            swapped.append((param, param.__class__))
            param.__class__ = param.subclass_type
    try:
        for shard_id, w in shards.items():
            q, s = i8.quantize_int8_per_channel(w)
            for name, t in (("weight", q), ("weight_scale", s)):
                param = getattr(layer, name)
                if shard_id is None:
                    param.weight_loader(param, t)
                else:
                    param.weight_loader(param, t, shard_id)
    finally:
        for param, cls in swapped:
            param.__class__ = cls


def _expected(shards) -> torch.Tensor:
    return torch.cat([_rtn(w) for w in shards.values()], dim=0)


@pytest.mark.usefixtures("vllm_ctx", "int8_patches")
@pytest.mark.parametrize("kind, build", [("qkv", _qkv), ("gate_up", _gate_up), ("down", _down), ("o_proj", _o_proj)])
class TestRefit:
    def _init(self, kind, build):
        layer = build(_quant_config())
        _load(layer, _shards(kind, seed=10))  # the engine's first load
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    def test_first_load_is_transposed_and_keeps_loader_metadata(self, kind, build):
        layer = self._init(kind, build)
        n = sum(layer.output_partition_sizes) if hasattr(layer, "output_partition_sizes") else layer.output_size
        assert layer.weight.shape[1] == n  # CUTLASS layout: [K, N]
        assert layer.weight.dtype == torch.int8
        for name in ("weight", "weight_scale"):
            param = getattr(layer, name)
            assert hasattr(param, "weight_loader") and getattr(param, "output_dim", None) == 0
        torch.testing.assert_close(_dequant(layer), _expected(_shards(kind, seed=10)))

    def test_refit_updates_values_in_place(self, kind, build):
        layer = self._init(kind, build)
        live_w, live_s = layer.weight, layer.weight_scale
        w_ptr, s_ptr = live_w.data_ptr(), live_s.data_ptr()

        staged = i8.stage_int8_params_for_loading(layer)
        assert staged == [layer]
        assert layer.weight is not live_w
        assert layer.weight.data_ptr() == w_ptr  # loads land straight in the live storage
        assert tuple(layer.weight.shape) == tuple(live_w.t().shape)  # checkpoint layout [N, K]

        new = _shards(kind, seed=20)
        _load(layer, new)
        i8.process_int8_weights_after_loading(staged)

        assert layer.weight is live_w and layer.weight_scale is live_s  # same Parameter objects
        assert layer.weight.data_ptr() == w_ptr and layer.weight_scale.data_ptr() == s_ptr
        torch.testing.assert_close(_dequant(layer), _expected(new))
        assert not hasattr(layer, i8._INT8_LIVE_ATTR)

    def test_survives_repeated_refits(self, kind, build):
        layer = self._init(kind, build)
        w_ptr = layer.weight.data_ptr()
        for seed in (30, 40, 50):
            staged = i8.stage_int8_params_for_loading(layer)
            new = _shards(kind, seed=seed)
            _load(layer, new)
            i8.process_int8_weights_after_loading(staged)
            torch.testing.assert_close(_dequant(layer), _expected(new))
            assert layer.weight.data_ptr() == w_ptr

    def test_dequantized_weights_track_the_bf16_weights(self, kind, build):
        layer = self._init(kind, build)
        new = _shards(kind, seed=60)
        staged = i8.stage_int8_params_for_loading(layer)
        _load(layer, new)
        i8.process_int8_weights_after_loading(staged)
        ref = torch.cat([w.float() for w in new.values()], dim=0)
        step = layer.weight_scale.data.float()
        assert torch.all((_dequant(layer) - ref).abs() <= step / 2 + 1e-6)


@pytest.mark.usefixtures("vllm_ctx", "int8_patches")
class TestRefitEdgeCases:
    def test_unpatched_layers_are_not_staged(self):
        assert i8.stage_int8_params_for_loading(torch.nn.Linear(4, 4)) == []

    def test_unexpected_layout_gets_a_poisoned_fresh_buffer(self):
        param = torch.nn.Parameter(torch.zeros(6, dtype=torch.int8), requires_grad=False)
        buf = i8._checkpoint_view(param, (2, 4), torch.int8)
        assert buf.data_ptr() != param.data_ptr() and torch.all(buf == -128)

    def test_square_transposed_view_is_staged_as_its_transpose(self):
        live = torch.arange(16, dtype=torch.int8).reshape(4, 4).t()  # the CUTLASS layout of a square weight
        param = torch.nn.Parameter(live, requires_grad=False)
        view = i8._checkpoint_view(param, (4, 4), torch.int8)
        assert view.data_ptr() == param.data_ptr() and view.is_contiguous()
        assert torch.equal(view, param.data.t())

    def test_matching_layout_reuses_storage(self):
        contiguous = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.int8), requires_grad=False)
        assert i8._checkpoint_view(contiguous, (4, 2), torch.int8).data_ptr() == contiguous.data_ptr()
        transposed = torch.nn.Parameter(torch.zeros(2, 4, dtype=torch.int8).t(), requires_grad=False)
        assert i8._checkpoint_view(transposed, (2, 4), torch.int8).data_ptr() == transposed.data_ptr()

    def test_a_contiguous_tensor_is_never_reinterpreted_as_its_transpose(self):
        param = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.int8), requires_grad=False)
        buf = i8._checkpoint_view(param, (2, 4), torch.int8)  # no view fits: fresh, poisoned buffer
        assert buf.data_ptr() != param.data_ptr() and torch.all(buf == -128)

    def test_fold_rejects_a_shape_change(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.int8), requires_grad=False)
        live = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.int8), requires_grad=False)
        with pytest.raises(RuntimeError, match="cannot be updated in place"):
            i8._fold_into_live(layer, "weight", live)

    def test_fold_copies_when_storage_moved(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.int8), requires_grad=False)
        live = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.int8), requires_grad=False)
        i8._fold_into_live(layer, "weight", live)
        assert layer.weight is live and torch.all(live == 1)

    def test_layout_is_recorded_once(self):
        layer = _qkv(_quant_config())
        layer.quant_method.process_weights_after_loading(layer)
        first = dict(getattr(layer, i8._INT8_LAYOUT_ATTR))
        layer.quant_method.process_weights_after_loading(layer)  # a second transform must not re-record [K, N]
        assert getattr(layer, i8._INT8_LAYOUT_ATTR) == first
        assert first["weight"] == (((HEADS + 2 * KV_HEADS) * HEAD, HIDDEN), torch.int8)


# --------------------------------------------------------------------------- weight stream + load path


class _TinyModel(torch.nn.Module):
    """A one-layer decoder shaped like vLLM's Qwen-style models: fused qkv / gate_up, stacked-param loading."""

    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
    _stacked = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    def __init__(self, qc):
        super().__init__()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = _qkv(qc)
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = _gate_up(qc)
        layer.mlp.down_proj = _down(qc)
        layer.input_layernorm = torch.nn.Module()
        layer.input_layernorm.weight = torch.nn.Parameter(torch.ones(HIDDEN), requires_grad=False)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer])
        self.model.embed_tokens = torch.nn.Module()
        self.model.embed_tokens.weight = torch.nn.Parameter(torch.zeros(10, HIDDEN), requires_grad=False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            for fused, part, shard in self._stacked:
                if part in name:
                    target = name.replace(part, fused)
                    params[target].weight_loader(params[target], tensor, shard)
                    loaded.add(target)
                    break
            else:
                param = params[name]
                loader = getattr(param, "weight_loader", None)
                if loader is not None:
                    loader(param, tensor)
                else:
                    param.data.copy_(tensor)
                loaded.add(name)
        return loaded


def _hf_weights(seed):
    p = "model.layers.0."
    return [
        (p + "self_attn.q_proj.weight", _rand(HEADS * HEAD, HIDDEN, seed=seed)),
        (p + "self_attn.k_proj.weight", _rand(KV_HEADS * HEAD, HIDDEN, seed=seed + 1)),
        (p + "self_attn.v_proj.weight", _rand(KV_HEADS * HEAD, HIDDEN, seed=seed + 2)),
        (p + "mlp.gate_proj.weight", _rand(INTER, HIDDEN, seed=seed + 3)),
        (p + "mlp.up_proj.weight", _rand(INTER, HIDDEN, seed=seed + 4)),
        (p + "mlp.down_proj.weight", _rand(HIDDEN, INTER, seed=seed + 5)),
        (p + "input_layernorm.weight", torch.full((HIDDEN,), 1.5, dtype=torch.bfloat16)),
        ("model.embed_tokens.weight", torch.full((10, HIDDEN), 0.25, dtype=torch.bfloat16)),
    ]


def _engine_model(seed=0):
    model = _TinyModel(_quant_config())
    model.load_weights(qu.quant_weights_int8(_hf_weights(seed), model))  # the engine's first load
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            method.process_weights_after_loading(module)
    return model


def _runner(model):
    vllm_config = SimpleNamespace(quant_config=_quant_config(), model_config=SimpleNamespace(dtype=torch.bfloat16))
    return SimpleNamespace(model=model, vllm_config=vllm_config)


@pytest.mark.usefixtures("vllm_ctx", "int8_patches")
class TestWeightStream:
    def test_only_int8_linear_weights_are_quantized(self):
        model = _engine_model()
        state = i8.stage_int8_params_for_loading(model)
        out = dict(qu.quant_weights_int8(_hf_weights(1), model))
        i8.process_int8_weights_after_loading(state)
        p = "model.layers.0."
        for proj in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            assert out[p + proj + ".weight"].dtype == torch.int8
            scale = out[p + proj + ".weight_scale"]
            assert scale.dtype == torch.float32 and scale.shape[1] == 1
        assert out[p + "input_layernorm.weight"].dtype == torch.bfloat16
        assert out["model.embed_tokens.weight"].dtype == torch.bfloat16
        assert p + "input_layernorm.weight_scale" not in out

    def test_already_quantized_weights_pass_through(self):
        model = _engine_model()
        q = torch.ones(HEADS * HEAD, HIDDEN, dtype=torch.int8)
        out = list(qu.quant_weights_int8([("model.layers.0.self_attn.q_proj.weight", q)], model))
        assert out == [("model.layers.0.self_attn.q_proj.weight", q)]

    def test_full_refit_cycle_matches_rtn_of_the_new_weights(self):
        model = _engine_model(seed=0)
        layer = model.model.layers[0]
        ptrs = {
            name: m.weight.data_ptr()
            for name, m in [
                ("qkv", layer.self_attn.qkv_proj),
                ("gate_up", layer.mlp.gate_up_proj),
                ("down", layer.mlp.down_proj),
            ]
        }
        runner = _runner(model)
        new = _hf_weights(seed=100)

        reload_state = qu.prepare_quanted_weights_for_loading(model)
        assert len(reload_state["int8_layers"]) == 3 and reload_state["fp8_layers"] == []
        half = len(new) // 2  # two buckets, as the bucketed IPC transfer delivers them
        qu.load_quanted_weights(new[:half], runner)
        qu.load_quanted_weights(new[half:], runner)
        qu.process_quanted_weights_after_loading(model, reload_state)

        w = dict(new)
        p = "model.layers.0."
        torch.testing.assert_close(
            _dequant(layer.self_attn.qkv_proj),
            torch.cat([_rtn(w[p + f"self_attn.{x}_proj.weight"]) for x in "qkv"]),
        )
        torch.testing.assert_close(
            _dequant(layer.mlp.gate_up_proj),
            torch.cat([_rtn(w[p + "mlp.gate_proj.weight"]), _rtn(w[p + "mlp.up_proj.weight"])]),
        )
        torch.testing.assert_close(_dequant(layer.mlp.down_proj), _rtn(w[p + "mlp.down_proj.weight"]))
        assert torch.all(layer.input_layernorm.weight == 1.5)
        assert torch.all(model.model.embed_tokens.weight == 0.25)
        assert ptrs == {
            "qkv": layer.self_attn.qkv_proj.weight.data_ptr(),
            "gate_up": layer.mlp.gate_up_proj.weight.data_ptr(),
            "down": layer.mlp.down_proj.weight.data_ptr(),
        }

    def test_load_without_staging_fails_loudly(self):
        # the live [K, N] layout does not fit the checkpoint tensors: staging is mandatory
        model = _engine_model()
        with pytest.raises((AssertionError, RuntimeError, ValueError)):
            qu.load_quanted_weights(_hf_weights(seed=7)[:1], _runner(model))


# --------------------------------------------------------------------------- startup patches and server config


def test_apply_vllm_quant_patches_installs_the_int8_patch(monkeypatch):
    original = CompressedTensorsW8A8Int8.process_weights_after_loading
    monkeypatch.setattr(qu.fp8_state, "vllm_patches", [])
    try:
        qu.apply_vllm_quant_patches()
        assert CompressedTensorsW8A8Int8.process_weights_after_loading is not original
    finally:
        for p in qu.fp8_state.vllm_patches:
            p.stop()
    assert CompressedTensorsW8A8Int8.process_weights_after_loading is original


def _server_self(quantization, num_layers=2):
    return SimpleNamespace(
        config=SimpleNamespace(quantization=quantization, quantization_config_file=None),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=num_layers)),
    )


@pytest.fixture
def server_mod(monkeypatch):
    mod = pytest.importorskip("verl.workers.rollout.vllm_rollout.vllm_async_server")
    calls = []
    monkeypatch.setattr(mod, "apply_vllm_quant_patches", lambda: calls.append(1))
    monkeypatch.delenv(i8.INT8_QUANT_ENABLED_ENV, raising=False)
    monkeypatch.delenv("VERL_VLLM_FP8_QUANT_ENABLED", raising=False)
    mod._test_patch_calls = calls
    return mod


def test_server_maps_int8_to_compressed_tensors(server_mod):
    quantization, overrides = server_mod.vLLMHttpServer._apply_quantization(_server_self("int8"))
    assert quantization == "compressed-tensors"
    assert overrides["quantization_config"] == i8.build_int8_w8a8_quant_config(SimpleNamespace(num_hidden_layers=2))
    assert os.environ[i8.INT8_QUANT_ENABLED_ENV] == "1"
    assert "VERL_VLLM_FP8_QUANT_ENABLED" not in os.environ
    assert server_mod._test_patch_calls == [1]


def test_server_fp8_path_is_unchanged(server_mod):
    quantization, overrides = server_mod.vLLMHttpServer._apply_quantization(_server_self("fp8"))
    assert quantization == "fp8"
    assert overrides["quantization_config"]["weight_block_size"] == [128, 128]
    assert i8.INT8_QUANT_ENABLED_ENV not in os.environ


@pytest.mark.parametrize("bad", ["int4", "INT8", "w8a8"])
def test_server_rejects_unknown_quantization(server_mod, bad):
    with pytest.raises(ValueError, match="int8"):
        server_mod.vLLMHttpServer._apply_quantization(_server_self(bad))


def test_server_without_quantization_is_untouched(server_mod):
    assert server_mod.vLLMHttpServer._apply_quantization(_server_self(None)) == (None, {})
