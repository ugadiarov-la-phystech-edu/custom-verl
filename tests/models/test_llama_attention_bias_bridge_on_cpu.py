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
"""verl.models.mcore.llama_attention_bias_bridge: Megatron-Bridge support for Llama attention biases.

``TestWithFakeBridge`` runs everywhere: it swaps a minimal fake Megatron-Bridge into ``sys.modules``
and checks the patch logic. ``TestWithRealBridge`` runs the same checks against the installed
Megatron-Bridge and is skipped where it cannot be imported (it needs TransformerEngine).
"""

import importlib
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

MODULE = "verl.models.mcore.llama_attention_bias_bridge"
QKV_BIAS = "decoder.layers.*.self_attention.linear_qkv.bias"
PROJ_BIAS = "decoder.layers.*.self_attention.linear_proj.bias"


def _hf_config(attention_bias=None, mlp_bias=None):
    cfg = SimpleNamespace(model_type="llama")
    if attention_bias is not None:
        cfg.attention_bias = attention_bias
    if mlp_bias is not None:
        cfg.mlp_bias = mlp_bias
    return cfg


# --------------------------------------------------------------------------- fake Megatron-Bridge


class _FakeMapping:
    def __init__(self, megatron_param, **hf):
        self.megatron_param = megatron_param
        self.hf = hf


class _FakeRegistry:
    def __init__(self, *mappings):
        self.mappings = list(mappings)


class _FakeProvider:
    def __init__(self):
        self.add_qkv_bias = False
        self.add_bias_linear = False
        self.pre_wrap_hooks = []

    def register_pre_wrap_hook(self, hook, prepend=False):
        self.pre_wrap_hooks.append(hook)


class _FakeLlamaBridge:
    hf_config = None

    def provider_bridge(self, hf_pretrained):
        provider = _FakeProvider()
        # the stock base-class translation: attention_bias -> add_qkv_bias, mlp_bias -> add_bias_linear
        provider.add_qkv_bias = bool(getattr(hf_pretrained.config, "attention_bias", False))
        provider.add_bias_linear = bool(getattr(hf_pretrained.config, "mlp_bias", False))
        return provider

    def mapping_registry(self):
        return _FakeRegistry(
            _FakeMapping("decoder.layers.*.self_attention.linear_qkv.weight"),
            _FakeMapping("decoder.layers.*.self_attention.linear_proj.weight"),
        )


@pytest.fixture
def fake_bridge(monkeypatch):
    """Install a fake megatron.bridge and import a fresh copy of the patch module against it."""
    llama_mod = types.ModuleType("megatron.bridge.models.llama.llama_bridge")

    class LlamaBridge(_FakeLlamaBridge):
        pass

    llama_mod.LlamaBridge = LlamaBridge
    model_bridge_mod = types.ModuleType("megatron.bridge.models.conversion.model_bridge")
    model_bridge_mod.logger = logging.getLogger("fake.megatron.bridge.model_bridge")
    model_bridge_mod.logger.filters.clear()
    registry_mod = types.ModuleType("megatron.bridge.models.conversion.mapping_registry")
    registry_mod.MegatronMappingRegistry = _FakeRegistry
    param_mod = types.ModuleType("megatron.bridge.models.conversion.param_mapping")
    param_mod.QKVMapping = lambda megatron_param, q, k, v: _FakeMapping(megatron_param, q=q, k=k, v=v)
    param_mod.AutoMapping = lambda megatron_param, hf_param: _FakeMapping(megatron_param, hf_param=hf_param)
    conversion_pkg = types.ModuleType("megatron.bridge.models.conversion")
    conversion_pkg.model_bridge = model_bridge_mod

    for name, mod in {
        "megatron.bridge": types.ModuleType("megatron.bridge"),
        "megatron.bridge.models": types.ModuleType("megatron.bridge.models"),
        "megatron.bridge.models.llama": types.ModuleType("megatron.bridge.models.llama"),
        "megatron.bridge.models.llama.llama_bridge": llama_mod,
        "megatron.bridge.models.conversion": conversion_pkg,
        "megatron.bridge.models.conversion.model_bridge": model_bridge_mod,
        "megatron.bridge.models.conversion.mapping_registry": registry_mod,
        "megatron.bridge.models.conversion.param_mapping": param_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    module = importlib.import_module(MODULE)
    yield SimpleNamespace(module=module, LlamaBridge=LlamaBridge, logger=model_bridge_mod.logger)
    sys.modules.pop(MODULE, None)


def _bridge(fake, hf_config):
    bridge = fake.LlamaBridge()
    bridge.hf_config = hf_config
    return bridge


class TestWithFakeBridge:
    def test_import_applies_the_patch_once(self, fake_bridge):
        assert getattr(fake_bridge.LlamaBridge, "_verl_attention_bias_patched", False)
        patched = fake_bridge.LlamaBridge.provider_bridge
        assert fake_bridge.module.apply_patch() is False  # idempotent
        assert fake_bridge.LlamaBridge.provider_bridge is patched
        assert len(fake_bridge.logger.filters) == 1

    @pytest.mark.parametrize("cfg", [_hf_config(), _hf_config(attention_bias=False)])
    def test_plain_llama_is_untouched(self, fake_bridge, cfg):
        bridge = _bridge(fake_bridge, cfg)
        provider = bridge.provider_bridge(SimpleNamespace(config=cfg))
        assert (provider.add_qkv_bias, provider.add_bias_linear) == (False, False)
        assert provider.pre_wrap_hooks == []
        params = [m.megatron_param for m in bridge.mapping_registry().mappings]
        assert QKV_BIAS not in params and PROJ_BIAS not in params
        assert len(params) == 2

    def test_attention_bias_enables_both_bias_flags_and_the_freeze_hook(self, fake_bridge):
        cfg = _hf_config(attention_bias=True, mlp_bias=False)
        provider = _bridge(fake_bridge, cfg).provider_bridge(SimpleNamespace(config=cfg))
        assert provider.add_qkv_bias is True
        assert provider.add_bias_linear is True
        assert provider.pre_wrap_hooks == [fake_bridge.module.freeze_absent_mlp_biases]

    def test_attention_bias_adds_exactly_the_qkv_and_o_proj_bias_mappings(self, fake_bridge):
        cfg = _hf_config(attention_bias=True)
        mappings = _bridge(fake_bridge, cfg).mapping_registry().mappings
        by_name = {m.megatron_param: m for m in mappings}
        assert len(mappings) == 4
        assert by_name[QKV_BIAS].hf == {
            "q": "model.layers.*.self_attn.q_proj.bias",
            "k": "model.layers.*.self_attn.k_proj.bias",
            "v": "model.layers.*.self_attn.v_proj.bias",
        }
        assert by_name[PROJ_BIAS].hf == {"hf_param": "model.layers.*.self_attn.o_proj.bias"}
        # the stock mappings are preserved, in order, before the additions
        assert [m.megatron_param for m in mappings[:2]] == [
            "decoder.layers.*.self_attention.linear_qkv.weight",
            "decoder.layers.*.self_attention.linear_proj.weight",
        ]
        assert not any("mlp" in m.megatron_param for m in mappings)

    def test_mlp_bias_is_refused(self, fake_bridge):
        cfg = _hf_config(attention_bias=True, mlp_bias=True)
        with pytest.raises(NotImplementedError, match="mlp_bias=False"):
            _bridge(fake_bridge, cfg).provider_bridge(SimpleNamespace(config=cfg))

    def test_warning_filter_drops_only_the_frozen_mlp_bias_messages(self, fake_bridge, caplog):
        log = fake_bridge.logger
        with caplog.at_level(logging.WARNING, logger=log.name):
            log.warning("WARNING: No mapping found for megatron_param: decoder.layers.3.mlp.linear_fc1.bias")
            log.warning("WARNING: No mapping found for megatron_param: decoder.layers.3.mlp.linear_fc2.bias")
            log.warning("No mapping found for global_name: decoder.layers.0.mlp.linear_fc2.bias")
            log.warning("WARNING: No mapping found for megatron_param: decoder.layers.3.mlp.linear_fc1.weight")
            log.warning(
                "WARNING: No mapping found for megatron_param: decoder.layers.3.self_attention.linear_proj.bias"
            )
            log.warning("something unrelated about mlp.linear_fc1.bias")
        messages = [r.getMessage() for r in caplog.records]
        assert messages == [
            "WARNING: No mapping found for megatron_param: decoder.layers.3.mlp.linear_fc1.weight",
            "WARNING: No mapping found for megatron_param: decoder.layers.3.self_attention.linear_proj.bias",
            "something unrelated about mlp.linear_fc1.bias",
        ]


class _TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = torch.nn.Module()
        self.self_attention.linear_qkv = torch.nn.Linear(4, 12)
        self.self_attention.linear_proj = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Module()
        self.mlp.linear_fc1 = torch.nn.Linear(4, 16)
        self.mlp.linear_fc2 = torch.nn.Linear(8, 4)


class _TinyGPT(torch.nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList(_TinyLayer() for _ in range(n_layers))


@pytest.fixture
def hook_module(fake_bridge):
    return fake_bridge.module


class TestFreezeHook:
    def test_freezes_and_zeroes_only_the_mlp_biases(self, hook_module):
        mod = hook_module
        chunks = [_TinyGPT(), _TinyGPT(n_layers=2)]  # virtual-pipeline style: several chunks
        with torch.no_grad():
            for chunk in chunks:
                for p in chunk.parameters():
                    p.fill_(0.5)
        out = mod.freeze_absent_mlp_biases(chunks)
        assert out is chunks
        frozen = 0
        for chunk in chunks:
            for name, p in chunk.named_parameters():
                if name.endswith(("mlp.linear_fc1.bias", "mlp.linear_fc2.bias")):
                    frozen += 1
                    assert not p.requires_grad
                    assert torch.count_nonzero(p) == 0
                else:
                    assert p.requires_grad, name
                    assert torch.all(p == 0.5), name
        assert frozen == 2 * (3 + 2)

    def test_logs_the_frozen_count(self, hook_module, caplog):
        mod = hook_module
        with caplog.at_level(logging.INFO, logger=mod.logger.name):
            mod.freeze_absent_mlp_biases([_TinyGPT(n_layers=4)])
        assert "froze 8 MLP bias tensors at zero" in caplog.text

    def test_frozen_biases_get_no_gradient(self, hook_module):
        mod = hook_module
        model = _TinyGPT(n_layers=1)
        mod.freeze_absent_mlp_biases([model])
        layer = model.decoder.layers[0]
        x = torch.randn(2, 4)
        h = layer.mlp.linear_fc2(torch.nn.functional.silu(layer.mlp.linear_fc1(x))[:, :8])
        (layer.self_attention.linear_proj(h).sum()).backward()
        assert layer.mlp.linear_fc1.bias.grad is None
        assert layer.mlp.linear_fc2.bias.grad is None
        assert layer.mlp.linear_fc1.weight.grad is not None
        assert layer.self_attention.linear_proj.bias.grad is not None

    @pytest.mark.parametrize(
        "cfg, expected",
        [(_hf_config(), False), (_hf_config(attention_bias=False), False), (_hf_config(attention_bias=True), True)],
    )
    def test_needs_attention_bias(self, hook_module, cfg, expected):
        assert hook_module.needs_attention_bias(cfg) is expected


# --------------------------------------------------------------------------- real Megatron-Bridge


def _real_bridge_or_skip():
    try:
        from megatron.bridge.models.llama.llama_bridge import LlamaBridge  # noqa: F401
    except Exception as e:  # TransformerEngine / modelopt missing on CPU-only boxes
        pytest.skip(f"megatron.bridge not importable: {type(e).__name__}")
    sys.modules.pop(MODULE, None)
    importlib.import_module(MODULE)
    from megatron.bridge.models.llama.llama_bridge import LlamaBridge

    return LlamaBridge


class TestWithRealBridge:
    def _tiny_llama(self, attention_bias):
        from transformers import LlamaConfig

        return LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            attention_bias=attention_bias,
            mlp_bias=False,
        )

    @pytest.mark.parametrize("attention_bias", [False, True])
    def test_mapping_registry(self, attention_bias):
        LlamaBridge = _real_bridge_or_skip()
        bridge = LlamaBridge()
        bridge.hf_config = self._tiny_llama(attention_bias)
        params = {m.megatron_param for m in bridge.mapping_registry().mappings}
        assert (QKV_BIAS in params) is attention_bias
        assert (PROJ_BIAS in params) is attention_bias
        assert "decoder.layers.*.self_attention.linear_qkv.weight" in params

    @pytest.mark.parametrize("attention_bias", [False, True])
    def test_provider_flags(self, attention_bias):
        LlamaBridge = _real_bridge_or_skip()
        cfg = self._tiny_llama(attention_bias)
        bridge = LlamaBridge()
        bridge.hf_config = cfg
        provider = bridge.provider_bridge(SimpleNamespace(config=cfg, generation_config=None))
        assert provider.add_qkv_bias is attention_bias
        assert provider.add_bias_linear is attention_bias
