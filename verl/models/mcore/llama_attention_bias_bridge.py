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
"""Megatron-Bridge support for ``LlamaForCausalLM`` checkpoints with ``attention_bias=True``.

HF Llama's ``attention_bias`` adds a bias to all four attention projections (q/k/v/o). This is the
layout of openPangu-Embedded-7B after ``scripts/realias_openpangu_to_llama.py``. Megatron-Bridge's
``LlamaBridge`` maps ``attention_bias`` to ``add_qkv_bias`` but registers no bias parameter
mappings, and unmapped parameters only log a warning. Without this module such a checkpoint trains
with all its attention biases silently dropped.

Megatron has no switch for "attention output bias only": ``add_bias_linear`` puts a bias on
``linear_proj`` and on both MLP linears. So, for ``attention_bias=True`` and ``mlp_bias=False``,
this module:

* enables ``add_qkv_bias`` and ``add_bias_linear`` on the provider;
* maps ``linear_qkv.bias`` <-> ``q/k/v_proj.bias`` and ``linear_proj.bias`` <-> ``o_proj.bias``;
* freezes the MLP biases at zero before DDP / optimizer construction (a pre-wrap hook). Zero bias
  is exactly "no bias", they have no HF counterpart, and they are neither loaded nor exported.

Everything is conditional on ``hf_config.attention_bias``, so plain Llama checkpoints are
unaffected.

Enable per run with ``actor_rollout_ref.model.external_lib=verl.models.mcore.llama_attention_bias_bridge``.
The import runs in every worker (``HFModelConfig.__post_init__``) and applies the patch once.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_PATCH_MARKER = "_verl_attention_bias_patched"

# Megatron parameter names of the MLP biases that add_bias_linear creates but the HF model lacks.
MLP_BIAS_SUFFIXES = ("mlp.linear_fc1.bias", "mlp.linear_fc2.bias")
_UNMAPPED_MLP_BIAS_RE = re.compile(r"No mapping found for (megatron_param|global_name): .*mlp\.linear_fc[12]\.bias")


def needs_attention_bias(hf_config) -> bool:
    """True for Llama configs whose attention projections (q/k/v/o) carry a bias."""
    return bool(getattr(hf_config, "attention_bias", False))


def attention_bias_mappings():
    """Bias mappings missing from the stock ``LlamaBridge``."""
    from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping

    return [
        QKVMapping(
            megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
            q="model.layers.*.self_attn.q_proj.bias",
            k="model.layers.*.self_attn.k_proj.bias",
            v="model.layers.*.self_attn.v_proj.bias",
        ),
        AutoMapping(
            megatron_param="decoder.layers.*.self_attention.linear_proj.bias",
            hf_param="model.layers.*.self_attn.o_proj.bias",
        ),
    ]


def freeze_absent_mlp_biases(model_chunks):
    """Pre-wrap hook: zero and freeze the MLP biases that ``add_bias_linear`` created.

    Returns the model chunks unchanged, as Megatron-Bridge pre-wrap hooks must.
    """
    import torch

    frozen = 0
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if name.endswith(MLP_BIAS_SUFFIXES):
                with torch.no_grad():
                    param.zero_()
                param.requires_grad_(False)
                frozen += 1
    logger.info(
        "[llama_attention_bias_bridge] add_bias_linear=True with mlp_bias=False: froze %d MLP bias tensors at zero",
        frozen,
    )
    return model_chunks


class _DropUnmappedMlpBiasWarnings(logging.Filter):
    """Silences the per-parameter "No mapping found" warnings for the frozen MLP biases only."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _UNMAPPED_MLP_BIAS_RE.search(record.getMessage())


def apply_patch() -> bool:
    """Patch ``LlamaBridge`` in place. Idempotent; returns True if this call applied the patch."""
    from megatron.bridge.models.conversion import model_bridge
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.llama.llama_bridge import LlamaBridge

    if getattr(LlamaBridge, _PATCH_MARKER, False):
        return False

    original_provider_bridge = LlamaBridge.provider_bridge
    original_mapping_registry = LlamaBridge.mapping_registry

    def provider_bridge(self, hf_pretrained):
        provider = original_provider_bridge(self, hf_pretrained)
        hf_config = hf_pretrained.config
        if needs_attention_bias(hf_config):
            if getattr(hf_config, "mlp_bias", False):
                raise NotImplementedError(
                    "llama_attention_bias_bridge supports attention_bias=True with mlp_bias=False only"
                )
            provider.add_qkv_bias = True
            provider.add_bias_linear = True
            provider.register_pre_wrap_hook(freeze_absent_mlp_biases)
        return provider

    def mapping_registry(self):
        registry = original_mapping_registry(self)
        if not needs_attention_bias(self.hf_config):
            return registry
        return MegatronMappingRegistry(*registry.mappings, *attention_bias_mappings())

    LlamaBridge.provider_bridge = provider_bridge
    LlamaBridge.mapping_registry = mapping_registry
    setattr(LlamaBridge, _PATCH_MARKER, True)
    model_bridge.logger.addFilter(_DropUnmappedMlpBiasWarnings())
    return True


apply_patch()
