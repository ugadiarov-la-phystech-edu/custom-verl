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

"""Online W8A8-INT8 rollout for vLLM (``actor_rollout_ref.rollout.quantization=int8``).

The rollout engine is built as a compressed-tensors W8A8-INT8 model: every ``Linear`` (except ``lm_head`` and MoE
routers) holds per-output-channel symmetric INT8 weights and quantizes its input per token at run time
(``CompressedTensorsW8A8Int8``, CUTLASS ``scaled_mm``). Nothing is calibrated: at every weight sync the BF16
weights from the trainer are rounded to nearest on a per-row ``absmax / 127`` grid (``quantize_int8_per_channel``)
and loaded with their scales.

A refit needs two things vLLM does not provide:

* The CUTLASS kernel's post-load step replaces ``weight`` with a transposed view and drops the parameter
  metadata the weight loaders dispatch on (``output_dim``, ``input_dim``, ...). The patch installed by
  ``build_int8_method_patchers`` records the checkpoint layout and carries the metadata over.
* ``load_weights`` writes checkpoint-layout ``[N, K]`` tensors. ``stage_int8_params_for_loading`` hands it an
  ``[N, K]`` view of the live (transposed) storage and ``process_int8_weights_after_loading`` re-applies the
  kernel transform and reinstates the original parameter objects, so the storage a captured CUDA graph points
  at never moves.

``verl/utils/vllm/vllm_quant_utils.py`` drives these from the rollout worker, next to the FP8 path.
"""

import logging
import os
from unittest.mock import patch

import torch

from verl.utils.vllm.vllm_fp8_utils import _copy_param_subclass_attrs, _restore_layer_param_subclass_attrs

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Set in the vLLM server process so its worker subprocesses install the refit patches before the model builds.
INT8_QUANT_ENABLED_ENV = "VERL_VLLM_INT8_QUANT_ENABLED"

INT8_QMAX = 127  # symmetric: [-127, 127], so the grid is the same for positive and negative weights

_INT8_LAYOUT_ATTR = "_verl_int8_layout"
_INT8_LIVE_ATTR = "_verl_int8_live_params"
_INT8_REFIT_PARAM_NAMES = ("weight", "weight_scale")

_W8A8_SCHEME_PATH = (
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8."
    "CompressedTensorsW8A8Int8.process_weights_after_loading"
)


def build_int8_w8a8_quant_config(hf_config) -> dict:
    """compressed-tensors ``quantization_config`` for an online W8A8-INT8 rollout engine (``hf_overrides``).

    Weights: INT8, symmetric, one scale per output channel. Activations: INT8, symmetric, one scale per token,
    computed at run time. ``lm_head`` and MoE router gates stay in the model dtype, as on the FP8 path.
    """
    num_layers = int(getattr(hf_config, "num_hidden_layers", 0) or 0)
    ignore = ["lm_head"] + [f"model.layers.{layer}.mlp.gate" for layer in range(num_layers)]
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
            }
        },
        "ignore": ignore,
    }


def quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest symmetric INT8 with one scale per output row.

    Returns ``(q, scale)`` with ``q`` INT8 of the weight's shape and ``scale`` FP32 ``[N, 1]`` such that
    ``q * scale`` approximates ``weight`` to within ``scale / 2`` per element. All-zero rows get scale 1 (their
    codes are 0), so a dequantized zero row stays exactly zero and no scale is 0 or non-finite.
    """
    if weight.dim() != 2:
        raise ValueError(f"INT8 per-channel quantization expects a 2-D weight, got shape {tuple(weight.shape)}")
    w = weight.float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = amax / INT8_QMAX
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    q = torch.round(w / scale).clamp_(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q, scale.to(torch.float32)


def _record_int8_layout(layer, params) -> None:
    """Remember the checkpoint layout (shape, dtype) of the refit params, once, before the first transform."""
    if hasattr(layer, _INT8_LAYOUT_ATTR):
        return
    layout = {}
    for name in _INT8_REFIT_PARAM_NAMES:
        param = params.get(name)
        if isinstance(param, torch.nn.Parameter):
            layout[name] = (tuple(param.shape), param.dtype)
    if layout:
        setattr(layer, _INT8_LAYOUT_ATTR, layout)


def _wrap_process_weights_after_loading(original_fn):
    def _patched_process_weights_after_loading(self, layer) -> None:
        old_params = dict(layer.named_parameters(recurse=False))
        _record_int8_layout(layer, old_params)
        original_fn(self, layer)
        # the kernel's replace_parameter keeps only weight_loader; the loaders also need output_dim & co.
        _restore_layer_param_subclass_attrs(layer, old_params)

    return _patched_process_weights_after_loading


def build_int8_method_patchers():
    """Patchers (not started) that make ``CompressedTensorsW8A8Int8`` survive a refit. Empty when vLLM lacks it."""
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
            CompressedTensorsW8A8Int8,
        )
    except ImportError:
        logger.warning("vLLM has no CompressedTensorsW8A8Int8; INT8 rollout refit patches are not installed")
        return []
    return [
        patch(
            _W8A8_SCHEME_PATH,
            _wrap_process_weights_after_loading(CompressedTensorsW8A8Int8.process_weights_after_loading),
        )
    ]


def _checkpoint_view(param: torch.nn.Parameter, shape, dtype) -> torch.Tensor:
    """A checkpoint-layout tensor over the live storage, or a fresh buffer when no view fits.

    The CUTLASS transform stores ``weight`` as the transpose view of the ``[N, K]`` checkpoint tensor, so
    ``param.t()`` is that tensor again, sharing storage. A fresh buffer (only for an unexpected layout) is
    filled with 0x80 bytes (-128, a code the symmetric quantizer never emits) so a slice the stream fails to
    write is detectable instead of silently stale.
    """
    data = param.data
    # Decide by memory layout, not shape: for a square weight the transposed live view has the checkpoint shape too.
    if data.dtype == dtype and data.dim() == 2 and tuple(data.t().shape) == tuple(shape) and data.t().is_contiguous():
        return data.t()
    if data.dtype == dtype and tuple(data.shape) == tuple(shape) and data.is_contiguous():
        return data
    buffer = torch.empty(shape, dtype=dtype, device=data.device)
    buffer.view(torch.uint8).fill_(0x80)
    return buffer


def stage_int8_params_for_loading(model) -> list:
    """Put checkpoint-layout parameters on every INT8 layer so ``load_weights`` can write into them.

    The live parameters are set aside on the layer and reinstated by ``process_int8_weights_after_loading``.
    Returns the staged layers (empty for a model without INT8 layers).
    """
    staged_layers = []
    for layer in model.modules():
        layout = getattr(layer, _INT8_LAYOUT_ATTR, None)
        if not layout:
            continue
        live = {}
        for name, (shape, dtype) in layout.items():
            param = getattr(layer, name, None)
            if not isinstance(param, torch.nn.Parameter):
                continue
            live[name] = param
            staged = torch.nn.Parameter(_checkpoint_view(param, shape, dtype), requires_grad=False)
            _copy_param_subclass_attrs(staged, param)
            setattr(layer, name, staged)
        setattr(layer, _INT8_LIVE_ATTR, live)
        staged_layers.append(layer)
    logger.info("Staged %d INT8 layers for refit", len(staged_layers))
    return staged_layers


def _fold_into_live(layer, name, live_param) -> None:
    new = getattr(layer, name)
    new_data = new.data if isinstance(new, torch.nn.Parameter) else new
    if tuple(new_data.shape) != tuple(live_param.shape) or new_data.dtype != live_param.dtype:
        raise RuntimeError(
            f"INT8 refit re-derived {name} as {tuple(new_data.shape)}/{new_data.dtype}, but the live parameter "
            f"is {tuple(live_param.shape)}/{live_param.dtype}; its storage cannot be updated in place."
        )
    same_view = new_data.data_ptr() == live_param.data_ptr() and new_data.stride() == live_param.stride()
    if not same_view:
        live_param.data.copy_(new_data)
    setattr(layer, name, live_param)


def process_int8_weights_after_loading(layers) -> None:
    """Re-apply the kernel layout to the freshly loaded weights and reinstate the live parameters in place."""
    for layer in layers:
        live = getattr(layer, _INT8_LIVE_ATTR, None) or {}
        quant_method = getattr(layer, "quant_method", None)
        process = getattr(quant_method, "process_weights_after_loading", None)
        if process is not None:
            process(layer)
        for name, live_param in live.items():
            _fold_into_live(layer, name, live_param)
        if hasattr(layer, _INT8_LIVE_ATTR):
            delattr(layer, _INT8_LIVE_ATTR)
