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
"""GPU test of the online W8A8-INT8 rollout refit on a real vLLM engine.

The engine is built exactly as ``vLLMHttpServer._apply_quantization`` builds it for ``quantization=int8``
(compressed-tensors W8A8-INT8 via ``hf_overrides``, dummy weights, refit patches installed first), then
refitted with bf16 checkpoint weights through the same ``prepare -> load_quanted_weights -> process`` sequence
the rollout worker runs. Checks: a refit replaces the dummy weights, refits keep the CUDA-graph storage, a
second refit with different weights changes the outputs, and the INT8 engine's log-probs stay close to a
bf16 engine's.

    INT8_TEST_MODEL=Qwen/Qwen3-4B pytest -s tests/workers/rollout/rollout_vllm/test_vllm_int8_refit.py
"""

import gc
import glob
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # model lives in this process: apply_model works
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # as in the launchers: no flashinfer JIT at sampling

from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from verl.utils.vllm import vllm_quant_utils as qu  # noqa: E402
from verl.utils.vllm.vllm_int8_utils import build_int8_w8a8_quant_config  # noqa: E402

MODEL = os.environ.get("INT8_TEST_MODEL", "Qwen/Qwen3-4B")
PROMPTS = ["The capital of France is", "1 + 1 ="]
TEXT = "The quick brown fox jumps over the lazy dog. Water boils at 100 degrees Celsius at sea level."


def _hf_weights():
    path = snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"])
    weights = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        weights.update(load_file(f))
    return weights


def _engine(int8: bool):
    kwargs = dict(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.35, max_model_len=1024, seed=0)
    if int8:
        qu.apply_vllm_quant_patches()
        hf = AutoConfig.from_pretrained(MODEL)
        kwargs.update(
            quantization="compressed-tensors",
            hf_overrides={"quantization_config": build_int8_w8a8_quant_config(hf)},
            load_format="dummy",
        )
    return LLM(**kwargs)


def _refit(llm, weights: dict):
    """The rollout worker's sequence (vLLMColocateWorkerExtension.update_weights_from_ipc for a quantized engine):
    stage, load in two buckets, re-derive. Runs on the worker, like the extension. Returns the staged-layer count."""

    def run(worker, weights):
        model_runner = worker.model_runner
        model = model_runner.model
        device = next(model.parameters()).device
        items = [(k, v.to(device)) for k, v in weights.items()]
        state = qu.prepare_quanted_weights_for_loading(model)
        half = len(items) // 2
        qu.load_quanted_weights(items[:half], model_runner)
        qu.load_quanted_weights(items[half:], model_runner)
        qu.process_quanted_weights_after_loading(model, state)
        torch.cuda.synchronize()
        return len(state["int8_layers"])

    return llm.collective_rpc(run, args=(weights,))[0]


def _weight_ptrs(llm):
    def run(worker):
        return {n: p.data_ptr() for n, p in worker.model_runner.model.named_parameters() if p.dtype == torch.int8}

    return llm.collective_rpc(run)[0]


def _greedy(llm):
    out = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=12))
    return [o.outputs[0].text for o in out]


def _prompt_logprobs(llm):
    out = llm.generate([TEXT], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
    lps = out[0].prompt_logprobs[1:]
    ids = out[0].prompt_token_ids[1:]
    return torch.tensor([lp[t].logprob for lp, t in zip(lps, ids, strict=True)])


@pytest.fixture(scope="module")
def results():
    weights = _hf_weights()
    llm = _engine(int8=True)
    dummy = _greedy(llm)
    ptrs = _weight_ptrs(llm)
    staged = _refit(llm, weights)
    real = _greedy(llm)
    lp_int8 = _prompt_logprobs(llm)
    ptrs_after = _weight_ptrs(llm)

    noisy = {
        k: (v + torch.randn_like(v, dtype=torch.float32).to(v.dtype) * v.float().std().to(v.dtype))
        if k.endswith("proj.weight")
        else v
        for k, v in weights.items()
    }
    _refit(llm, noisy)
    perturbed = _greedy(llm)
    _refit(llm, weights)
    restored = _greedy(llm)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    bf16 = _engine(int8=False)
    lp_bf16 = _prompt_logprobs(bf16)
    ref = _greedy(bf16)
    del bf16
    gc.collect()
    torch.cuda.empty_cache()
    return SimpleNamespace(
        dummy=dummy,
        staged=staged,
        real=real,
        perturbed=perturbed,
        restored=restored,
        ptrs=ptrs,
        ptrs_after=ptrs_after,
        lp_int8=lp_int8,
        lp_bf16=lp_bf16,
        ref=ref,
    )


def test_every_decoder_linear_is_staged(results):
    cfg = AutoConfig.from_pretrained(MODEL)
    assert results.staged == 4 * cfg.num_hidden_layers  # qkv, o, gate_up, down per layer


def test_refit_replaces_the_dummy_weights(results):
    assert results.real != results.dummy
    assert "Paris" in results.real[0]


def test_refits_keep_the_weight_storage(results):
    assert results.ptrs and results.ptrs == results.ptrs_after


def test_refit_with_other_weights_changes_outputs_and_back(results):
    assert results.perturbed != results.real
    assert results.restored == results.real


def test_int8_logprobs_track_bf16(results):
    diff = (results.lp_int8 - results.lp_bf16).abs()
    tail = diff[1:]  # the first scored position (right after the first token) is the most uncertain one
    print(
        f"\n|logp_int8 - logp_bf16| over {diff.numel()} tokens: mean {diff.mean():.4f} median {diff.median():.4f} "
        f"max {diff.max():.4f}; without the first position: mean {tail.mean():.4f}"
    )
    print(f"greedy int8: {results.real}\ngreedy bf16: {results.ref}")
    assert diff.median() < 0.05
    assert tail.mean() < 0.2
