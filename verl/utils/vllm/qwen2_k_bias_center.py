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

"""Key-bias centring for vLLM's Qwen2 attention (``VERL_QWEN2_K_BIAS_CENTER=1``), the fix for Qwen2.5 + FP8 KV cache.

Why. Qwen2/2.5 keep a bias on k_proj, and in Qwen2.5-7B it is large (up to |171| in layer 0, |414| in layer 27) and
sits in the lowest-frequency RoPE pairs, which barely rotate. Every cached key therefore carries a near-constant
offset of 100-500 in those channels. With ``kv_cache_dtype=fp8_e4m3`` the e4m3 grid there is 16-32 wide, so the small
token-dependent part of the key is rounded away and attention degenerates: 0% accuracy, ~30% of responses at the
length cap. Scales (calibrated, per-head, per-channel) cannot help - e4m3's precision is relative. See
notes/qwen25_fp8_kv_cache_fix.md.

What. The patched ``Qwen2Attention.forward`` caches ``RoPE(k) - b_k`` instead of ``RoPE(k)``:

    k = k - qkv_proj.bias[q_size : q_size + kv_size]     # right after rotary_emb, before attn

This is exact, not an approximation: for a query q_i every key it attends to is shifted by the same vector b_k, so
every logit of the row moves by the same q_i . b_k and softmax cancels it. The attention output - and so the policy
the trainer optimises - is unchanged; only the cached values become small enough for FP8 (the large bias sits where
RoPE is ~identity, so ``R_j b_k - b_k`` is tiny). Queries are deliberately left alone: shifting q would move each
logit by a different c . k_j and change the model.

Refit. The bias is read from the live parameter on every forward, never captured at init: the rollout engine starts
on dummy weights and every real weight, biases included, arrives through the weight-sync path, which updates the
parameter storage in place (so CUDA graphs see the new values too).

Opt-in: set ``VERL_QWEN2_K_BIAS_CENTER=1`` in the launching environment (forwarded to Ray actors by
``verl.trainer.constants_ppo.get_ppo_ray_runtime_env``). It is applied in every vLLM worker by
``vLLMColocateWorkerExtension.__new__`` before the model is built. Only ``Qwen2Attention`` with a qkv bias and no
QK-norm is affected; Qwen3 has its own attention class and is untouched.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

K_BIAS_CENTER_ENV = "VERL_QWEN2_K_BIAS_CENTER"

_ORIGINAL_FORWARD = None


def k_bias_center_enabled() -> bool:
    """Whether ``VERL_QWEN2_K_BIAS_CENTER`` asks for the patch.

    "1" turns it on; unset, "" or "0" leave it off; anything else is an error (no silent typos).
    """
    value = os.environ.get(K_BIAS_CENTER_ENV, "").strip()
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{K_BIAS_CENTER_ENV} must be '0' or '1', got {value!r}")


def _k_bias_centered_forward(self, positions, hidden_states):
    """``Qwen2Attention.forward`` (vLLM 0.24) with the k-bias removed from the keys right after RoPE."""
    bias = getattr(self.qkv_proj, "bias", None)
    if self.qk_norm or bias is None:
        # QK-norm variants (BAGEL etc.) and bias-free projections have no constant key offset to remove.
        return _ORIGINAL_FORWARD(self, positions, hidden_states)
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    # Live, TP-local k-bias (same [q, k, v] layout as the fused output); constant per query row -> exact.
    k = k - bias[self.q_size : self.q_size + self.kv_size]
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


def apply_qwen2_k_bias_center() -> None:
    """Install the key-bias-centred forward on vLLM's ``Qwen2Attention``. Idempotent."""
    global _ORIGINAL_FORWARD
    from vllm.model_executor.models import qwen2

    if qwen2.Qwen2Attention.forward is _k_bias_centered_forward:
        return
    _ORIGINAL_FORWARD = qwen2.Qwen2Attention.forward
    qwen2.Qwen2Attention.forward = _k_bias_centered_forward
    logger.info("Applied Qwen2 key-bias centring (%s=1): keys are cached as RoPE(k) - b_k", K_BIAS_CENTER_ENV)


def remove_qwen2_k_bias_center() -> None:
    """Restore the original ``Qwen2Attention.forward`` (no-op if the patch is not installed)."""
    global _ORIGINAL_FORWARD
    from vllm.model_executor.models import qwen2

    if qwen2.Qwen2Attention.forward is _k_bias_centered_forward:
        qwen2.Qwen2Attention.forward = _ORIGINAL_FORWARD
    _ORIGINAL_FORWARD = None


def maybe_apply_qwen2_k_bias_center() -> bool:
    """Apply the patch iff ``VERL_QWEN2_K_BIAS_CENTER=1``; returns whether it was applied."""
    if not k_bias_center_enabled():
        return False
    apply_qwen2_k_bias_center()
    return True
