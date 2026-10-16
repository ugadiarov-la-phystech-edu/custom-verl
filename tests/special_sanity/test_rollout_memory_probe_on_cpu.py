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
"""CPU tests for scripts/rollout_memory_probe.py: arguments, phase / variant parsing, summary maths, snapshot parsing."""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.group_outcome_stats as gos  # noqa: E402
import scripts.rollout_memory_probe as probe  # noqa: E402


def test_defaults_match_the_2p6_run():
    a = probe.parse_args([])
    assert a.precision == "fp8+fp8kv" and a.fp8_weights and a.kv_cache_dtype == "fp8_e4m3"
    assert (a.max_num_seqs, a.max_num_batched_tokens, a.max_model_len) == (512, 10240, 10240)
    assert a.gpu_memory_utilization == 0.5 and a.load_format == "auto"
    assert a.phase_list == ["val", "sync", "train", "sync", "val", "sync"]
    assert a.val_logprobs == "none" and a.train_logprobs == "0"
    assert a.val_files == probe.DEFAULT_VAL_FILES and a.variant_list == []
    assert not (a.no_sleep_mode or a.no_async_scheduling or a.no_prefix_caching or a.enforce_eager)


@pytest.mark.parametrize("precision", list(gos.PRECISIONS))
def test_precision_mapping_is_group_outcome_stats(precision):
    a = probe.parse_args(["--precision", precision])
    assert (a.fp8_weights, a.kv_cache_dtype) == gos.PRECISIONS[precision]


@pytest.mark.parametrize(
    "argv",
    [
        ["--phases", ""],
        ["--phases", "train,warmup"],
        ["--val-share", "0"],
        ["--val-share", "1.5"],
        ["--val-requests", "0"],
        ["--train-concurrency", "8", "--n", "16"],
        ["--sample-interval", "0"],
        ["--max-model-len", "2048"],
        ["--variants", ";;"],
        ["--variants", "base;--max-num-seqs 256", "--num-gpus", "1"],
    ],
)
def test_invalid_args_rejected(argv):
    with pytest.raises(SystemExit):
        probe.parse_args(argv)


def test_parse_phases():
    assert probe.parse_phases(" train, sync ,val") == ["train", "sync", "val"]
    with pytest.raises(ValueError, match="unknown phase"):
        probe.parse_phases("val,eval")


def test_parse_variants():
    assert probe.parse_variants(None) == []
    got = probe.parse_variants("base; --max-num-seqs 256 ;--no-async-scheduling;--val-logprobs 0")
    assert got == [
        ("base", []),
        ("max-num-seqs_256", ["--max-num-seqs", "256"]),
        ("no-async-scheduling", ["--no-async-scheduling"]),
        ("val-logprobs_0", ["--val-logprobs", "0"]),
    ]


def test_strip_launcher_flags():
    argv = ["--precision", "fp8", "--variants", "base;--enforce-eager", "--num-gpus=2", "--output-dir", "o", "--n", "4"]
    assert probe.strip_launcher_flags(argv) == ["--precision", "fp8", "--n", "4"]


def test_select_val():
    rows = list(range(1920))
    assert probe.select_val(rows, 1.0, None) == rows
    half = probe.select_val(rows, 0.5, None)
    assert len(half) == 960 and half[:3] == [0, 2, 4]
    assert probe.select_val(rows, 0.5, 100) == half[:100]
    assert probe.select_val(rows, 1.0, 5000) == rows


def _rows(phase_idx, used, reserved=None):
    reserved = reserved or [-1] * len(used)
    return [
        {"phase_idx": phase_idx, "gpu_used_mib": u, "torch_reserved_mib": r} for u, r in zip(used, reserved, strict=True)
    ]


def test_summarize_val_peak_released_by_sync():
    phases = ["train", "sync", "val", "sync"]
    samples = (
        _rows(0, [77000, 77200, 76900], [1000, 1100, 1000])
        + _rows(1, [77100, 76800])
        + _rows(2, [77000, 89000, 89100], [900, 13000, 13100])
        + _rows(3, [89100, 76500])
    )
    s = probe.summarize(samples, phases)
    assert [r["phase"] for r in s] == phases
    assert s[0]["baseline_mib"] == 77000
    val = s[2]
    assert (val["start_mib"], val["peak_mib"], val["end_mib"]) == (77000, 89100, 89100)
    assert val["peak_over_baseline_mib"] == 12100
    assert (val["reserved_start_mib"], val["reserved_peak_mib"]) == (900, 13100)
    assert val["after_next_sync_mib"] == 76500 and val["released_by_next_sync"] is True
    assert "reserved_peak_mib" not in s[1]  # -1 = no RPC reading yet


def test_summarize_val_peak_kept_and_first_val_without_train():
    phases = ["val", "sync", "val"]
    samples = _rows(0, [77000, 109000]) + _rows(1, [100000]) + _rows(2, [100000, 101000])
    s = probe.summarize(samples, phases)
    assert s[0]["baseline_mib"] == 77000 and s[0]["peak_over_baseline_mib"] == 32000
    assert s[0]["released_by_next_sync"] is False  # 100000 > 77000 + tolerance
    assert "released_by_next_sync" not in s[2]  # no sync after it
    assert probe.summarize([], ["train"]) == [{"phase_idx": 0, "phase": "train", "samples": 0}]


def _ev(action, size, filename, line=1, name="f"):
    return {"action": action, "size": size, "frames": [{"filename": "/x/torch/cuda/a.py", "line": 9, "name": "t"}, {"filename": filename, "line": line, "name": name}]}


def test_top_alloc_sites():
    sp = "/env/lib/python3.12/site-packages/vllm/v1/sample/sampler.py"
    mr = "/env/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"
    snap = {
        "device_traces": [
            [
                _ev("segment_alloc", 6 * probe.MIB, sp, 10, "sample"),
                _ev("segment_alloc", 2 * probe.MIB, sp, 10, "sample"),
                _ev("segment_alloc", 4 * probe.MIB, mr, 20, "execute_model"),
                _ev("alloc", 1 * probe.MIB, mr, 20, "execute_model"),
                _ev("free_completed", 6 * probe.MIB, sp, 10, "sample"),
                {"action": "segment_alloc", "size": probe.MIB, "frames": []},
            ]
        ]
    }
    seg = probe.top_alloc_sites(snap, "segment_alloc")
    assert seg[0] == {"site": "vllm/v1/sample/sampler.py:10 sample", "count": 2, "mib": 8.0, "max_mib": 6.0}
    assert seg[1]["site"] == "vllm/v1/worker/gpu_model_runner.py:20 execute_model" and seg[1]["mib"] == 4.0
    assert seg[2]["site"] == "<no stack>"
    assert len(probe.top_alloc_sites(snap, "segment_alloc", k=1)) == 1
    assert probe.top_alloc_sites(snap, "alloc")[0]["count"] == 1
    assert probe.top_alloc_sites({}, "alloc") == []


def test_site_falls_back_to_first_frame():
    assert probe._site([{"filename": "/usr/lib/python3.12/foo.py", "line": 3, "name": "g"}]) == "/usr/lib/python3.12/foo.py:3 g"


def test_build_engine_args_mirror_rollout_server(tmp_path):
    a = probe.parse_args(["--precision", "bf16", "--cudagraph-max-size", "256", "--no-async-scheduling"])
    kw = probe.build_engine_args(a, str(tmp_path), str(tmp_path))
    assert kw["worker_extension_cls"] == probe.EXTENSION_CLS
    assert kw["distributed_executor_backend"] == "mp" and kw["enable_sleep_mode"] is True
    assert kw["load_format"] == "auto" and kw["logprobs_mode"] == "processed_logprobs"
    assert kw["compilation_config"] == {"cudagraph_mode": "FULL_AND_PIECEWISE", "max_cudagraph_capture_size": 256}
    assert kw["async_scheduling"] is False and kw["disable_log_stats"] is False
    assert (kw["max_num_seqs"], kw["max_num_batched_tokens"], kw["kv_cache_dtype"]) == (512, 10240, "auto")
    kw = probe.build_engine_args(probe.parse_args(["--precision", "bf16", "--no-sleep-mode", "--no-prefix-caching"]), "m", "m")
    assert kw["enable_sleep_mode"] is False and kw["enable_prefix_caching"] is False and "async_scheduling" not in kw


def test_configure_env(monkeypatch):
    for var in ("VERL_QWEN2_K_BIAS_CENTER", "VLLM_USE_DEEP_GEMM", "PYTORCH_CUDA_ALLOC_CONF", "VLLM_DISABLE_COMPILE_CACHE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    probe.configure_env(probe.parse_args(["--qwen2-k-bias-center", "--no-sleep-mode"]))
    assert os.environ["VERL_QWEN2_K_BIAS_CENTER"] == "1" and os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    assert os.environ["VLLM_USE_DEEP_GEMM"] == "1"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert os.environ["PYTHONPATH"].split(os.pathsep) == [REPO_ROOT, "/elsewhere"]


def test_extension_exposes_probe_rpcs():
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    cls = probe.MemoryProbeExtension
    assert issubclass(cls, vLLMColocateWorkerExtension)
    for name in ("load_hf_checkpoint", "probe_memory_stats", "probe_reset_peak", "probe_empty_cache", "probe_snapshot_start", "probe_snapshot_dump"):
        assert callable(getattr(cls, name)), name
    assert probe.EXTENSION_CLS == f"scripts.rollout_memory_probe.{cls.__name__}"
