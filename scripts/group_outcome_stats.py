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
"""Share of all-correct / all-wrong GRPO groups a model generates on DAPO-Math-17k, bf16 vs FP8 vLLM.

Samples ``--num-prompts`` prompts from the dataset, generates a group of ``--n`` responses per prompt with
vLLM (one TP1 engine per GPU) and scores every response with verl's math_dapo scorer. A group is all-correct
when all ``n`` responses are correct, all-wrong when none is: both have zero GRPO advantage and are what the
replay rollouter drops as degenerate.

    python -m scripts.group_outcome_stats --precision bf16      --num-prompts 1024
    python -m scripts.group_outcome_stats --precision fp8       --num-prompts 1024
    python -m scripts.group_outcome_stats --precision fp8+fp8kv --num-prompts 1024

Run from the repo root (``-m`` puts it on sys.path; vLLM's engine process imports the weight-loader extension
below by name).

THE ENGINE IS SET UP AS IN TRAINING (verl/workers/rollout/vllm_rollout/vllm_async_server.py):
  * --precision bf16 = examples/baselines/run_qwen2.5-7b_dapo17k_grpo_sync_tis_overlong_*.sh: bf16 weights,
    bf16 KV cache (kv_cache_dtype=auto).
  * --precision fp8  : quantization=fp8 with verl's 128x128-block FP8 config injected through hf_overrides and
    verl's vLLM quant patches (ROLLOUT_QUANT=fp8), bf16 KV cache (kv_cache_dtype=auto).
  * --precision fp8+fp8kv = ..._2+6_..._fp8_kv-fp8_tis-c8_token-mean_qwen2.5-7b.sh: the fp8 weights above plus an
    FP8 KV cache (kv_cache_dtype=fp8_e4m3, no calibrated scales).
  * In both, the engine starts on dummy weights (load_format=dummy) and the bf16 HF checkpoint is then pushed
    through the worker extension's _update_weights - the path every trainer->rollout weight sync takes, which
    for FP8 quantizes the bf16 weights on the fly (load_quanted_weights). The OOV-logit patch
    (monkey_patch_model) is applied as in training.
  * Sampling as in the rollout: temperature 1.0, top_p 1.0, top_k -1, 8,192 response tokens, n=16; prompts
    through the chat template (the Qwen2.5 template adds its default system prompt), prompts over 2,048 tokens
    filtered out (data.filter_overlong_prompts). Correctness is math_dapo's acc on the decoded response
    (skip_special_tokens=True), as in the reward managers; the overlong penalty does not change acc.
  * Engine knobs: max_num_seqs 256, max_num_batched_tokens 8192, chunked prefill and prefix caching on,
    cudagraph_mode FULL_AND_PIECEWISE, attention backend default (FlashAttention 3).

SEED. --seed fixes the prompt sample (a permutation of the filtered dataset) and the sampling: every prompt gets
its own vLLM request seed derived from (seed, prompt position), so two runs with the same seed and
--num-prompts see the same prompts with the same random streams, whatever --num-gpus is. bf16 and FP8 still
produce different responses (different logits).

OUTPUT (--output-dir, default outputs/group_outcome_stats/<precision>_seed<seed>_<UTC time>):
  * metrics.json  totals: prompts processed, generation time, all-correct / all-wrong / mixed shares, plus mean
                  accuracy, truncation share and mean response length; rewritten at every progress print.
  * groups.jsonl  one line per prompt group (dataset row, number correct, truncated, response lengths), to pair
                  bf16 and FP8 runs prompt by prompt.
Every --print-interval seconds (default 30) the current statistics are printed to stdout.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from datetime import datetime, timezone

DEFAULT_DATA = "/data2/datasets/math_datasets/dapo/dapo-math-17k.parquet"
EXTENSION_CLS = "scripts.group_outcome_stats.HFCheckpointLoaderExtension"
# --precision -> (FP8 weights, KV-cache dtype)
PRECISIONS = {"bf16": (False, "auto"), "fp8": (True, "auto"), "fp8+fp8kv": (True, "fp8_e4m3")}
# Per-prompt request seed: prompt k gets seed * SEED_STRIDE + k * n, its j-th response seed + j (vLLM's
# parallel sampling gives child j the parent seed + j).
SEED_STRIDE = 1 << 32


def _build_extension():
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    class HFCheckpointLoaderExtension(vLLMColocateWorkerExtension):
        """vLLM worker extension that pushes a bf16 HF checkpoint through verl's weight-sync path."""

        def load_hf_checkpoint(self, model_dir: str) -> dict:
            import glob

            import torch
            from safetensors import safe_open
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
            from verl.utils.vllm.vllm_quant_utils import (
                is_refit_quant_model,
                prepare_quanted_weights_for_loading,
                process_quanted_weights_after_loading,
            )

            if self.device is None:
                self.device = torch.device(f"cuda:{self.local_rank}")
            # Steps 1-3 of vLLMColocateWorkerExtension.update_weights_from_ipc, with the safetensors shards as
            # the buckets.
            quant = is_refit_quant_model(self.model_runner.vllm_config)
            quant_reload_states = None
            if quant:
                quant_reload_states = [
                    (model, prepare_quanted_weights_for_loading(model)) for model in self._iter_all_models()
                ]
            else:
                for model in self._iter_all_models():
                    patch_vllm_moe_model_weight_loader(model)
            files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
            if not files:
                raise FileNotFoundError(f"no *.safetensors in {model_dir}")
            n_tensors = 0
            for path in files:
                with safe_open(path, framework="pt", device=str(self.device)) as f:
                    weights = [(name, f.get_tensor(name)) for name in f.keys()]
                n_tensors += len(weights)
                self._update_weights(weights, peft_config=None, base_sync_done=False)
                del weights
                torch.cuda.empty_cache()
            if quant:
                for model, reload_state in quant_reload_states:
                    process_quanted_weights_after_loading(model, reload_state)
            else:
                for model, model_config in self._iter_all_models_with_config():
                    process_weights_after_loading(model, model_config, self.device)
            dtypes = {}
            for name, param in self.model_runner.model.named_parameters():
                if name.startswith("model.layers.0."):
                    dtypes[name] = str(param.dtype)
            return {"tensors": n_tensors, "files": len(files), "layer0_dtypes": dtypes}

    return HFCheckpointLoaderExtension


# vLLM's engine process imports the extension by name (EXTENSION_CLS); building it lazily keeps the CLI's
# own process free of the verl/vLLM import until a worker needs it.
def __getattr__(name):
    if name == "HFCheckpointLoaderExtension":
        cls = _build_extension()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model id or local path (default: %(default)s)")
    p.add_argument("--data", default=DEFAULT_DATA, help="DAPO-Math-17k parquet (default: %(default)s)")
    p.add_argument("--seed", type=int, default=1, help="prompt sample and vLLM sampling seed (default: %(default)s)")
    p.add_argument("--num-gpus", type=int, default=None, help="GPUs to use, one TP1 engine each (default: all)")
    p.add_argument("--num-prompts", type=int, required=True, help="prompts sampled from the dataset")
    p.add_argument(
        "--precision",
        choices=list(PRECISIONS),
        default="bf16",
        help="bf16: bf16 weights + KV; fp8: FP8 weights, bf16 KV; fp8+fp8kv: FP8 weights + fp8_e4m3 KV",
    )
    p.add_argument("--n", type=int, default=16, help="responses per prompt (default: %(default)s)")
    p.add_argument("--max-prompt-length", type=int, default=2048, help="longer prompts are filtered out")
    p.add_argument("--max-response-length", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--print-interval", type=float, default=30.0, help="seconds between progress prints")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args(argv)
    args.fp8_weights, args.kv_cache_dtype = PRECISIONS[args.precision]
    if args.num_prompts <= 0 or args.n <= 0:
        p.error("--num-prompts and --n must be positive")
    if args.print_interval <= 0:
        p.error("--print-interval must be positive")
    return args


def resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])


def load_prompts(args, tokenizer):
    """Filtered, seed-permuted prompt list: dicts with the dataset row, prompt token ids and ground truth."""
    import numpy as np
    import pandas as pd

    from verl.utils.tokenizer.chat_template import apply_chat_template

    df = pd.read_parquet(args.data)
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        messages = [dict(m) for m in row.prompt]
        ids = apply_chat_template(tokenizer, messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = list(ids)
        if len(ids) > args.max_prompt_length:
            continue
        rows.append(
            {
                "row": i,
                "prompt_ids": ids,
                "ground_truth": str(row.reward_model["ground_truth"]),
                "data_source": row.data_source,
            }
        )
    n_filtered = len(df) - len(rows)
    if args.num_prompts > len(rows):
        raise SystemExit(f"--num-prompts {args.num_prompts} > {len(rows)} prompts left after filtering")
    order = np.random.RandomState(args.seed).permutation(len(rows))[: args.num_prompts]
    prompts = [dict(rows[j], pos=k) for k, j in enumerate(order)]
    return prompts, len(df), n_filtered


def engine_kwargs(args, model_dir):
    kwargs = dict(
        model=model_dir,
        dtype="bfloat16",
        load_format="dummy",
        max_model_len=args.max_prompt_length + args.max_response_length,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
        worker_extension_cls=EXTENSION_CLS,
        disable_log_stats=True,
    )
    if args.fp8_weights:
        from transformers import AutoConfig

        n_layers = AutoConfig.from_pretrained(model_dir).num_hidden_layers
        kwargs["quantization"] = "fp8"
        kwargs["hf_overrides"] = {
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "ignored_layers": [f"model.layers.{i}.mlp.gate" for i in range(n_layers)],
            }
        }
    return kwargs


def worker(rank, args, model_dir, prompts, out_q, barrier):
    """One TP1 engine on the GPU in CUDA_VISIBLE_DEVICES (set by the parent before spawning)."""
    try:
        if args.fp8_weights:
            os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"
            from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches

            apply_vllm_quant_patches()
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind

        from verl.utils.reward_score import default_compute_score

        t_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        llm = LLM(seed=args.seed + rank, **engine_kwargs(args, model_dir))
        info = llm.collective_rpc("load_hf_checkpoint", args=(model_dir,))[0]
        llm.collective_rpc("monkey_patch_model", kwargs={"vocab_size": len(tokenizer), "banned_token_ids": None})
        llm.reset_prefix_cache()
        out_q.put(("ready", rank, {"load_s": time.time() - t_load, **info}))

        barrier.wait()
        engine = llm.llm_engine
        by_id = {}
        for pr in prompts:
            params = SamplingParams(
                n=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=1.0,
                max_tokens=args.max_response_length,
                seed=args.seed * SEED_STRIDE + pr["pos"] * args.n,
                detokenize=False,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            rid = str(pr["pos"])
            by_id[rid] = pr
            engine.add_request(rid, {"prompt_token_ids": pr["prompt_ids"]}, params)
        out_q.put(("started", rank, time.time()))

        while engine.has_unfinished_requests():
            for out in engine.step():
                if not out.finished:
                    continue
                pr = by_id.pop(out.request_id)
                correct, lengths, truncated = 0, [], 0
                for c in out.outputs:
                    text = tokenizer.decode(c.token_ids, skip_special_tokens=True)
                    res = default_compute_score(pr["data_source"], text, pr["ground_truth"])
                    acc = bool(res["acc"]) if isinstance(res, dict) else float(res) > 0
                    correct += acc
                    lengths.append(len(c.token_ids))
                    truncated += c.finish_reason == "length"
                rec = {
                    "pos": pr["pos"],
                    "row": pr["row"],
                    "n": len(out.outputs),
                    "correct": correct,
                    "truncated": truncated,
                    "lengths": lengths,
                    "gpu_rank": rank,
                }
                out_q.put(("group", rank, rec))
        out_q.put(("done", rank, time.time()))
    except BaseException:
        out_q.put(("error", rank, traceback.format_exc()))
        try:
            barrier.abort()
        except Exception:
            pass
        raise


class Stats:
    def __init__(self, n_total):
        self.n_total = n_total
        self.groups = 0
        self.all_correct = 0
        self.all_wrong = 0
        self.responses = 0
        self.correct = 0
        self.truncated = 0
        self.tokens = 0

    def add(self, rec):
        self.groups += 1
        self.all_correct += rec["correct"] == rec["n"]
        self.all_wrong += rec["correct"] == 0
        self.responses += rec["n"]
        self.correct += rec["correct"]
        self.truncated += rec["truncated"]
        self.tokens += sum(rec["lengths"])

    def as_dict(self, gen_s):
        g, r = max(self.groups, 1), max(self.responses, 1)
        return {
            "prompts_processed": self.groups,
            "prompts_total": self.n_total,
            "generation_time_s": round(gen_s, 1),
            "all_correct_pct": round(100.0 * self.all_correct / g, 2),
            "all_wrong_pct": round(100.0 * self.all_wrong / g, 2),
            "mixed_pct": round(100.0 * (self.groups - self.all_correct - self.all_wrong) / g, 2),
            "all_correct_groups": self.all_correct,
            "all_wrong_groups": self.all_wrong,
            "response_acc_pct": round(100.0 * self.correct / r, 2),
            "truncated_pct": round(100.0 * self.truncated / r, 3),
            "mean_response_len": round(self.tokens / r, 1),
            "generated_tokens": self.tokens,
        }


def fmt_line(m):
    return (
        f"[{datetime.now(timezone.utc):%H:%M:%S}] prompts {m['prompts_processed']}/{m['prompts_total']} | "
        f"gen {m['generation_time_s'] / 60:.1f} min | all-correct {m['all_correct_pct']:.2f}% | "
        f"all-wrong {m['all_wrong_pct']:.2f}% | mixed {m['mixed_pct']:.2f}% | acc {m['response_acc_pct']:.2f}% | "
        f"trunc {m['truncated_pct']:.3f}% | len {m['mean_response_len']:.0f}"
    )


def main(argv=None):
    args = parse_args(argv)
    import torch
    from transformers import AutoTokenizer

    n_visible = torch.cuda.device_count()
    num_gpus = args.num_gpus or n_visible
    if not 1 <= num_gpus <= n_visible:
        raise SystemExit(f"--num-gpus {num_gpus}: {n_visible} GPUs visible")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = visible.split(",")[:num_gpus] if visible else [str(i) for i in range(num_gpus)]

    model_dir = resolve_model_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts, n_rows, n_filtered = load_prompts(args, tokenizer)
    out_dir = args.output_dir or os.path.join(
        "outputs", "group_outcome_stats", f"{args.precision}_seed{args.seed}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    )
    os.makedirs(out_dir, exist_ok=True)
    config = {
        **{k: v for k, v in vars(args).items() if k != "output_dir"},
        "num_gpus": num_gpus,
        "model_dir": model_dir,
        "dataset_rows": n_rows,
        "filtered_overlong_prompts": n_filtered,
    }
    print(f"[group_outcome_stats] {json.dumps(config)}", flush=True)
    print(f"[group_outcome_stats] output: {out_dir}", flush=True)

    # vLLM's engine processes import EXTENSION_CLS (and verl) by name: put the repo root on their path.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (repo_root, os.environ.get("PYTHONPATH")) if p)

    # Prompt k goes to GPU k % num_gpus; sampling seeds depend only on k, not on the shard.
    shards = [prompts[r::num_gpus] for r in range(num_gpus)]
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    barrier = ctx.Barrier(num_gpus)
    procs = []
    saved_cvd = visible
    for rank in range(num_gpus):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
        proc = ctx.Process(target=worker, args=(rank, args, model_dir, shards[rank], out_q, barrier), daemon=False)
        proc.start()
        procs.append(proc)
    if saved_cvd is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd

    stats = Stats(len(prompts))
    ready, started, done = {}, {}, {}
    t_gen0 = None
    groups_f = open(os.path.join(out_dir, "groups.jsonl"), "w")

    def gen_elapsed(now=None):
        if t_gen0 is None:
            return 0.0
        end = max(done.values()) if len(done) == num_gpus else (now or time.time())
        return end - t_gen0

    def write_metrics(final=False):
        m = stats.as_dict(gen_elapsed())
        payload = {"final": final, "metrics": m, "engine": ready, "config": config}
        with open(os.path.join(out_dir, "metrics.json.tmp"), "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(os.path.join(out_dir, "metrics.json.tmp"), os.path.join(out_dir, "metrics.json"))
        return m

    next_print = time.time() + args.print_interval
    failed = None
    try:
        while len(done) < num_gpus:
            try:
                kind, rank, payload = out_q.get(timeout=1.0)
            except queue.Empty:
                kind = None
            if kind == "group":
                stats.add(payload)
                groups_f.write(json.dumps(payload) + "\n")
            elif kind == "ready":
                ready[rank] = payload
                print(f"[group_outcome_stats] GPU rank {rank} ready: {json.dumps(payload)}", flush=True)
            elif kind == "started":
                started[rank] = payload
                t_gen0 = min(started.values())
            elif kind == "done":
                done[rank] = payload
            elif kind == "error":
                failed = f"GPU rank {rank} failed:\n{payload}"
                break
            if kind is None:
                dead = [r for r, pr in enumerate(procs) if not pr.is_alive() and r not in done]
                if dead:
                    failed = f"GPU ranks {dead} exited without finishing"
                    break
            if time.time() >= next_print:
                next_print += args.print_interval
                groups_f.flush()
                if t_gen0 is not None:
                    print(fmt_line(write_metrics()), flush=True)
                else:
                    print(f"[group_outcome_stats] loading engines: {len(ready)}/{num_gpus} ready", flush=True)
    finally:
        groups_f.close()
        if failed:
            for pr in procs:
                if pr.is_alive():
                    pr.terminate()
        for pr in procs:
            pr.join(timeout=60)
    if failed:
        write_metrics()
        print(f"[group_outcome_stats] ERROR {failed}", file=sys.stderr, flush=True)
        return 1
    m = write_metrics(final=True)
    print(fmt_line(m), flush=True)
    print(f"[group_outcome_stats] final metrics: {json.dumps(m)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
