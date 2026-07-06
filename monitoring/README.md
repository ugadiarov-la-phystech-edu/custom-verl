# Per-GPU phase & utilization monitoring

> Step-by-step instruction (install → run `grpo_4k_4gpu_oversample.sh` with monitoring →
> read the results): **[USAGE.md](USAGE.md)**.

Live + historical view of **what every GPU is doing** during a verl run — generating with
vLLM, computing log-probs, training, syncing weights, sleeping, or idle — aligned with each
GPU's utilization/memory, in Grafana (backed by Prometheus, integrated with the Ray dashboard).

Works for all three trainer topologies:

| Trainer | Topology | What you see |
|---|---|---|
| `verl.trainer.main_ppo` | colocated, sequential | each GPU cycles gen → logprob → ref → update_actor → weight_sync |
| `verl.experimental.one_step_off_policy` | 2 pools, concurrent | rollout-pool GPUs in `gen` **while** actor-pool GPUs are in `update_actor` |
| `verl.experimental.fully_async_policy` | 2 actors, concurrent | same as above, fully decoupled |

## How it works

- **`verl_gpu_phase`** — a Ray custom metric (`verl/utils/gpu_phase.py`), one time series per
  GPU tagged `(rank, gpu, node, role)`. Trainer phases are emitted from the `register()`
  worker-dispatch decorator; `gen`/`sleep` are emitted by the vLLM async server on
  wake-up/sleep. Enabled by `VERL_GPU_PHASE_MONITOR=1`, zero overhead when unset.
- **`ray_node_gpus_utilization` / `ray_node_gram_*`** — Ray's built-in per-GPU metrics
  (`GpuIndex` matches the `gpu` tag, so phase and utilization line up with no mapping).
- **Prometheus** scrapes every node (Ray writes its service-discovery file), retains 15 days
  by default → "what was happening an hour ago" is just a Grafana time-range change.
- **Grafana** is auto-provisioned from Ray's session files + the dashboard in
  [`grafana/verl_gpu_phase_dashboard.json`](grafana/verl_gpu_phase_dashboard.json).

### Phase codes

| code | phase | emitted by |
|---|---|---|
| 0 | `idle` | between any bracketed phase (waiting for data / slow rollouts) |
| 1 | `gen` | rollout server wake-up (vLLM generating) |
| 2 | `logprob_fwd` | `compute_log_prob` |
| 3 | `ref_fwd` | `compute_ref_log_prob` |
| 4 | `values_fwd` | `compute_values` (critic runs only) |
| 5 | `update_actor` | `update_actor` |
| 6 | `update_critic` | `update_critic` |
| 7 | `weight_sync` | `update_weights` |
| 8 | `sleep` | rollout server sleep (colocated: GPUs handed to training) |

In colocated runs the same GPU has **two lanes**: a `rollout` lane (gen/sleep) and a trainer
lane (logprob/update/…), because both the vLLM server and the FSDP worker own it in turns.

## Quickstart (single training node)

No docker and no root needed: Prometheus and Grafana run as plain user processes from
standalone binaries, auto-downloaded on first use into `~/.verl-monitoring` (override with
`VERL_MONITORING_HOME`; Grafana version with `GRAFANA_VERSION`).

```bash
# 1. start the run -- the launch scripts already export VERL_GPU_PHASE_MONITOR=1
bash grpo_4k_8gpu_oversample.sh        # or async_grpo_4k_8gpu.sh, ...

# 2. once Ray is up (a few seconds in), start Prometheus + Grafana:
bash monitoring/start_monitoring.sh

# 3. open:
#    Grafana   http://<node>:3000/d/verl-gpu-phase   <- the per-GPU phase timeline
#    Ray       http://<node>:8265                    <- live cluster view (Metrics tab embeds Grafana)
#    Prometheus http://<node>:9090                   <- raw queries
```

To keep monitoring across runs, start Ray yourself first so the session (and Prometheus
service discovery) outlives a single script:

```bash
ray start --head --dashboard-host=0.0.0.0
bash monitoring/start_monitoring.sh
bash grpo_4k_8gpu_oversample.sh          # ray.init() attaches to the running cluster
```

Stop with `bash monitoring/stop_monitoring.sh` (Prometheus history is kept on disk).

## Historical queries

Any Grafana panel accepts an absolute or relative time range (e.g. "Last 1 hour",
"2026-07-07 10:00 → 11:00"). Raw PromQL examples in Prometheus/Grafana Explore:

```promql
ray_verl_gpu_phase                               # current phase per (gpu, role)
ray_verl_gpu_phase[1h]                           # phase history, last hour
avg_over_time(ray_node_gpus_utilization[10m])    # smoothed utilization
sum by (gpu) (ray_verl_gpu_phase == bool 0)      # which GPUs are idle right now
```

(The metric is registered as `verl_gpu_phase`; Ray prefixes custom metrics with `ray_`
on Prometheus export.)

## Remote access

If the training node is remote and ports are closed, tunnel:

```bash
ssh -L 3000:localhost:3000 -L 8265:localhost:8265 -L 9090:localhost:9090 <node>
```

For the Ray dashboard's embedded Metrics tab to render Grafana panels, the launch scripts
export `RAY_PROMETHEUS_HOST` / `RAY_GRAFANA_HOST` / `RAY_GRAFANA_IFRAME_HOST`
(defaulting to `localhost`); override them with the externally reachable URLs when the
browser is not on the training node, **before** the run starts Ray.

## Multi-node notes

- Set `VERL_GPU_PHASE_MONITOR=1` on every node (the launch script export covers workers
  spawned through Ray runtime env inheritance; `ray start` nodes need it in their env).
- Prometheus (started on the head node) scrapes all nodes automatically via Ray's
  service-discovery file.
- `/tmp/ray/session_latest` is recreated per Ray session — re-run
  `monitoring/start_monitoring.sh` after restarting the cluster to re-provision Grafana.
