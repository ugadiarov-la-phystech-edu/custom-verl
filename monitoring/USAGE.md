# How to: run `grpo_4k_4gpu_oversample.sh` with GPU monitoring

Step-by-step instruction: install the monitoring binaries, run the training script with
per-GPU phase monitoring, and read what monitoring collects. (Design/details:
[README.md](README.md).)

---

## 1. Install required binaries

Nothing needs root, docker, or pip. Requirements on the training node:

- `bash`, `curl`, `tar` (standard on any Linux node)
- the verl python env with `ray` on `PATH` (e.g. `source /samsung/uv-envs/verl-env/.venv/bin/activate`,
  or however `setup.sh` activates it)

**Prometheus and Grafana install themselves on the first run** of
`monitoring/start_monitoring.sh`:

| binary | source | installed to |
|---|---|---|
| Prometheus 3.x | downloaded by `ray metrics launch-prometheus` | `~/.verl-monitoring/prometheus-<ver>.linux-amd64/` |
| Grafana OSS 13.1.0 | `https://dl.grafana.com/oss/release/grafana-13.1.0.linux-amd64.tar.gz` (~350 MB) | `~/.verl-monitoring/grafana-13.1.0/` |

Change the install dir with `VERL_MONITORING_HOME=/path`, the Grafana version with
`GRAFANA_VERSION=x.y.z`. To uninstall everything: `rm -rf ~/.verl-monitoring`.

Binaries already present under `~/.verl-monitoring` are reused — the start script only
downloads what is missing. So for an **air-gapped node**, download the two tarballs
elsewhere and unpack them there yourself:

```bash
mkdir -p ~/.verl-monitoring && cd ~/.verl-monitoring
# fetch these two on a machine with internet, copy over, then:
tar -xzf grafana-13.1.0.linux-amd64.tar.gz && rm grafana-13.1.0.linux-amd64.tar.gz
tar -xzf prometheus-3.13.0.linux-amd64.tar.gz && rm prometheus-3.13.0.linux-amd64.tar.gz
# sources:
#   https://dl.grafana.com/oss/release/grafana-13.1.0.linux-amd64.tar.gz
#   https://github.com/prometheus/prometheus/releases/download/v3.13.0/prometheus-3.13.0.linux-amd64.tar.gz
#   (any prometheus-*/ linux-amd64 release works; the newest one found is used)
```

---

## 2. Run the training script with monitoring

`grpo_4k_4gpu_oversample.sh` already exports everything monitoring needs
(`VERL_GPU_PHASE_MONITOR=1` + the `RAY_PROMETHEUS_HOST`/`RAY_GRAFANA_HOST` vars),
so monitoring is on by default. Two ways to run:

### Option A — recommended: start Ray first (monitoring survives across runs)

```bash
cd /samsung/projects/custom-verl

# 1. start a Ray head that outlives any single training run
#    (add --port=6380 if something like redis already owns :6379)
ray start --head --dashboard-host=0.0.0.0

# 2. start Prometheus + Grafana (first run downloads the binaries, ~1-2 min)
bash monitoring/start_monitoring.sh

# 3. run training -- ray.init() inside main_ppo attaches to the running cluster
bash grpo_4k_4gpu_oversample.sh
```

You can now restart / re-run training scripts as often as you like; the Ray session,
Prometheus history, and Grafana stay up the whole time.

### Option B — quick one-off

```bash
bash grpo_4k_4gpu_oversample.sh &          # script starts its own Ray session
# wait ~20-30 s until "ray init kwargs" / dashboard messages appear, then:
bash monitoring/start_monitoring.sh
```

Caveat: here the Ray session (and Prometheus service discovery) belongs to this run;
after the run exits, re-run `start_monitoring.sh` for the next one.

### Turning monitoring off

`VERL_GPU_PHASE_MONITOR=0 bash grpo_4k_4gpu_oversample.sh` — the phase metric becomes a
zero-overhead no-op. Prometheus/Grafana simply show only Ray's built-in metrics.

---

## 3. Access what monitoring collects

Open (from the training node, or through the tunnel below):

| URL | What's there |
|---|---|
| `http://<node>:3000/d/verl-gpu-phase` | **the main view** — Grafana dashboard "verl · per-GPU phase & utilization" |
| `http://<node>:8265` | Ray dashboard: live cluster state, per-actor GPU/CPU, task timeline; its **Metrics** tab embeds the Grafana panels |
| `http://<node>:9090` | Prometheus: raw metric queries (PromQL) |

Grafana needs no login (anonymous viewer access is enabled by Ray's config).

### The Grafana dashboard

- **Per-GPU phase** (state timeline) — one lane per (GPU × role), each block labeled with
  what that GPU was doing:

  | phase | meaning |
  |---|---|
  | `idle` | waiting — no compute phase active (e.g. waiting for slow rollouts) |
  | `gen` | vLLM generating rollouts |
  | `logprob_fwd` / `ref_fwd` / `values_fwd` | log-prob / reference / critic forward passes |
  | `update_actor` / `update_critic` | training updates |
  | `weight_sync` | actor → rollout weight synchronization |
  | `sleep` | vLLM engine slept (colocated: GPUs handed over to training) |

  In this colocated script each GPU has **two lanes**: a `rollout` lane (gen/sleep) and a
  trainer lane (logprob/update/…) — together they show the full gen → train cycle.
- **GPU utilization (%)** and **GPU memory used (%)** — per-GPU lines whose `GpuIndex`
  matches the `gpu` of the phase lanes, so you can see e.g. utilization collapsing while
  a phase lane sits in `gen` during the decode long-tail.

### Comparing two experiments

Freeze each finished run into its own dashboard, pinned to that run's Ray session and time window:

```bash
python3 monitoring/grafana/make_run_dashboard.py --latest --name exp-a   # after run A
python3 monitoring/grafana/make_run_dashboard.py --latest --name exp-b   # after run B
python3 monitoring/grafana/make_run_dashboard.py --list                  # all known sessions
```

To watch a run *while it happens*, either open the always-live `/d/verl-gpu-phase`, or add
`--live` for a view scoped to that one experiment; re-run without `--live` afterwards to freeze
it at the same URL.

They appear at `/d/vgp-<name>-<hash>` within ~10s and keep working after Ray is gone. Do **not**
run `clean_monitoring_data.sh` in between — it deletes the metric history both dashboards read.
See [README.md](README.md#one-dashboard-per-experiment).

### Historical data ("what was happening an hour ago?")

Prometheus retains 90 days (`VERL_PROM_RETENTION`). In Grafana, use the time-range picker (top
right): pick "Last 1 hour", or an absolute range like `2026-07-07 10:00 → 11:00`. Zoom by dragging
on any panel. For raw history, query Prometheus (or Grafana → Explore):

```promql
ray_verl_gpu_phase                       # current phase per (gpu, role)
ray_verl_gpu_phase[1h]                   # phase history over the last hour
ray_node_gpus_utilization                # per-GPU utilization
100 * ray_node_gram_used / (ray_node_gram_used + ray_node_gram_available)  # VRAM %
```

### Accessing from your laptop (remote training node)

```bash
ssh -L 3000:localhost:3000 -L 8265:localhost:8265 -L 9090:localhost:9090 <training-node>
# then open http://localhost:3000/d/verl-gpu-phase locally
```

---

## 4. Stop monitoring

```bash
bash monitoring/stop_monitoring.sh    # stops Prometheus + Grafana
ray stop                              # only if you started Ray yourself (Option A)
```

Prometheus history and Grafana state stay on disk — the next
`start_monitoring.sh` resumes with all previous data.
