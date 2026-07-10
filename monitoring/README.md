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
  Phases are set at transitions and re-recorded by a heartbeat thread every
  `VERL_GPU_PHASE_HEARTBEAT_S` seconds (default 15, `0` disables) — Ray's metrics agent stops
  exporting custom metrics that aren't re-recorded, which would leave long phases as isolated
  samples in Prometheus.
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
Each lane is titled `gpu <gpu> · <role> · <node ip>`.

### The handoff seam

The two lanes are emitted by two different processes (the FSDP worker and the vLLM replica),
and Ray pushes each process's gauge to the node's metrics agent only every 10s
(`metrics_report_interval_ms`), on independent schedules. At the `update_actor` → `gen`
handoff the trainer's *stale* value can therefore reach Prometheus alongside the rollout's
*fresh* `gen`, making one 10s sample look as though the GPU is generating and training at
once. It is always this pair, always at most one sample (~1% of samples, twice per GPU per
hour on a 9-minute iteration), and never any other combination — the reverse handoff passes
through `sleep`, which isn't real work, so it never collides.

A colocated GPU cannot train while vLLM generates on it, so the phase panel encodes that:
the trainer lane is pinned to `idle` whenever its GPU's rollout lane reads `gen`.

```promql
(ray_verl_gpu_phase{role!="rollout"} unless on (gpu, NodeAddress) (ray_verl_gpu_phase{role="rollout"} == 1))
  or (ray_verl_gpu_phase{role!="rollout"} * 0 and on (gpu, NodeAddress) (ray_verl_gpu_phase{role="rollout"} == 1))
```

The `unless` drops the trainer's sample during `gen`; the `* 0` branch puts it back as `idle`
rather than leaving a gap (a gap would let `spanNulls` stretch the previous `update_actor`
band across it, which is worse than the seam). Sample coverage is unchanged. To narrow the
seam at the source instead, lower Ray's push interval —
`ray.init(_system_config={"metrics_report_interval_ms": 1000})` — at the cost of 10× the
metric traffic; it shrinks the seam but never closes it.

## Utilization colored by phase

The **`GPU <n> · utilization by phase`** panel repeats once per GPU (pick which ones with the
`gpu` dashboard variable, and the node with `node`), and draws that GPU's utilization line
recolored by what it was doing: blue while generating, orange while running `update_actor`,
and so on — the same palette as the timeline. It answers "was this GPU at 40% *because* it
was generating, or because it was waiting?" without eyeballing two panels side by side.

Grafana can't color one series by the value of another, so the panel runs **one query per
phase**, each returning the utilization samples only while that GPU is in that phase:

```promql
label_replace(ray_node_gpus_utilization{GpuIndex=~"$gpu", ip=~"$node"}, "gpu", "$1", "GpuIndex", "(.+)")
  and on (gpu, ip) (
    label_replace(<effective phase>                  == 5, "ip", "$1", "NodeAddress", "(.+)")
 or label_replace(<effective phase, offset $__interval> == 5, "ip", "$1", "NodeAddress", "(.+)")
)
```

The `label_replace` calls exist because the two metrics don't share label names —
utilization is tagged `GpuIndex`/`ip`, the phase metric `gpu`/`NodeAddress`. Nine such
queries partition the line by phase, and `spanNulls: false` turns the transitions between
them into the color changes you see.

The `offset $__interval` half is what keeps the line **continuous**. Without it each series
stops one step before the next one starts, so Grafana has no segment to draw across a phase
boundary and the line breaks at every transition. Offsetting by exactly one step (`$__interval`
is the range query's step) also admits the sample at which the phase *was* 5 one step ago — so
the outgoing series' last point sits on the incoming series' first point and the two segments
meet. The boundary sample belongs to two series and is drawn twice, in the outgoing phase's
color and then the incoming one's; that single shared point is the whole cost.

The GPU's two lanes are collapsed into one **effective phase** — whichever lane is doing real
work wins, since in a colocated step exactly one of them is:

```promql
max by (gpu, NodeAddress) (ray_verl_gpu_phase{role="rollout"} == 1)   # generating beats all
  or max by (gpu, NodeAddress) (ray_verl_gpu_phase != 0 != 8)         # else: a lane doing real work
  or max by (gpu, NodeAddress) (ray_verl_gpu_phase)                   # else: idle (0) or sleep (8)
```

So `gen` beats the trainer's `idle`, `update_actor` beats the rollout's `sleep`, and a GPU
whose lanes are all idle/sleeping shows `idle`/`sleep`. Disaggregated pools, where a GPU has
only one lane, fall through the same expression unchanged. The first clause exists because of
the handoff seam described below: plain `max()` would let a stale `update_actor` (5) outrank a
fresh `gen` (1).

This panel issues 9 queries per displayed GPU per refresh — 72 range queries on an 8-GPU node.
That does **not** slow training down: measured against a live run, the 72 queries cost ~290 ms
of Prometheus CPU (~4 ms each, flat in the dashboard's time range), i.e. ~3% of one core at the
10s refresh, in a process that only reads its own TSDB and never touches the Ray workers. What
training pays for monitoring is unchanged by any panel: one `gauge.set()` per phase transition,
the heartbeat re-record, and a 10s scrape.

Over an SSH tunnel the *browser* is a different story — each query pays the tunnel round-trip
(~0.5 s on a typical link), and 72 of them make the dashboard feel sluggish. That is latency in
your terminal, not load on the node; narrow the `gpu` variable or slow the refresh if it bothers
you.

### Phase colors

The dashboard ships a default color per phase. Two ways to change them.

**In the browser.** You are browsing as an anonymous *Viewer*, which can't edit any panel —
that is why nothing is clickable, on this panel or on the utilization ones. Log in at
`/login` as `admin`/`admin` and both become editable. Phase colors are *value mappings*
(color per phase), not per-series colors: **Edit panel → Value mappings → color swatch**.
Ray's dashboard provider sets `allowUiUpdates`, so **Save** persists to Grafana's database
— but the file provider overwrites it whenever `verl_gpu_phase_dashboard.json` changes, and
`clean_monitoring_data.sh` wipes it. Good for trying colors out; not for keeping them.

**In the palette (reproducible).** `start_monitoring.sh` recolors the copy it hands to
Grafana, leaving the repo JSON untouched.

```bash
# one-off, for a single run
VERL_PHASE_COLORS='gen=#e0b400,sleep=#2a2a2a' bash monitoring/start_monitoring.sh

# persistent: create grafana/phase_colors.json (picked up automatically)
echo '{ "gen": "#e0b400", "update_actor": "purple", "fallback": "#333333" }' \
    > monitoring/grafana/phase_colors.json

# what colors am I actually using?
python3 monitoring/grafana/apply_phase_colors.py --print
```

Keys are phase names (`gen`) or codes (`1`) from the table above, plus `fallback` for values
with no mapping. Values are hex (`#3987e5`), `rgb()`/`rgba()`, or Grafana palette names
(`purple`, `semi-dark-blue`). `VERL_PHASE_COLORS` wins over the palette file, which wins over
the dashboard defaults; an unknown phase or malformed color fails the start with an error
rather than silently rendering gray. Point at a different palette file with
`VERL_PHASE_COLORS_FILE`.

To keep colors you picked in the UI, copy them out of **Dashboard settings → JSON Model**
(the `mappings` block) into `phase_colors.json`.

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
To start over, `bash monitoring/clean_monitoring_data.sh` stops both processes and wipes all
state (metric history, Grafana db, logs) while keeping the downloaded binaries.

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

If the training node is remote and ports are closed, tunnel. Grafana alone is enough for the
dashboard (the datasource is server-side `access: proxy`, so the browser never talks to
Prometheus directly):

```bash
ssh -L 3000:localhost:3000 <node>
```

For debugging, forward Prometheus and the Ray dashboard too:

```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 8265:localhost:8265 <node>
```

- **9090 (Prometheus)** — query the raw series when a Grafana panel looks wrong, to tell
  apart "metric not emitted" from "panel misconfigured":

  ```bash
  # is the phase metric there right now, and with which tags?
  curl -s 'http://localhost:9090/api/v1/query?query=ray_verl_gpu_phase'
  # history over a window (spot isolated samples / gaps in lanes)
  curl -s 'http://localhost:9090/api/v1/query_range?query=ray_verl_gpu_phase&start=<unix>&end=<unix>&step=60'
  ```

  `http://localhost:9090/targets` shows whether Prometheus is scraping all Ray nodes
  (service discovery breaks when the Ray session restarts under Prometheus).
- **8265 (Ray dashboard)** — live actor/worker state: check that workers are alive and
  where they run when a lane stops updating; the Metrics tab embeds the Grafana panels
  (needs `RAY_GRAFANA_IFRAME_HOST` reachable from the browser, see above).

For the Ray dashboard's embedded Metrics tab to render Grafana panels, the launch scripts
export `RAY_PROMETHEUS_HOST` / `RAY_GRAFANA_HOST` / `RAY_GRAFANA_IFRAME_HOST`
(defaulting to `localhost`); override them with the externally reachable URLs when the
browser is not on the training node, **before** the run starts Ray.

## Shared nodes: set `RAY_TMPDIR`

Ray's temp root (`/tmp/ray` by default) is world-writable and **shared by every user on the
node**. Two files in it are single-copy-per-root, not per-session:

- `prom_metrics_service_discovery.json` — the target list Prometheus watches, rewritten by
  Ray every 5s (write to `tmp_prom_metrics_service_discovery.json`, then rename).
- `session_latest` — the symlink monitoring resolves to find Ray's Prometheus/Grafana config.

If another user's Ray owns either one, yours cannot overwrite it. Ray logs the failed write
as a `WARNING` in `session_latest/logs/dashboard_ReportHead.log` and keeps running, so
Prometheus goes on scraping **the previous owner's ports**. Ports that happen to be reused
still answer, which is why the symptom is not "everything is down" but the far more
confusing "two targets up, one down, and every `ray_node_*` metric missing" — the reporter
agent, which publishes GPU utilization and memory, is the one that moved.

Give your Ray a temp root nobody else writes to, **before starting Ray** and before the
monitoring scripts (they read the same variable):

```bash
export RAY_TMPDIR=/tmp/ray-$(id -un)    # e.g. in setup.sh / activate.sh
```

Mind the indirection: `RAY_TMPDIR` names the *system* temp dir and Ray appends `ray` to it
(`ray._common.utils.get_default_ray_temp_dir`), so the root above is `/tmp/ray-<user>/ray`,
not `/tmp/ray-<user>`. The monitoring scripts ask Ray for the resolved root rather than
recomputing it, so they follow whatever the running Ray decided.

`start_monitoring.sh` preflights all of this and refuses to start with an explanation rather
than bringing up a Grafana whose panels are quietly empty. If you are already wedged, delete
the stale file (Ray recreates it within 5s) — `/tmp/ray` has no sticky bit, so you may unlink
another user's leftovers:

```bash
rm -f /tmp/ray/tmp_prom_metrics_service_discovery.json
curl -s localhost:9090/api/v1/targets | grep -c '"health":"up"'
```

## Multi-node notes

- Set `VERL_GPU_PHASE_MONITOR=1` on every node (the launch script export covers workers
  spawned through Ray runtime env inheritance; `ray start` nodes need it in their env).
- Prometheus (started on the head node) scrapes all nodes automatically via Ray's
  service-discovery file.
- `<ray root>/session_latest` (`/tmp/ray/session_latest` by default) is recreated per Ray session — re-run
  `monitoring/start_monitoring.sh` after restarting the cluster to re-provision Grafana.
- Export `RAY_TMPDIR` on every node (and in the shell that runs the monitoring scripts),
  or on none — a mismatch sends Prometheus looking for a service-discovery file that Ray
  is not writing.
