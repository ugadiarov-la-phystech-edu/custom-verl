#!/usr/bin/env bash
# Start Prometheus + Grafana for verl per-GPU phase monitoring. No docker, no root:
# both run as plain user processes from standalone binaries downloaded on first use
# into ${VERL_MONITORING_HOME} (default ~/.verl-monitoring).
#
# Prerequisite: a Ray session must be live on this node -- either the training script
# already called ray.init(), or you started one with:
#     ray start --head --dashboard-host=0.0.0.0
# (start Ray first so its Prometheus service-discovery + Grafana provisioning files exist).
#
# On a node shared with other users, set RAY_TMPDIR (before starting Ray *and* here) to a
# path only you own -- see the preflight below for why. Ray puts its root at $RAY_TMPDIR/ray:
#     export RAY_TMPDIR=/tmp/ray-$(id -un)
#
# After this script:
#   Ray dashboard  http://<node>:8265   (live per-GPU util, actor/task view)
#   Prometheus     http://<node>:9090   (raw metrics + history)
#   Grafana        http://<node>:3000   (dashboard "verl · per-GPU phase & utilization")
set -euo pipefail

# Ray's temp root. Everything below lives under it: the session dir (Prometheus + Grafana
# config Ray generates) and the Prometheus service-discovery file, which is per-temp-root
# rather than per-session and therefore shared by every Ray run that uses this root.
#
# NB: RAY_TMPDIR is the *system* temp dir, not the root -- Ray appends "ray" to it, so
# RAY_TMPDIR=/tmp/ray-alice puts the root at /tmp/ray-alice/ray. Ask Ray itself rather than
# reimplement that rule, and fall back to it only when ray is not importable.
ray_temp_root() {
    python3 - <<'PY' 2>/dev/null
try:
    from ray._common.utils import get_default_ray_temp_dir   # ray >= 2.49
except ImportError:
    from ray._private.utils import get_ray_temp_dir as get_default_ray_temp_dir
print(get_default_ray_temp_dir())
PY
}
RAY_ROOT="$(ray_temp_root || true)"
[ -n "${RAY_ROOT}" ] || RAY_ROOT="${RAY_TMPDIR:-${TMPDIR:-/tmp}}/ray"
RAY_SESSION="${RAY_ROOT}/session_latest"
SD_FILE="${RAY_ROOT}/prom_metrics_service_discovery.json"
SD_TMP="${RAY_ROOT}/tmp_prom_metrics_service_discovery.json"
MONITORING_HOME="${VERL_MONITORING_HOME:-${HOME}/.verl-monitoring}"
GRAFANA_VERSION="${GRAFANA_VERSION:-13.1.0}"
# Tarball layout has varied across releases (grafana-13.1.0/ vs grafana-v11.x/).
GRAFANA_HOME=""
for d in "${MONITORING_HOME}/grafana-${GRAFANA_VERSION}" "${MONITORING_HOME}/grafana-v${GRAFANA_VERSION}"; do
    [ -x "${d}/bin/grafana" ] && GRAFANA_HOME="${d}" && break
done
GRAFANA_PID_FILE="${MONITORING_HOME}/grafana.pid"
GRAFANA_LOG="${MONITORING_HOME}/grafana.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_SRC="${SCRIPT_DIR}/grafana/verl_gpu_phase_dashboard.json"

mkdir -p "${MONITORING_HOME}"

if [ ! -e "${RAY_SESSION}/metrics" ]; then
    echo "No live Ray session at ${RAY_SESSION}."
    echo "Start the training run (or 'ray start --head --dashboard-host=0.0.0.0') first."
    if [ "${RAY_ROOT}" != "/tmp/ray" ]; then
        echo "(resolved Ray root: ${RAY_ROOT} -- was Ray started with the same RAY_TMPDIR?)"
    fi
    exit 1
fi

# --- 0. Preflight: can Ray actually keep Prometheus pointed at itself? -----------------
# Ray rewrites ${SD_FILE} (the Prometheus file_sd target list) every 5s, via a write to
# ${SD_TMP} + rename. Both live in the Ray root, which is world-writable and shared across
# users. If another user's Ray left either file behind, ours cannot overwrite it: the
# write fails with EPERM, Ray logs it as a WARNING to dashboard_ReportHead.log and carries
# on, and Prometheus keeps scraping whatever ports the *previous* owner's session used.
# Some of those ports get reused by chance, so the failure looks like "two targets up, one
# down, and ray_node_* metrics missing" rather than anything pointing at permissions.
suggest_tmpdir() {
    echo
    echo "Fix: give your Ray a temp root nobody else writes to, then restart Ray and this script:"
    echo "    export RAY_TMPDIR=/tmp/ray-\$(id -un)      # before starting Ray, and before this script"
    echo "  (Ray appends '/ray', so the root becomes /tmp/ray-\$(id -un)/ray.)"
}

for f in "${SD_TMP}" "${SD_FILE}"; do
    [ -e "${f}" ] || continue
    [ -w "${f}" ] && continue
    echo "ERROR: ${f}"
    echo "  exists but is not writable by $(id -un) (owner: $(stat -c %U "${f}" 2>/dev/null || echo '?'))."
    echo "  Ray cannot refresh the Prometheus target list, so metrics will silently go stale:"
    echo "  ray_node_gpus_utilization and the phase panels will be empty."
    if [ -w "${RAY_ROOT}" ] && [ ! -k "${RAY_ROOT}" ]; then
        echo
        echo "  It is a stale leftover and you may unlink it (no sticky bit on ${RAY_ROOT}):"
        echo "      rm -f ${f}"
        echo "  Ray recreates it within 5s. Verify with: curl -s localhost:9090/api/v1/targets"
    fi
    suggest_tmpdir
    exit 1
done

# Same story for the session dir: we provision our dashboard into it.
DASHBOARD_DIR="${RAY_SESSION}/metrics/grafana/dashboards"
mkdir -p "${DASHBOARD_DIR}" 2>/dev/null || true   # Ray usually made it; report below if we cannot
if [ ! -w "${DASHBOARD_DIR}" ]; then
    echo "ERROR: ${DASHBOARD_DIR} is not writable by $(id -un)"
    echo "  (owner: $(stat -Lc %U "${DASHBOARD_DIR}" 2>/dev/null || echo '?'))."
    echo "  Cannot provision the dashboard. When ${RAY_ROOT} is shared between users,"
    echo "  session_latest points at whichever Ray started last -- possibly not yours."
    suggest_tmpdir
    exit 1
fi

# Targets older than the session mean Ray never managed to refresh them for this run.
if [ -e "${SD_FILE}" ] && [ "${SD_FILE}" -ot "${RAY_SESSION}/metrics" ]; then
    echo "WARNING: ${SD_FILE} predates the current Ray session."
    echo "  Prometheus is scraping ports from an older session; expect missing ray_node_* metrics."
    echo "  Check for write errors: grep -i permission ${RAY_SESSION}/logs/dashboard_ReportHead.log"
fi

# --- 1. Prometheus -------------------------------------------------------------------
# Uses the config Ray generated for this session: scrapes every node in the cluster via
# ${SD_FILE} every 10s. Default retention is 15d, so
# "what was happening an hour ago" is a Grafana time-range change away.
# A binary already present under MONITORING_HOME (from a previous run, or hand-installed
# on an air-gapped node) is started directly; otherwise Ray's launcher downloads one.
# NB: `ray metrics launch-prometheus` re-downloads unconditionally, hence the local-first
# check; it also downloads into the CWD, so it is run from MONITORING_HOME.
PROM_PID_FILE="${MONITORING_HOME}/prometheus.pid"
PROM_LOG="${MONITORING_HOME}/prometheus.log"
if curl -sf http://localhost:9090/-/ready >/dev/null 2>&1; then
    echo "Prometheus already running on :9090"
    # It may predate the current Ray session, or point at a different RAY_TMPDIR entirely.
    # Its scrape config decides which SD file it reads, so a mismatch means it is watching
    # someone else's targets. Cheap to check, impossible to spot from an empty dashboard.
    PROM_CFG="$(curl -sf http://localhost:9090/api/v1/status/flags 2>/dev/null \
        | sed -n 's/.*"config\.file":"\([^"]*\)".*/\1/p')"
    case "${PROM_CFG}" in
        "${RAY_ROOT}"/*|"") ;;
        *) echo "WARNING: it was started with --config.file ${PROM_CFG},"
           echo "  which is outside the Ray root ${RAY_ROOT}. It is scraping another Ray's targets."
           echo "  Restart it: bash $(dirname "${BASH_SOURCE[0]}")/stop_monitoring.sh && $0" ;;
    esac
else
    PROM_BIN="$(ls -d "${MONITORING_HOME}"/prometheus-*/prometheus 2>/dev/null | sort -V | tail -1)"
    if [ -n "${PROM_BIN}" ] && [ -x "${PROM_BIN}" ]; then
        # Same data dir the Ray launcher uses (its CWD default), so history is continuous.
        nohup "${PROM_BIN}" \
            --config.file "${RAY_SESSION}/metrics/prometheus/prometheus.yml" \
            --storage.tsdb.path "${MONITORING_HOME}/data" \
            --web.enable-lifecycle \
            >"${PROM_LOG}" 2>&1 &
        echo $! > "${PROM_PID_FILE}"
    else
        (cd "${MONITORING_HOME}" && ray metrics launch-prometheus)
    fi
    for _ in $(seq 30); do
        curl -sf http://localhost:9090/-/ready >/dev/null 2>&1 && break
        sleep 1
    done
    curl -sf http://localhost:9090/-/ready >/dev/null || {
        echo "Prometheus did not come up on :9090 -- see ${PROM_LOG}"; exit 1; }
    echo "Prometheus up on :9090"
fi

# --- 2. Provision the verl dashboard --------------------------------------------------
# Drop our dashboard next to Ray's built-in ones; the dashboard provider Ray generated
# under session_latest/metrics/grafana/provisioning picks up everything in that folder.
# Phase-lane colors can be overridden per run without editing the checked-in JSON, via
# grafana/phase_colors.json or VERL_PHASE_COLORS='gen=#e0b400,sleep=#2a2a2a'.
DASHBOARD_DST="${DASHBOARD_DIR}/$(basename "${DASHBOARD_SRC}")"
RECOLOR="${SCRIPT_DIR}/grafana/apply_phase_colors.py"
if command -v python3 >/dev/null 2>&1 && [ -f "${RECOLOR}" ]; then
    python3 "${RECOLOR}" "${DASHBOARD_SRC}" "${DASHBOARD_DST}"
else
    cp "${DASHBOARD_SRC}" "${DASHBOARD_DST}"
fi
echo "Provisioned $(basename "${DASHBOARD_SRC}")"

# --- 3. Grafana -----------------------------------------------------------------------
# Standalone OSS binary (first run downloads ~350MB, unpacked under MONITORING_HOME).
# Started with the grafana.ini Ray generated for this session: it enables anonymous
# viewer access + iframe embedding (for the Ray dashboard Metrics tab) and points
# [paths]provisioning at the session's datasource/dashboard provisioning.
# Grafana's own state (sqlite db, logs, plugins) stays under its homepath data/.
if curl -sf http://localhost:3000/api/health >/dev/null 2>&1; then
    echo "Grafana already running on :3000"
else
    if [ -z "${GRAFANA_HOME}" ]; then
        echo "Downloading grafana ${GRAFANA_VERSION} (first run only)..."
        tarball="${MONITORING_HOME}/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
        curl -fL --retry 3 -o "${tarball}" \
            "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
        tar -xzf "${tarball}" -C "${MONITORING_HOME}"
        rm -f "${tarball}"
        for d in "${MONITORING_HOME}/grafana-${GRAFANA_VERSION}" "${MONITORING_HOME}/grafana-v${GRAFANA_VERSION}"; do
            [ -x "${d}/bin/grafana" ] && GRAFANA_HOME="${d}" && break
        done
        [ -n "${GRAFANA_HOME}" ] || {
            echo "Unexpected grafana tarball layout under ${MONITORING_HOME}"; exit 1; }
    fi
    nohup "${GRAFANA_HOME}/bin/grafana" server \
        --homepath "${GRAFANA_HOME}" \
        --config "${RAY_SESSION}/metrics/grafana/grafana.ini" \
        >"${GRAFANA_LOG}" 2>&1 &
    echo $! > "${GRAFANA_PID_FILE}"
    for _ in $(seq 60); do
        curl -sf http://localhost:3000/api/health >/dev/null 2>&1 && break
        sleep 1
    done
    curl -sf http://localhost:3000/api/health >/dev/null || {
        echo "Grafana did not come up on :3000 -- see ${GRAFANA_LOG}"; exit 1; }
    echo "Grafana up on :3000 (pid $(cat "${GRAFANA_PID_FILE}"), log ${GRAFANA_LOG})"
fi

echo
echo "Ray dashboard: http://localhost:8265"
echo "Prometheus:    http://localhost:9090"
echo "Grafana:       http://localhost:3000/d/verl-gpu-phase"
