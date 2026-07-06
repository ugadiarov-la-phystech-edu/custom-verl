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
# After this script:
#   Ray dashboard  http://<node>:8265   (live per-GPU util, actor/task view)
#   Prometheus     http://<node>:9090   (raw metrics + history)
#   Grafana        http://<node>:3000   (dashboard "verl · per-GPU phase & utilization")
set -euo pipefail

RAY_SESSION=/tmp/ray/session_latest
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
    exit 1
fi

# --- 1. Prometheus -------------------------------------------------------------------
# Uses the config Ray generated for this session: scrapes every node in the cluster via
# /tmp/ray/prom_metrics_service_discovery.json every 10s. Default retention is 15d, so
# "what was happening an hour ago" is a Grafana time-range change away.
# A binary already present under MONITORING_HOME (from a previous run, or hand-installed
# on an air-gapped node) is started directly; otherwise Ray's launcher downloads one.
# NB: `ray metrics launch-prometheus` re-downloads unconditionally, hence the local-first
# check; it also downloads into the CWD, so it is run from MONITORING_HOME.
PROM_PID_FILE="${MONITORING_HOME}/prometheus.pid"
PROM_LOG="${MONITORING_HOME}/prometheus.log"
if curl -sf http://localhost:9090/-/ready >/dev/null 2>&1; then
    echo "Prometheus already running on :9090"
else
    PROM_BIN="$(ls -d "${MONITORING_HOME}"/prometheus-*/prometheus 2>/dev/null | sort -V | tail -1)"
    if [ -n "${PROM_BIN}" ] && [ -x "${PROM_BIN}" ]; then
        # Same data dir the Ray launcher uses (its CWD default), so history is continuous.
        nohup "${PROM_BIN}" \
            --config.file /tmp/ray/session_latest/metrics/prometheus/prometheus.yml \
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
cp "${DASHBOARD_SRC}" "${RAY_SESSION}/metrics/grafana/dashboards/"
echo "Provisioned verl_gpu_phase_dashboard.json"

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
        --config /tmp/ray/session_latest/metrics/grafana/grafana.ini \
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
