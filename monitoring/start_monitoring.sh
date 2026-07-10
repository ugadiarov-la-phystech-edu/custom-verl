#!/usr/bin/env bash
set -euo pipefail

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

if [ -e "${SD_FILE}" ] && [ "${SD_FILE}" -ot "${RAY_SESSION}/metrics" ]; then
    echo "WARNING: ${SD_FILE} predates the current Ray session."
    echo "  Prometheus is scraping ports from an older session; expect missing ray_node_* metrics."
    echo "  Check for write errors: grep -i permission ${RAY_SESSION}/logs/dashboard_ReportHead.log"
fi

PROM_PID_FILE="${MONITORING_HOME}/prometheus.pid"
PROM_LOG="${MONITORING_HOME}/prometheus.log"
if curl -sf http://localhost:9090/-/ready >/dev/null 2>&1; then
    echo "Prometheus already running on :9090"
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

DASHBOARD_DST="${DASHBOARD_DIR}/$(basename "${DASHBOARD_SRC}")"
RECOLOR="${SCRIPT_DIR}/grafana/apply_phase_colors.py"
if command -v python3 >/dev/null 2>&1 && [ -f "${RECOLOR}" ]; then
    python3 "${RECOLOR}" "${DASHBOARD_SRC}" "${DASHBOARD_DST}"
else
    cp "${DASHBOARD_SRC}" "${DASHBOARD_DST}"
fi
echo "Provisioned $(basename "${DASHBOARD_SRC}")"

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
