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

STABLE_PROM_CFG="${MONITORING_HOME}/prometheus/prometheus.yml"
STABLE_GRAFANA_INI="${MONITORING_HOME}/grafana/grafana.ini"
STABLE_PROVISIONING="${MONITORING_HOME}/grafana/provisioning"
DASHBOARD_DIR="${MONITORING_HOME}/grafana/dashboards"
PROM_RETENTION="${VERL_PROM_RETENTION:-90d}"
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

suggest_tmpdir() {
    echo
    echo "Fix: give your Ray a temp root nobody else writes to, then restart Ray and this script:"
    echo "    export RAY_TMPDIR=/tmp/ray-\$(id -un)      # before starting Ray, and before this script"
    echo "  (Ray appends '/ray', so the root becomes /tmp/ray-\$(id -un)/ray.)"
}

preflight_live() {
    local f
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

    if [ -e "${SD_FILE}" ] && [ "${SD_FILE}" -ot "${RAY_SESSION}/metrics" ]; then
        echo "WARNING: ${SD_FILE} predates the current Ray session."
        echo "  Prometheus is scraping ports from an older session; expect missing ray_node_* metrics."
        echo "  Check for write errors: grep -i permission ${RAY_SESSION}/logs/dashboard_ReportHead.log"
    fi
}

mirror_ray_config() {
    local session
    session="$(readlink -f "${RAY_SESSION}")"
    mkdir -p "${MONITORING_HOME}/prometheus" "${MONITORING_HOME}/grafana" "${DASHBOARD_DIR}"
    cp "${session}/metrics/prometheus/prometheus.yml" "${STABLE_PROM_CFG}"
    cp "${session}/metrics/grafana/grafana.ini" "${STABLE_GRAFANA_INI}"
    rm -rf "${STABLE_PROVISIONING}"
    cp -rL "${session}/metrics/grafana/provisioning" "${STABLE_PROVISIONING}"
    chmod -R u+w "${MONITORING_HOME}/grafana" "${MONITORING_HOME}/prometheus"

    cp "${session}"/metrics/grafana/dashboards/*.json "${DASHBOARD_DIR}/" 2>/dev/null || true

    sed -i "s#^provisioning = .*#provisioning = ${STABLE_PROVISIONING}#" "${STABLE_GRAFANA_INI}"
    find "${STABLE_PROVISIONING}/dashboards" -name '*.yml' -exec \
        sed -i "s#^\( *path:\).*#\1 ${DASHBOARD_DIR}#" {} +

    if grep -rn "session_" "${STABLE_GRAFANA_INI}" "${STABLE_PROVISIONING}" "${STABLE_PROM_CFG}"; then
        echo "ERROR: a Ray session path survived the rewrite (above). Ray's config layout changed."
        exit 1
    fi
    echo "Mirrored Ray's monitoring config into ${MONITORING_HOME}"
}

if [ -e "${RAY_SESSION}/metrics" ]; then
    preflight_live
    mirror_ray_config                       # idempotent: refresh on every start
elif [ -f "${STABLE_GRAFANA_INI}" ] && [ -f "${STABLE_PROM_CFG}" ]; then
    echo "No live Ray session -- starting from the saved config (post-hoc browsing)."
    mkdir -p "${DASHBOARD_DIR}"
else
    echo "No live Ray session at ${RAY_SESSION}, and no saved config under ${MONITORING_HOME}."
    echo "The first run needs Ray alive once to generate the monitoring config:"
    echo "  start the training run (or 'ray start --head --dashboard-host=0.0.0.0'), then re-run this."
    if [ "${RAY_ROOT}" != "/tmp/ray" ]; then
        echo "(resolved Ray root: ${RAY_ROOT} -- was Ray started with the same RAY_TMPDIR?)"
    fi
    exit 1
fi

PROM_PID_FILE="${MONITORING_HOME}/prometheus.pid"
PROM_LOG="${MONITORING_HOME}/prometheus.log"
if curl -sf http://localhost:9090/-/ready >/dev/null 2>&1; then
    echo "Prometheus already running on :9090"
    PROM_CFG="$(curl -sf http://localhost:9090/api/v1/status/flags 2>/dev/null \
        | sed -n 's/.*"config\.file":"\([^"]*\)".*/\1/p')"
    if [ -n "${PROM_CFG}" ] && [ "${PROM_CFG}" != "${STABLE_PROM_CFG}" ]; then
        echo "WARNING: it was started with --config.file ${PROM_CFG},"
        echo "  not the mirrored ${STABLE_PROM_CFG}."
        echo "  It may be scraping another Ray's targets, and its retention is not ${PROM_RETENTION}."
        echo "  Restart it: bash $(dirname "${BASH_SOURCE[0]}")/stop_monitoring.sh && $0"
    fi
else
    PROM_BIN="$(ls -d "${MONITORING_HOME}"/prometheus-*/prometheus 2>/dev/null | sort -V | tail -1 || true)"
    if [ -n "${PROM_BIN}" ] && [ -x "${PROM_BIN}" ]; then
        nohup "${PROM_BIN}" \
            --config.file "${STABLE_PROM_CFG}" \
            --storage.tsdb.path "${MONITORING_HOME}/data" \
            --storage.tsdb.retention.time "${PROM_RETENTION}" \
            --web.enable-lifecycle \
            >"${PROM_LOG}" 2>&1 &
        echo $! > "${PROM_PID_FILE}"
    else
        echo "No prometheus binary under ${MONITORING_HOME}; falling back to Ray's launcher"
        echo "  (it ignores --storage.tsdb.retention.time ${PROM_RETENTION} and needs a live Ray session)."
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
        --config "${STABLE_GRAFANA_INI}" \
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
