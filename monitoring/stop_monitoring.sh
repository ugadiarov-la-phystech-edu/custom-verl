#!/usr/bin/env bash
# Stop the Prometheus + Grafana started by start_monitoring.sh.
# Note: stopping Prometheus ends metric collection but keeps its on-disk history;
# restarting later resumes with the old data intact. Grafana state (sqlite db)
# likewise persists under its homepath.
set -uo pipefail

# Keep in step with start_monitoring.sh: `ray metrics shutdown-prometheus` resolves the
# running Prometheus through Ray's temp root, so it must see the same RAY_TMPDIR.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
MONITORING_HOME="${VERL_MONITORING_HOME:-${HOME}/.verl-monitoring}"
GRAFANA_PID_FILE="${MONITORING_HOME}/grafana.pid"
PROM_PID_FILE="${MONITORING_HOME}/prometheus.pid"

# Prometheus: started by us (pidfile) or by `ray metrics launch-prometheus`.
if [ -f "${PROM_PID_FILE}" ] && kill -0 "$(cat "${PROM_PID_FILE}")" 2>/dev/null; then
    kill "$(cat "${PROM_PID_FILE}")" && echo "Prometheus stopped (pid $(cat "${PROM_PID_FILE}"))"
    rm -f "${PROM_PID_FILE}"
else
    rm -f "${PROM_PID_FILE}"
    ray metrics shutdown-prometheus || true
fi

if [ -f "${GRAFANA_PID_FILE}" ] && kill -0 "$(cat "${GRAFANA_PID_FILE}")" 2>/dev/null; then
    kill "$(cat "${GRAFANA_PID_FILE}")" && echo "Grafana stopped (pid $(cat "${GRAFANA_PID_FILE}"))"
    rm -f "${GRAFANA_PID_FILE}"
else
    # Fallback: match the server process by its homepath under MONITORING_HOME.
    pkill -f "${MONITORING_HOME}/grafana-.*/bin/grafana server" 2>/dev/null \
        && echo "Grafana stopped (matched by process name)" \
        || echo "Grafana was not running"
    rm -f "${GRAFANA_PID_FILE}"
fi
