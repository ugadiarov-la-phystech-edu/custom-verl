#!/usr/bin/env bash
set -uo pipefail

MONITORING_HOME="${VERL_MONITORING_HOME:-${HOME}/.verl-monitoring}"
GRAFANA_PID_FILE="${MONITORING_HOME}/grafana.pid"
PROM_PID_FILE="${MONITORING_HOME}/prometheus.pid"

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
    pkill -f "${MONITORING_HOME}/grafana-.*/bin/grafana server" 2>/dev/null \
        && echo "Grafana stopped (matched by process name)" \
        || echo "Grafana was not running"
    rm -f "${GRAFANA_PID_FILE}"
fi
