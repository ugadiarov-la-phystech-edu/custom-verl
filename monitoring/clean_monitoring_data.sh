#!/usr/bin/env bash
set -euo pipefail

MONITORING_HOME="${VERL_MONITORING_HOME:-${HOME}/.verl-monitoring}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSUME_YES="${VERL_CLEAN_YES:-0}"
case "${1:-}" in -y|--yes) ASSUME_YES=1 ;; esac

if [ ! -d "${MONITORING_HOME}" ]; then
    echo "Nothing to clean: ${MONITORING_HOME} does not exist."
    exit 0
fi

if [ "${ASSUME_YES}" != "1" ]; then
    echo "About to delete ALL metric history under ${MONITORING_HOME}/data."
    runs="$(ls -1 "${MONITORING_HOME}/grafana/dashboards"/vgp-*.json 2>/dev/null | wc -l)"
    if [ "${runs}" -gt 0 ]; then
        echo "There are ${runs} per-experiment dashboard(s); they will render blank afterwards:"
        ls -1 "${MONITORING_HOME}/grafana/dashboards"/vgp-*.json | sed 's|.*/|  |'
    fi
    if [ ! -t 0 ]; then
        echo "Refusing to clean non-interactively without -y/--yes."; exit 1
    fi
    printf 'Type "yes" to continue: '
    read -r reply
    [ "${reply}" = "yes" ] || { echo "Aborted."; exit 1; }
fi

bash "${SCRIPT_DIR}/stop_monitoring.sh"

rm -rf "${MONITORING_HOME}/data"          # Prometheus TSDB (all metric history)
rm -rf "${MONITORING_HOME}"/grafana-*/data  # Grafana state: sqlite db, cache, UI edits
rm -f "${MONITORING_HOME}"/*.log "${MONITORING_HOME}"/*.pid

echo "Cleaned monitoring data under ${MONITORING_HOME} (binaries kept):"
ls "${MONITORING_HOME}"
