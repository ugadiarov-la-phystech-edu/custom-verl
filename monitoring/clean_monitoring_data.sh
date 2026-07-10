#!/usr/bin/env bash
# Wipe all monitoring state (Prometheus TSDB history, Grafana sqlite db, logs, pidfiles)
# while keeping the downloaded Prometheus/Grafana binaries, so the next
# start_monitoring.sh starts clean without re-downloading anything.
#
# Stops both processes first: deleting the TSDB under a live Prometheus corrupts it.
# Note: this also discards any dashboard edits made through the Grafana UI -- the
# repo's grafana/verl_gpu_phase_dashboard.json is re-provisioned on next start.
#
# DESTRUCTIVE. The TSDB is the only copy of every finished experiment's metrics: per-run
# dashboards survive as files but render blank forever afterwards. Never run this between
# experiments you intend to compare. Pass -y/--yes (or VERL_CLEAN_YES=1) to skip the prompt.
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
