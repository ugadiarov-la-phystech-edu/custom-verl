#!/usr/bin/env bash
set -euo pipefail

MONITORING_HOME="${VERL_MONITORING_HOME:-${HOME}/.verl-monitoring}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "${MONITORING_HOME}" ]; then
    echo "Nothing to clean: ${MONITORING_HOME} does not exist."
    exit 0
fi

bash "${SCRIPT_DIR}/stop_monitoring.sh"

rm -rf "${MONITORING_HOME}/data"          # Prometheus TSDB (all metric history)
rm -rf "${MONITORING_HOME}"/grafana-*/data  # Grafana state: sqlite db, cache, UI edits
rm -f "${MONITORING_HOME}"/*.log "${MONITORING_HOME}"/*.pid

echo "Cleaned monitoring data under ${MONITORING_HOME} (binaries kept):"
ls "${MONITORING_HOME}"
