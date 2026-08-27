#!/usr/bin/env bash
set -euo pipefail

PID_DIR="${PID_DIR:-/tmp/seqavatar_gpu_reserve_${USER}}"
MONITOR_PID_FILE="${PID_DIR}/monitor.pid"

kill_pid_file() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 0

  local pid
  pid="$(<"${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
  fi
  rm -f "${pid_file}"
}

kill_pid_file "${MONITOR_PID_FILE}"

if [[ -d "${PID_DIR}" ]]; then
  for pid_file in "${PID_DIR}"/gpu_*.pid; do
    [[ -e "${pid_file}" ]] || continue
    kill_pid_file "${pid_file}"
  done
fi
