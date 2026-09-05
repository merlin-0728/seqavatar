#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLDER_SCRIPT="${SCRIPT_DIR}/train.py"

GPU_IDS="${GPU_IDS:-0,1}"
THRESHOLD_MIB="${THRESHOLD_MIB:-100}"
TARGET_MIB="${TARGET_MIB:-10000}"
BLOCK_MIB="${BLOCK_MIB:-256}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"
PID_DIR="${PID_DIR:-/tmp/seqavatar_gpu_reserve_${USER}}"
DEFAULT_PYTHON="/media/coding/ckx/.conda/envs/seqavatar/bin/python"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_PYTHON}"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

mkdir -p "${PID_DIR}"

is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1

  local pid
  pid="$(<"${pid_file}")"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

gpu_used_mib() {
  local gpu_id="$1"
  nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits |
    tr -dc '0-9'
}

while true; do
  IFS=',' read -ra ids <<< "${GPU_IDS}"

  for gpu_id in "${ids[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    [[ -n "${gpu_id}" ]] || continue

    pid_file="${PID_DIR}/gpu_${gpu_id}.pid"
    if is_running "${pid_file}"; then
      continue
    fi
    rm -f "${pid_file}"

    used_mib="$(gpu_used_mib "${gpu_id}" 2>/dev/null || true)"
    if [[ ! "${used_mib}" =~ ^[0-9]+$ ]]; then
      continue
    fi

    if (( used_mib < THRESHOLD_MIB )); then
      CUDA_VISIBLE_DEVICES="${gpu_id}" nohup "${PYTHON_BIN}" "${HOLDER_SCRIPT}" \
        --target-mib "${TARGET_MIB}" \
        --block-mib "${BLOCK_MIB}" \
        >/dev/null 2>&1 &
      echo "$!" > "${pid_file}"
      sleep 5

      if ! is_running "${pid_file}"; then
        rm -f "${pid_file}"
      fi
    fi
  done

  sleep "${CHECK_INTERVAL_SEC}"
done
