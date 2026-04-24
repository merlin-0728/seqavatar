#!/usr/bin/env bash
set -euo pipefail

RUN_TS="${RUN_TS:-20260423_fair_opt2_waitrun}"
LOG="/media/image/mxz/human/SeqAvatar/logs/fs_fair_seed0_fixed_${RUN_TS}.wait.log"
ROOT="/media/image/mxz/human/SeqAvatar"

mkdir -p "${ROOT}/logs"

echo "[START] $(date +%F_%T) waiting for free GPU2/3" >> "${LOG}"
while true; do
  U2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')
  U3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
  echo "[WAIT] $(date +%F_%T) gpu2_used=${U2}MiB gpu3_used=${U3}MiB" >> "${LOG}"
  if [[ "${U2}" -lt 6000 && "${U3}" -lt 6000 ]]; then
    echo "[RUN] $(date +%F_%T) launching fair script on GPU2/3" >> "${LOG}"
    cd "${ROOT}"
    RUN_TS="${RUN_TS}" GPU_SEQ_0007=2 GPU_SEQ_0019=3 GPU_SEQ_0044=2 bash scripts/fs_fair_seed0_fixed.sh >> "${LOG}" 2>&1
    echo "[DONE] $(date +%F_%T) script finished" >> "${LOG}"
    exit 0
  fi
  sleep 300
done
