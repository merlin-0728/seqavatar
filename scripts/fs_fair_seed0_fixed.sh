#!/usr/bin/env bash
set -euo pipefail

# Strict controlled experiment:
# - Same code version
# - Same script
# - Same params
# - Same seed (fixed)
# - Only switch: --depth_prior_enable
# - Fixed evaluation: parse PSNR/SSIM/LPIPS/FPS from render.py novelview output

ROOT_DIR="/media/image/mxz/human/SeqAvatar"
PYTHON_BIN="${PYTHON_BIN:-/media/image/mxz/.conda/envs/seqavatar/bin/python}"
PYTHON_BIN_DIR="$(dirname "${PYTHON_BIN}")"
export PATH="${PYTHON_BIN_DIR}:${PATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"

DATA_ROOT="${DATA_ROOT:-/media/image/mxz/human/SeqAvatar/DNA-Rendering}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/image/mxz/human/SeqAvatar/output/fs_fair_seed0_fixed}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TS}"
LOG_DIR="${RUN_ROOT}/logs"
ROW_DIR="${RUN_ROOT}/rows"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_DELTA_CSV="${RUN_ROOT}/summary_delta.csv"

mkdir -p "${LOG_DIR}" "${ROW_DIR}"

SEQUENCES=(${SEQUENCES:-0007_04 0019_10 0044_11})
SEED="${SEED:-0}"
ITERATIONS="${ITERATIONS:-25000}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-1500}"

# Keep per-sequence fairness by running baseline/fs on the same GPU.
# Use two GPUs overall for throughput.
GPU_SEQ_0007="${GPU_SEQ_0007:-2}"
GPU_SEQ_0019="${GPU_SEQ_0019:-3}"
GPU_SEQ_0044="${GPU_SEQ_0044:-2}"

common_train_args=(
  --eval
  --motion_offset_flag --smpl_type smplx --actor_gender neutral
  --iterations "${ITERATIONS}" --densify_until_iter "${DENSIFY_UNTIL_ITER}"
  --seq_len 8 --seq_xyz_knn 8
  --time_step_num 3 --max_time_step 3 --minimal_time_step 1
  --l1_loss_w 1.0 --ssim_loss_w 0.01 --lpips_loss_w 0.01
  --seed "${SEED}"
)

common_render_args=(
  --motion_offset_flag --smpl_type smplx --actor_gender neutral
  --iteration "${ITERATIONS}" --skip_train
  --seq_len 8 --seq_xyz_knn 8
  --time_step_num 3 --max_time_step 3 --minimal_time_step 1
  --data_device cuda
)

run_one() {
  local seq="$1"
  local mode="$2"   # baseline | fs
  local gpu="$3"

  local data_dir="${DATA_ROOT}/${seq}"
  local depth_dir="${data_dir}/fs_depth"
  local exp_name="fs_fair_seed0_fixed/${RUN_TS}/${seq}/${mode}"
  local model_dir="${ROOT_DIR}/output/${exp_name}"
  local train_log="${LOG_DIR}/${seq}_${mode}_train.log"
  local render_log="${LOG_DIR}/${seq}_${mode}_render.log"
  local row_file="${ROW_DIR}/${seq}_${mode}.csv"

  if [[ ! -d "${data_dir}" ]]; then
    echo "[ERR] missing data dir: ${data_dir}" | tee -a "${RUN_ROOT}/main.log"
    return 1
  fi
  if [[ ! -d "${depth_dir}" ]]; then
    echo "[ERR] missing fs depth dir: ${depth_dir}" | tee -a "${RUN_ROOT}/main.log"
    return 1
  fi

  mkdir -p "${model_dir}/logs"
  echo "[RUN] seq=${seq} mode=${mode} gpu=${gpu} seed=${SEED}" | tee -a "${RUN_ROOT}/main.log"

  local -a depth_args
  # Keep args identical in both branches; only add --depth_prior_enable for FS.
  depth_args=(--depth_prior_dir "${depth_dir}")
  if [[ "${mode}" == "fs" ]]; then
    depth_args+=(--depth_prior_enable)
  fi

  (
    cd "${ROOT_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" train.py \
      -s "${data_dir}" --exp_name "${exp_name}" \
      "${common_train_args[@]}" \
      "${depth_args[@]}" \
      2>&1 | tee "${train_log}"
  )

  (
    cd "${ROOT_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" render.py \
      -s "${data_dir}" -m "${model_dir}" \
      "${common_render_args[@]}" \
      2>&1 | tee "${render_log}"
  )

  "${PYTHON_BIN}" - <<PY > "${row_file}"
import re
seq = "${seq}"
mode = "${mode}"
gpu = "${gpu}"
seed = "${SEED}"
model_dir = "${model_dir}"
render_log = "${render_log}"

fps = None
psnr = None
ssim = None
lpips = None

with open(render_log, "r", encoding="utf-8", errors="ignore") as f:
    txt = f.read()

fps_matches = re.findall(r"FPS:\\s*([0-9eE+\\-.]+)", txt)
if fps_matches:
    fps = float(fps_matches[-1])

metric_matches = re.findall(
    r"Evaluating\\s+novelview\\s+#\\d+:\\s+PSNR\\s+([0-9eE+\\-.]+)\\s+SSIM\\s+([0-9eE+\\-.]+)\\s+LPIPS\\s+([0-9eE+\\-.]+)",
    txt
)
if metric_matches:
    m = metric_matches[-1]
    psnr = float(m[0]); ssim = float(m[1]); lpips = float(m[2])

if any(v is None for v in [psnr, ssim, lpips, fps]):
    raise RuntimeError(f"failed to parse metrics/fps from {render_log}")

print("sequence,mode,seed,gpu,psnr,ssim,lpips,fps,model_path")
print(f"{seq},{mode},{seed},{gpu},{psnr},{ssim},{lpips},{fps},{model_dir}")
PY

  echo "[OK] seq=${seq} mode=${mode} done" | tee -a "${RUN_ROOT}/main.log"
}

run_seq_pair() {
  local seq="$1"
  local gpu="$2"
  run_one "${seq}" "baseline" "${gpu}"
  run_one "${seq}" "fs" "${gpu}"
}

echo "sequence,mode,seed,gpu,psnr,ssim,lpips,fps,model_path" > "${SUMMARY_CSV}"

# Two workers: keep pair-level fairness and use both GPU 2/3.
(
  run_seq_pair "0007_04" "${GPU_SEQ_0007}"
  run_seq_pair "0044_11" "${GPU_SEQ_0044}"
) &
pid_a=$!

(
  run_seq_pair "0019_10" "${GPU_SEQ_0019}"
) &
pid_b=$!

wait "${pid_a}"
wait "${pid_b}"

for seq in "${SEQUENCES[@]}"; do
  for mode in baseline fs; do
    row_file="${ROW_DIR}/${seq}_${mode}.csv"
    if [[ ! -f "${row_file}" ]]; then
      echo "[ERR] missing row file: ${row_file}" | tee -a "${RUN_ROOT}/main.log"
      exit 2
    fi
    tail -n 1 "${row_file}" >> "${SUMMARY_CSV}"
  done
done

"${PYTHON_BIN}" - <<PY > "${SUMMARY_DELTA_CSV}"
import csv
from collections import defaultdict

summary_csv = "${SUMMARY_CSV}"
rows = list(csv.DictReader(open(summary_csv, "r", encoding="utf-8")))
by_seq = defaultdict(dict)
for r in rows:
    by_seq[r["sequence"]][r["mode"]] = r

print("sequence,baseline_psnr,fs_psnr,delta_psnr,baseline_ssim,fs_ssim,delta_ssim,baseline_lpips,fs_lpips,delta_lpips,baseline_fps,fs_fps,delta_fps")
for seq in ["0007_04", "0019_10", "0044_11"]:
    b = by_seq[seq]["baseline"]
    f = by_seq[seq]["fs"]
    bps, fpsnr = float(b["psnr"]), float(f["psnr"])
    bss, fss = float(b["ssim"]), float(f["ssim"])
    blp, flp = float(b["lpips"]), float(f["lpips"])
    bfp, ffp = float(b["fps"]), float(f["fps"])
    print(
        f"{seq},"
        f"{bps},{fpsnr},{fpsnr-bps},"
        f"{bss},{fss},{fss-bss},"
        f"{blp},{flp},{flp-blp},"
        f"{bfp},{ffp},{ffp-bfp}"
    )
PY

echo "[DONE] run_root=${RUN_ROOT}" | tee -a "${RUN_ROOT}/main.log"
echo "[DONE] summary=${SUMMARY_CSV}" | tee -a "${RUN_ROOT}/main.log"
echo "[DONE] delta=${SUMMARY_DELTA_CSV}" | tee -a "${RUN_ROOT}/main.log"

