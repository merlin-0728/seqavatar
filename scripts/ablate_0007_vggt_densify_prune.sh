#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <gpu_id> <densify_grad_threshold> <prune_opacity_threshold> [iterations]"
  exit 1
fi

GPU_ID="$1"
DENSIFY_GRAD_THRESHOLD="$2"
PRUNE_OPACITY_THRESHOLD="$3"
ITERATIONS="${4:-25000}"

PYTHON_BIN="/media/image/mxz/.conda/envs/seqavatar/bin/python"
export PATH="/media/image/mxz/.conda/envs/seqavatar/bin:${PATH}"
SEQ="0007_04"
DATASET_PATH="/media/image/mxz/human/SeqAvatar/DNA-Rendering/${SEQ}/"
VGGT_INIT_PATH="${DATASET_PATH}OUTPUT_PT/vggt_canonical_init.pt"

TAG_DG="${DENSIFY_GRAD_THRESHOLD//./p}"
TAG_PR="${PRUNE_OPACITY_THRESHOLD//./p}"
EXP_TAG="dg${TAG_DG}_pr${TAG_PR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/media/image/mxz/human/SeqAvatar/output/ablation_vggt_0007/${RUN_TS}"
LOG_DIR="/media/image/mxz/human/SeqAvatar/logs/ablation_vggt_0007"
LOG_PATH="${LOG_DIR}/${RUN_TS}.log"

mkdir -p "${MODEL_PATH}" "${LOG_DIR}"

{
  echo "========================================================="
  echo "Ablation start: ${EXP_TAG}"
  echo "RUN_TS=${RUN_TS}"
  echo "GPU=${GPU_ID} ITERATIONS=${ITERATIONS}"
  echo "densify_grad_threshold=${DENSIFY_GRAD_THRESHOLD}"
  echo "prune_opacity_threshold=${PRUNE_OPACITY_THRESHOLD}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "LOG_PATH=${LOG_PATH}"
  echo "========================================================="

  export WANDB_PROJECT="SeqAvatar_DNA_Rendering"
  export WANDB_NAME="ablation_${EXP_TAG}_train"

  TRAIN_ARGS=(
    -s "${DATASET_PATH}"
    --eval
    --model_path "${MODEL_PATH}"
    --exp_name "ablation_vggt_0007/${RUN_TS}"
    --motion_offset_flag
    --smpl_type smplx
    --actor_gender neutral
    --iterations "${ITERATIONS}"
    --densify_until_iter 1500
    --densify_grad_threshold "${DENSIFY_GRAD_THRESHOLD}"
    --prune_opacity_threshold "${PRUNE_OPACITY_THRESHOLD}"
    --vggt_init_path "${VGGT_INIT_PATH}"
    --vggt_max_points 60000
    --vggt_jitter_std 0.0
    --vggt_feat_dim 1
    --seq_len 8
    --seq_xyz_knn 8
    --time_step_num 3
    --max_time_step 3
    --minimal_time_step 1
    --l1_loss_w 1.0
    --ssim_loss_w 0.01
    --lpips_loss_w 0.01
    --test_iterations "${ITERATIONS}"
    --save_iterations "${ITERATIONS}"
  )

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
    "${TRAIN_ARGS[@]}"

  export WANDB_NAME="ablation_${EXP_TAG}_eval"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" render.py \
    -s "${DATASET_PATH}" \
    -m "${MODEL_PATH}" \
    --motion_offset_flag \
    --smpl_type smplx \
    --actor_gender neutral \
    --iteration "${ITERATIONS}" \
    --skip_train \
    --vggt_init_path "${VGGT_INIT_PATH}" \
    --vggt_max_points 60000 \
    --vggt_jitter_std 0.0 \
    --vggt_feat_dim 1 \
    --seq_len 8 \
    --seq_xyz_knn 8 \
    --time_step_num 3 \
    --max_time_step 3 \
    --minimal_time_step 1

  RENDER_DIR="${MODEL_PATH}/novelview/ours_${ITERATIONS}/renders"
  GT_DIR="${MODEL_PATH}/novelview/ours_${ITERATIONS}/gt"
  RENDER_COUNT="$(find "${RENDER_DIR}" -type f -name '*.png' | wc -l || true)"
  GT_COUNT="$(find "${GT_DIR}" -type f -name '*.png' | wc -l || true)"
  echo "Render artifacts: renders=${RENDER_COUNT} gts=${GT_COUNT}"
  if [ "${RENDER_COUNT}" -lt 120 ] || [ "${GT_COUNT}" -lt 120 ]; then
    echo "Render artifact check failed: expected >=120 images for renders and gt."
    exit 2
  fi

  echo "Ablation done: ${EXP_TAG} (run=${RUN_TS})"
} 2>&1 | tee "${LOG_PATH}"
