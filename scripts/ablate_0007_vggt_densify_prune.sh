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

TAG_DG="${DENSIFY_GRAD_THRESHOLD//./p}"
TAG_PR="${PRUNE_OPACITY_THRESHOLD//./p}"
EXP_TAG="dg${TAG_DG}_pr${TAG_PR}"
MODEL_PATH="/media/image/mxz/human/SeqAvatar/output/ablation_vggt_0007/${EXP_TAG}"
LOG_DIR="/media/image/mxz/human/SeqAvatar/logs/ablation_vggt_0007"
LOG_PATH="${LOG_DIR}/${EXP_TAG}.log"

mkdir -p "${MODEL_PATH}" "${LOG_DIR}"

{
  echo "========================================================="
  echo "Ablation start: ${EXP_TAG}"
  echo "GPU=${GPU_ID} ITERATIONS=${ITERATIONS}"
  echo "densify_grad_threshold=${DENSIFY_GRAD_THRESHOLD}"
  echo "prune_opacity_threshold=${PRUNE_OPACITY_THRESHOLD}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "========================================================="

  export WANDB_PROJECT="SeqAvatar_DNA_Rendering"
  export WANDB_NAME="ablation_${EXP_TAG}_train"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
    -s "${DATASET_PATH}" \
    --eval \
    --model_path "${MODEL_PATH}" \
    --exp_name "ablation_vggt_0007/${EXP_TAG}" \
    --motion_offset_flag \
    --smpl_type smplx \
    --actor_gender neutral \
    --iterations "${ITERATIONS}" \
    --densify_until_iter 1500 \
    --densify_grad_threshold "${DENSIFY_GRAD_THRESHOLD}" \
    --prune_opacity_threshold "${PRUNE_OPACITY_THRESHOLD}" \
    --seq_len 8 \
    --seq_xyz_knn 8 \
    --time_step_num 3 \
    --max_time_step 3 \
    --minimal_time_step 1 \
    --l1_loss_w 1.0 \
    --ssim_loss_w 0.01 \
    --lpips_loss_w 0.01 \
    --test_iterations "${ITERATIONS}" \
    --save_iterations "${ITERATIONS}"

  export WANDB_NAME="ablation_${EXP_TAG}_eval"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" render.py \
    -s "${DATASET_PATH}" \
    -m "${MODEL_PATH}" \
    --motion_offset_flag \
    --smpl_type smplx \
    --actor_gender neutral \
    --iteration "${ITERATIONS}" \
    --skip_train \
    --seq_len 8 \
    --seq_xyz_knn 8 \
    --time_step_num 3 \
    --max_time_step 3 \
    --minimal_time_step 1

  echo "Ablation done: ${EXP_TAG}"
} 2>&1 | tee "${LOG_PATH}"
