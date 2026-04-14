#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/media/image/mxz/human/SeqAvatar"
cd "${ROOT_DIR}"
GPU_id="${1:-2}"
SEQUENCES=(ID1_1 ID1_2 ID2_1 ID3_1)

data_path="${ROOT_DIR}/I3D-Human"
iter=15000
densify_until_iter=1800

seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=42
minimal_time_step=24

l1_loss_w=1.0
ssim_loss_w=0.1
lpips_loss_w=0.1
lambda_vggt="${LAMBDA_VGGT:-0.2}"
phase1_vggt_iters="${PHASE1_VGGT_ITERS:-0}"
frame_index="${FRAME_INDEX:-301}"
split="${I3D_SPLIT:-train}"
run_extract="${RUN_EXTRACT:-1}"
run_vggt_init="${RUN_VGGT_INIT:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY_BIN="${PYTHON_BIN}"
elif [[ -x "/media/image/mxz/.conda/envs/seqavatar/bin/python" ]]; then
    PY_BIN="/media/image/mxz/.conda/envs/seqavatar/bin/python"
elif command -v python >/dev/null 2>&1; then
    PY_BIN="python"
else
    PY_BIN="python3"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-SeqAvatar_VGGT_I3D_Human}"

for SEQUENCE in "${SEQUENCES[@]}"; do
    frame6="$(printf "%06d" "${frame_index}")"
    dataset_path="${data_path}/${SEQUENCE}-${split}/"
    exp_name="VGGT-Init/I3D-Human/${SEQUENCE}/frame_${frame6}/"
    model_path="output/${exp_name}"
    mkdir -p "${model_path}/logs"

    if [[ "${run_extract}" == "1" ]]; then
        /bin/bash "${ROOT_DIR}/scripts/extracted.sh" I3D "${SEQUENCE}" "${frame_index}" "${split}"
    fi
    if [[ "${run_vggt_init}" == "1" ]]; then
        /bin/bash "${ROOT_DIR}/scripts/vggt.sh" I3D "${SEQUENCE}" "${frame_index}" "${split}"
    fi

    vggt_init_path="${dataset_path}/OUTPUT_PT/vggt_canonical_init_frame${frame6}.pt"
    if [[ ! -f "${vggt_init_path}" ]]; then
        echo "Error: VGGT init PT not found: ${vggt_init_path}"
        exit 1
    fi

    export WANDB_NAME="train_vggt_${SEQUENCE}_f${frame6}"
    echo "Training on GPU ${GPU_id} for sequence ${SEQUENCE} (frame ${frame6})"
    CUDA_VISIBLE_DEVICES="${GPU_id}" "${PY_BIN}" train.py -s "${dataset_path}" --eval --exp_name "${exp_name}" \
        --motion_offset_flag --smpl_type smpl --actor_gender neutral \
        --iterations "${iter}" --densify_until_iter "${densify_until_iter}" \
        --seq_len "${seq_len}" --seq_xyz_knn "${seq_xyz_knn}" \
        --time_step_num "${time_step_num}" --max_time_step "${max_time_step}" --minimal_time_step "${minimal_time_step}" \
        --l1_loss_w "${l1_loss_w}" --ssim_loss_w "${ssim_loss_w}" --lpips_loss_w "${lpips_loss_w}" \
        --vggt_init_path "${vggt_init_path}" --lambda_vggt "${lambda_vggt}" --phase1_vggt_iters "${phase1_vggt_iters}" \
        2>&1 | tee "${model_path}/logs/train_${SEQUENCE}_f${frame6}.log"

    export WANDB_NAME="eval_vggt_${SEQUENCE}_f${frame6}"
    echo "Evaluating on GPU ${GPU_id} for sequence ${SEQUENCE} (frame ${frame6})"
    CUDA_VISIBLE_DEVICES="${GPU_id}" "${PY_BIN}" render.py -s "${dataset_path}" -m "${model_path}" \
        --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration "${iter}" --skip_train \
        --seq_len "${seq_len}" --seq_xyz_knn "${seq_xyz_knn}" \
        --time_step_num "${time_step_num}" --max_time_step "${max_time_step}" --minimal_time_step "${minimal_time_step}" \
        2>&1 | tee "${model_path}/logs/render_${SEQUENCE}_f${frame6}.log"
done
