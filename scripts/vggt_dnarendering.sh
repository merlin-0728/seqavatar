#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/media/image/mxz/human/SeqAvatar"
cd "${ROOT_DIR}"
GPU_id="${1:-2}"
DEFAULT_SEQUENCES=(0007_04 0019_10 0044_11 0051_09 0206_04 0813_05)
if [[ -n "${SEQUENCE_LIST:-}" ]]; then
    read -r -a SEQUENCES <<< "${SEQUENCE_LIST//,/ }"
else
    SEQUENCES=("${DEFAULT_SEQUENCES[@]}")
fi
CANONICAL_ROOT="${ROOT_DIR}/PT/DNA_Rendering"

data_path="${ROOT_DIR}/DNA-Rendering"
iter="${ITERATIONS:-25000}"
densify_until_iter="${DENSIFY_UNTIL_ITER:-1500}"
densify_grad_threshold="${DENSIFY_GRAD_THRESHOLD:-0.0002}"
prune_opacity_threshold="${PRUNE_OPACITY_THRESHOLD:-0.001}"
vggt_max_points="${VGGT_MAX_POINTS:-60000}"
data_device="${DATA_DEVICE:-cpu}"

seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=3
minimal_time_step=1

l1_loss_w=1.0
ssim_loss_w=0.01
lpips_loss_w=0.01
lambda_vggt="${LAMBDA_VGGT:-0.05}"
phase1_vggt_iters="${PHASE1_VGGT_ITERS:-5000}"
phase1_vggt_ramp_iters="${PHASE1_VGGT_RAMP_ITERS:-2000}"
vggt_loss_phase1_only="${VGGT_LOSS_PHASE1_ONLY:-1}"
run_extract="${RUN_EXTRACT:-0}"
run_vggt_init="${RUN_VGGT_INIT:-0}"

declare -A FRAME_BY_SEQ=(
    [0007_04]=25
    [0019_10]=7
    [0044_11]=34
    [0051_09]=131
    [0206_04]=10
    [0813_05]=56
)

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY_BIN="${PYTHON_BIN}"
elif [[ -x "/media/image/mxz/.conda/envs/seqavatar/bin/python" ]]; then
    PY_BIN="/media/image/mxz/.conda/envs/seqavatar/bin/python"
elif command -v python >/dev/null 2>&1; then
    PY_BIN="python"
else
    PY_BIN="python3"
fi
export PATH="$(dirname "${PY_BIN}"):${PATH}"

export WANDB_PROJECT="${WANDB_PROJECT:-SeqAvatar_VGGT_DNA_Rendering}"

for SEQUENCE in "${SEQUENCES[@]}"; do
    if [[ -z "${FRAME_BY_SEQ[$SEQUENCE]:-}" ]]; then
        echo "Error: unknown sequence '${SEQUENCE}'. Available: ${!FRAME_BY_SEQ[*]}"
        exit 1
    fi
    frame_index="${FRAME_BY_SEQ[$SEQUENCE]}"
    frame6="$(printf "%06d" "${frame_index}")"
    canonical_name="$(ls -1t "${CANONICAL_ROOT}"/canonical_*_DNA_"${SEQUENCE}".pt 2>/dev/null | head -n 1 | xargs -r basename)"
    if [[ -z "${canonical_name}" ]]; then
        echo "Error: no canonical PT found for ${SEQUENCE} under ${CANONICAL_ROOT}"
        exit 1
    fi
    canonical_tag="${canonical_name%.pt}"
    exp_name="VGGT-Init/DNA-Rendering/${SEQUENCE}/${canonical_tag}/"
    dataset_path="${data_path}/${SEQUENCE}/"
    model_path="output/${exp_name}"
    mkdir -p "${model_path}/logs"

    if [[ "${run_extract}" == "1" ]]; then
        /bin/bash "${ROOT_DIR}/scripts/extracted.sh" DNA "${SEQUENCE}" "${frame_index}"
    fi
    if [[ "${run_vggt_init}" == "1" ]]; then
        /bin/bash "${ROOT_DIR}/scripts/vggt.sh" DNA "${SEQUENCE}" "${frame_index}"
    fi

    vggt_init_path="${CANONICAL_ROOT}/${canonical_name}"
    if [[ ! -f "${vggt_init_path}" ]]; then
        echo "Error: VGGT init PT not found: ${vggt_init_path}"
        exit 1
    fi

    export WANDB_NAME="train_vggt_${SEQUENCE}_f${frame6}"
    echo "Training on GPU ${GPU_id} for sequence ${SEQUENCE} (frame ${frame6})"
    CUDA_VISIBLE_DEVICES="${GPU_id}" "${PY_BIN}" train.py -s "${dataset_path}" --eval --exp_name "${exp_name}" \
        --motion_offset_flag --smpl_type smplx --actor_gender neutral \
        --data_device "${data_device}" \
        --iterations "${iter}" --densify_until_iter "${densify_until_iter}" \
        --densify_grad_threshold "${densify_grad_threshold}" --prune_opacity_threshold "${prune_opacity_threshold}" \
        --seq_len "${seq_len}" --seq_xyz_knn "${seq_xyz_knn}" \
        --time_step_num "${time_step_num}" --max_time_step "${max_time_step}" --minimal_time_step "${minimal_time_step}" \
        --l1_loss_w "${l1_loss_w}" --ssim_loss_w "${ssim_loss_w}" --lpips_loss_w "${lpips_loss_w}" \
        --vggt_init_path "${vggt_init_path}" --vggt_max_points "${vggt_max_points}" \
        --lambda_vggt "${lambda_vggt}" --phase1_vggt_iters "${phase1_vggt_iters}" \
        --phase1_vggt_ramp_iters "${phase1_vggt_ramp_iters}" --vggt_loss_phase1_only "${vggt_loss_phase1_only}" \
        2>&1 | tee "${model_path}/logs/train_${SEQUENCE}_f${frame6}.log"

    export WANDB_NAME="eval_vggt_${SEQUENCE}_f${frame6}"
    echo "Evaluating on GPU ${GPU_id} for sequence ${SEQUENCE} (frame ${frame6})"
    CUDA_VISIBLE_DEVICES="${GPU_id}" "${PY_BIN}" render.py -s "${dataset_path}" -m "${model_path}" \
        --motion_offset_flag --smpl_type smplx --actor_gender neutral --iteration "${iter}" --skip_train \
        --data_device "${data_device}" \
        --seq_len "${seq_len}" --seq_xyz_knn "${seq_xyz_knn}" \
        --time_step_num "${time_step_num}" --max_time_step "${max_time_step}" --minimal_time_step "${minimal_time_step}" \
        2>&1 | tee "${model_path}/logs/render_${SEQUENCE}_f${frame6}.log"
done
