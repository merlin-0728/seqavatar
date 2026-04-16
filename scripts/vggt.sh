#!/usr/bin/env bash
set -euo pipefail

SEQAVATAR_ROOT="/media/image/mxz/human/SeqAvatar"
VGGT_ROOT="/media/image/mxz/human/VGGT/vggt-main"
EXTRACT_ROOT="${SEQAVATAR_ROOT}/extracted_frame"
SMPL_MODEL_DIR="${SEQAVATAR_ROOT}/smpl_model/models"
VGGT_PT_ROOT="/media/image/mxz/human/VGGT/vggt-main/OUTPUT_PT"
SEQ_PT_ROOT="${SEQAVATAR_ROOT}/PT"

EXTRACT_GARMENT_PY="${VGGT_ROOT}/extract_garment_prior.py"
BIND_CANONICAL_PY="${VGGT_ROOT}/bind_canonical.py"
SELECT_BIND_FRAME_PY="${VGGT_ROOT}/select_bind_frame.py"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY_BIN="${PYTHON_BIN}"
elif [[ -x "/media/image/mxz/.conda/envs/seqavatar/bin/python" ]]; then
    PY_BIN="/media/image/mxz/.conda/envs/seqavatar/bin/python"
elif command -v python >/dev/null 2>&1; then
    PY_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
else
    echo "Error: python not found."
    exit 1
fi

usage() {
    cat <<'EOF'
Usage:
  ./scripts/vggt.sh [DATASET] [SEQUENCE] [FRAME_INDEX] [SPLIT] [MASK_TYPE]

Arguments:
  DATASET      DNA | I3D | ZJU
  SEQUENCE     e.g. 0007_04 / ID1_1 / CoreView_377
  FRAME_INDEX  optional integer frame id. If omitted, auto-select mode is used.
  SPLIT        I3D only: train|novelview|novelpose (default=train)
  MASK_TYPE    ZJU only: mask|mask_cihp (default=mask)

Environment overrides:
  PYTHON_BIN             python executable
  NPZ_PATH               explicit bind input npz (single anchor only)
  MAX_POINTS             bind output max points (default 60000)
  SAMPLE_METHOD          random|fps (default fps)
  RUN_GARMENT_EXTRACT    1|0 regenerate garment ply (default 1)
  AUTO_EXTRACT_MISSING   1|0 run extracted.sh when extracted frame dirs are missing (default 1)

  BODY_KNN               cloth-to-body knn for weighted binding (default 4)
  PATCH_KNN              patch size for residual Procrustes (default 12)
  SIGMA_SCALE            gaussian weight scale (default 1.0)
  INVERSE_LBS_EPS        inverse-LBS stability epsilon (default 1e-6)
  RESIDUAL_WEIGHT        residual local-Procrustes weight (default 0.1)
  CHUNK_SIZE             bind chunk size (default 200000)

  BIND_FRAMES            explicit comma-separated anchor frames, e.g. 25,32,40
  ANCHOR_TOPK            auto-select top-k anchors (default 3)
  AUTO_SELECT_FRAME      1|0 whether to auto-select anchors (default auto)
  ANCHOR_FUSE_VOXEL_SIZE canonical fusion voxel size (default 0.005)
  MIN_ANCHOR_SUPPORT     min anchors supporting a fused voxel (default 2)

  SELECT_W_POSE          frame selection pose weight (default 1.0)
  SELECT_W_JOINT         frame selection joint weight (default 0.2)
  SELECT_W_CONTACT       frame selection self-contact weight (default 0.1)

  SEED                   random seed (default 0)
EOF
}

choose_dataset_interactive() {
    echo "请选择要处理的数据集:"
    echo "  1) DNA"
    echo "  2) I3D"
    echo "  3) ZJU"
    read -r -p "输入编号 [1-3]: " choice
    case "${choice}" in
        1) echo "DNA" ;;
        2) echo "I3D" ;;
        3) echo "ZJU" ;;
        *) echo "Invalid choice: ${choice}" >&2; exit 1 ;;
    esac
}

join_by_comma() {
    local IFS=','
    echo "$*"
}

frame6() {
    printf "%06d" "$1"
}

resolve_npz_for_frame() {
    local frame_index="$1"
    local frame6_local
    frame6_local="$(frame6 "${frame_index}")"

    if [[ -n "${NPZ_PATH:-}" ]]; then
        echo "${NPZ_PATH}"
        return 0
    fi

    if [[ "${DATASET}" == "DNA" ]]; then
        local cand_npz="${SEQ_DATA_ROOT}/model/${frame6_local}.npz"
        if [[ ! -f "${cand_npz}" ]]; then
            echo "Error: bind input npz not found: ${cand_npz}" >&2
            exit 1
        fi
        echo "${cand_npz}"
        return 0
    fi

    if [[ "${DATASET}" == "I3D" ]]; then
        local cand_npz="${SEQ_DATA_ROOT}/model/${frame6_local}.npz"
        if [[ -f "${cand_npz}" ]]; then
            echo "${cand_npz}"
            return 0
        fi

        local mesh_infos="${SEQ_DATA_ROOT}/mesh_infos.pkl"
        local frameid_pose="${SEQ_DATA_ROOT}/frameid_pose.pkl"
        local out_npz="${RUN_DIR}/i3d_bind_input_frame${frame6_local}.npz"
        if [[ ! -f "${mesh_infos}" || ! -f "${frameid_pose}" ]]; then
            echo "Error: I3D bind source missing: ${mesh_infos} / ${frameid_pose}" >&2
            exit 1
        fi

        "${PY_BIN}" - "${mesh_infos}" "${frameid_pose}" "${out_npz}" "${frame_index}" "${SEQAVATAR_ROOT}" <<'PY'
import os
import pickle
import sys

import numpy as np

mesh_infos_pkl, frameid_pose_pkl, out_npz, frame_idx_str, seqavatar_root = sys.argv[1:6]
frame_idx = int(frame_idx_str)

with open(mesh_infos_pkl, "rb") as f:
    mesh_infos = pickle.load(f)
with open(frameid_pose_pkl, "rb") as f:
    frameid_pose = pickle.load(f)

if frame_idx >= len(frameid_pose):
    raise IndexError(f"frame_idx={frame_idx} out of range ({len(frameid_pose)})")

frame_name = f"frame_{frame_idx:06d}_view_01"
mesh = mesh_infos.get(frame_name, None)
if mesh is None:
    raise KeyError(f"cannot find key in mesh_infos: {frame_name}")

rh = np.asarray(mesh["Rh"], dtype=np.float32).reshape(3)
th = np.asarray(mesh["Th"], dtype=np.float32).reshape(3)

def axis_angle_to_matrix(vec):
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)
    axis = vec / theta
    kx, ky, kz = axis.tolist()
    k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float32)
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)

R_mat = axis_angle_to_matrix(rh)

data = frameid_pose[frame_idx]
poses_data = np.asarray(data["poses"], dtype=np.float32).reshape(-1, 3)
full_poses = np.zeros((72,), dtype=np.float32)
full_poses[3:] = poses_data.flatten()
betas = np.zeros((10,), dtype=np.float32)

sys.path.append(seqavatar_root)
from smpl_model.smpl.smpl_numpy import SMPL  # pylint: disable=import-error

smpl_model = SMPL(sex="neutral", model_dir=os.path.join(seqavatar_root, "smpl_model/models/"))
xyz, _ = smpl_model(full_poses[None, :].astype(np.float32), betas.reshape(-1))
obs_xyz = (xyz @ R_mat.T + th[None, :]).astype(np.float32)

np.savez(
    out_npz,
    obs_xyz=obs_xyz,
    R=R_mat.astype(np.float32),
    Th=th.astype(np.float32),
    Rh=rh.astype(np.float32),
    poses=full_poses.reshape(1, -1).astype(np.float32),
    shapes=betas.reshape(1, -1).astype(np.float32),
    betas=betas.reshape(1, -1).astype(np.float32),
)
print(f"Built I3D bind npz: {out_npz}")
PY
        echo "${out_npz}"
        return 0
    fi

    if [[ "${DATASET}" == "ZJU" ]]; then
        local param_npy="${SEQ_DATA_ROOT}/new_params/${frame_index}.npy"
        local vert_npy="${SEQ_DATA_ROOT}/new_vertices/${frame_index}.npy"
        local out_npz="${RUN_DIR}/zju_bind_input_frame${frame6_local}.npz"
        if [[ ! -f "${param_npy}" || ! -f "${vert_npy}" ]]; then
            echo "Error: ZJU frame params missing: ${param_npy} / ${vert_npy}" >&2
            exit 1
        fi

        "${PY_BIN}" - "${param_npy}" "${vert_npy}" "${out_npz}" <<'PY'
import sys
import numpy as np

param_npy, vert_npy, out_npz = sys.argv[1:4]
param = np.load(param_npy, allow_pickle=True).item()
obs_xyz = np.load(vert_npy).astype(np.float32)

rh = np.asarray(param["Rh"], dtype=np.float32).reshape(-1, 3)[0]
th = np.asarray(param["Th"], dtype=np.float32).reshape(-1, 3)[0]
base_pose = np.asarray(param["poses"], dtype=np.float32).reshape(1, -1)
full_pose = np.concatenate([rh.reshape(1, 3), base_pose[:, 3:]], axis=-1).astype(np.float32)
shapes = np.asarray(param.get("shapes", np.zeros((1, 10), dtype=np.float32)), dtype=np.float32).reshape(1, -1)


def axis_angle_to_matrix(vec):
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)
    axis = vec / theta
    kx, ky, kz = axis.tolist()
    k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float32)
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)

rot = axis_angle_to_matrix(rh)

np.savez(
    out_npz,
    obs_xyz=obs_xyz,
    R=rot.astype(np.float32),
    Th=th.astype(np.float32),
    Rh=rh.astype(np.float32),
    poses=full_pose.astype(np.float32),
    shapes=shapes.astype(np.float32),
)
print(f"Built ZJU bind npz: {out_npz}")
PY
        echo "${out_npz}"
        return 0
    fi

    echo "Error: unsupported dataset in resolve_npz_for_frame: ${DATASET}" >&2
    exit 1
}

ensure_extracted_for_frame() {
    local frame_index="$1"
    local frame6_local
    frame6_local="$(frame6 "${frame_index}")"

    local seq_root="${EXTRACT_ROOT}/${EXTRACT_DATASET}/${EXTRACT_SEQ_KEY}"
    local img_dir="${seq_root}/extracted_frame_${PREFIX}_${frame6_local}"
    local mask_dir="${seq_root}/extracted_frame_${PREFIX}_mask_${frame6_local}"

    if [[ ! -d "${img_dir}" || ! -d "${mask_dir}" ]]; then
        if [[ "${AUTO_EXTRACT_MISSING}" == "1" ]]; then
            /bin/bash "${SEQAVATAR_ROOT}/scripts/extracted.sh" "${DATASET}" "${SEQUENCE}" "${frame_index}" "${SPLIT}" "${MASK_TYPE}"
        fi
    fi

    if [[ ! -d "${img_dir}" ]]; then
        echo "Error: extracted image dir not found: ${img_dir}" >&2
        exit 1
    fi
    if [[ ! -d "${mask_dir}" ]]; then
        echo "Error: extracted mask dir not found: ${mask_dir}" >&2
        exit 1
    fi

    LAST_IMG_DIR="${img_dir}"
    LAST_MASK_DIR="${mask_dir}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

DATASET_RAW="${1:-}"
if [[ -z "${DATASET_RAW}" ]]; then
    DATASET_RAW="$(choose_dataset_interactive)"
fi
DATASET="$(echo "${DATASET_RAW}" | tr '[:lower:]' '[:upper:]')"

SEQUENCE="${2:-}"
FRAME_INDEX="${3:-}"
SPLIT="${4:-train}"
MASK_TYPE="${5:-mask}"

if [[ -z "${SEQUENCE}" ]]; then
    read -r -p "请输入序列名 (e.g. 0007_04 / ID1_1 / CoreView_377): " SEQUENCE
fi

MAX_POINTS="${MAX_POINTS:-60000}"
SAMPLE_METHOD="${SAMPLE_METHOD:-fps}"
RUN_GARMENT_EXTRACT="${RUN_GARMENT_EXTRACT:-1}"
AUTO_EXTRACT_MISSING="${AUTO_EXTRACT_MISSING:-1}"

BODY_KNN="${BODY_KNN:-4}"
PATCH_KNN="${PATCH_KNN:-12}"
SIGMA_SCALE="${SIGMA_SCALE:-1.0}"
INVERSE_LBS_EPS="${INVERSE_LBS_EPS:-1e-6}"
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.1}"
CHUNK_SIZE="${CHUNK_SIZE:-200000}"

BIND_FRAMES="${BIND_FRAMES:-}"
ANCHOR_TOPK="${ANCHOR_TOPK:-3}"
AUTO_SELECT_FRAME="${AUTO_SELECT_FRAME:-}"
ANCHOR_FUSE_VOXEL_SIZE="${ANCHOR_FUSE_VOXEL_SIZE:-0.005}"
MIN_ANCHOR_SUPPORT="${MIN_ANCHOR_SUPPORT:-2}"

SELECT_W_POSE="${SELECT_W_POSE:-1.0}"
SELECT_W_JOINT="${SELECT_W_JOINT:-0.2}"
SELECT_W_CONTACT="${SELECT_W_CONTACT:-0.1}"

CROSS_VIEW_MIN_SUPPORT="${CROSS_VIEW_MIN_SUPPORT:-2}"
DEPTH_ABS_TOL="${DEPTH_ABS_TOL:-0.01}"
DEPTH_REL_TOL="${DEPTH_REL_TOL:-0.03}"
VOXEL_SIZE="${VOXEL_SIZE:-0.005}"
RADIUS_OUTLIER_RADIUS="${RADIUS_OUTLIER_RADIUS:-0.02}"
RADIUS_OUTLIER_MIN_NEIGHBORS="${RADIUS_OUTLIER_MIN_NEIGHBORS:-6}"
STAT_NB_NEIGHBORS="${STAT_NB_NEIGHBORS:-20}"
STAT_STD_RATIO="${STAT_STD_RATIO:-2.0}"
DBSCAN_EPS="${DBSCAN_EPS:-0.025}"
DBSCAN_MIN_POINTS="${DBSCAN_MIN_POINTS:-20}"

SEED="${SEED:-0}"

EXTRACT_DATASET=""
PREFIX=""
EXTRACT_SEQ_KEY=""
SEQ_DATA_ROOT=""
MODEL_TYPE=""
BODY_MODEL_PATH=""

case "${DATASET}" in
    DNA)
        EXTRACT_DATASET="DNA_Rendering"
        PREFIX="DNA"
        EXTRACT_SEQ_KEY="${SEQUENCE}"
        SEQ_DATA_ROOT="${SEQAVATAR_ROOT}/DNA-Rendering/${SEQUENCE}"
        MODEL_TYPE="smplx"
        BODY_MODEL_PATH="${SMPL_MODEL_DIR}/SMPLX_NEUTRAL.npz"
        ;;
    I3D)
        EXTRACT_DATASET="I3D-Human"
        PREFIX="I3D"
        if [[ "${SEQUENCE}" == *-train || "${SEQUENCE}" == *-novelview || "${SEQUENCE}" == *-novelpose ]]; then
            I3D_SEQ_DIR="${SEQUENCE}"
            EXTRACT_SEQ_KEY="${SEQUENCE%-*}"
        else
            I3D_SEQ_DIR="${SEQUENCE}-${SPLIT}"
            EXTRACT_SEQ_KEY="${SEQUENCE}"
        fi
        SEQ_DATA_ROOT="${SEQAVATAR_ROOT}/I3D-Human/${I3D_SEQ_DIR}"
        MODEL_TYPE="smpl"
        BODY_MODEL_PATH="${SMPL_MODEL_DIR}/SMPL_NEUTRAL.pkl"
        ;;
    ZJU)
        EXTRACT_DATASET="ZJU-MoCap"
        PREFIX="ZJU"
        EXTRACT_SEQ_KEY="${SEQUENCE}"
        if [[ "${MASK_TYPE}" != "mask" && "${MASK_TYPE}" != "mask_cihp" ]]; then
            echo "Error: ZJU MASK_TYPE must be 'mask' or 'mask_cihp'"
            exit 1
        fi
        SEQ_DATA_ROOT="${SEQAVATAR_ROOT}/ZJU-MoCap/${SEQUENCE}"
        MODEL_TYPE="smpl"
        BODY_MODEL_PATH="${SMPL_MODEL_DIR}/SMPL_NEUTRAL.pkl"
        ;;
    *)
        echo "Error: unsupported dataset '${DATASET_RAW}', use DNA|I3D|ZJU"
        exit 1
        ;;
esac

if [[ ! -f "${EXTRACT_GARMENT_PY}" || ! -f "${BIND_CANONICAL_PY}" || ! -f "${SELECT_BIND_FRAME_PY}" ]]; then
    echo "Error: required VGGT scripts not found under ${VGGT_ROOT}"
    exit 1
fi
if [[ ! -d "${SEQ_DATA_ROOT}" ]]; then
    echo "Error: sequence data root not found: ${SEQ_DATA_ROOT}"
    exit 1
fi
if [[ ! -f "${BODY_MODEL_PATH}" ]]; then
    echo "Error: body model not found: ${BODY_MODEL_PATH}"
    exit 1
fi

STAMP_MINUTE="${STAMP_MINUTE:-$(date +%Y%m%d_%H%M)}"
RUN_TAG="${RUN_TAG:-bind_${STAMP_MINUTE}}"
RUN_DIR="${EXTRACT_ROOT}/${EXTRACT_DATASET}/${EXTRACT_SEQ_KEY}/vggt_outputs/${RUN_TAG}"
mkdir -p "${RUN_DIR}" "${SEQ_DATA_ROOT}/OUTPUT_PT"

case "${DATASET}" in
    DNA)
        VGGT_PT_DIR="${VGGT_PT_ROOT}/DNA"
        SEQ_PT_DIR="${SEQ_PT_ROOT}/DNA_Rendering"
        ;;
    I3D)
        VGGT_PT_DIR="${VGGT_PT_ROOT}/I3D-Human"
        SEQ_PT_DIR="${SEQ_PT_ROOT}/I3D-Human"
        ;;
    ZJU)
        VGGT_PT_DIR="${VGGT_PT_ROOT}/ZJU-MoCap"
        SEQ_PT_DIR="${SEQ_PT_ROOT}/ZJU-MoCap"
        ;;
esac
mkdir -p "${VGGT_PT_DIR}" "${SEQ_PT_DIR}"

if [[ -z "${AUTO_SELECT_FRAME}" ]]; then
    if [[ -n "${BIND_FRAMES}" ]]; then
        AUTO_SELECT_FRAME=0
    elif [[ -n "${FRAME_INDEX}" ]]; then
        AUTO_SELECT_FRAME=0
    else
        AUTO_SELECT_FRAME=1
    fi
fi

frames=()
if [[ -n "${BIND_FRAMES}" ]]; then
    IFS=',' read -r -a frames <<< "${BIND_FRAMES}"
elif [[ "${AUTO_SELECT_FRAME}" == "1" ]]; then
    selector_json="${RUN_DIR}/selected_frames.json"
    sel_out="$("${PY_BIN}" "${SELECT_BIND_FRAME_PY}" \
        --dataset "${DATASET}" \
        --sequence "${SEQUENCE}" \
        --split "${SPLIT}" \
        --topk "${ANCHOR_TOPK}" \
        --seqavatar_root "${SEQAVATAR_ROOT}" \
        --w_pose "${SELECT_W_POSE}" \
        --w_joint "${SELECT_W_JOINT}" \
        --w_contact "${SELECT_W_CONTACT}" \
        --output_json "${selector_json}")"
    echo "${sel_out}"
    frame_csv="$(echo "${sel_out}" | awk -F= '/^SELECTED_FRAMES=/{print $2}' | tail -n1)"
    if [[ -z "${frame_csv}" ]]; then
        echo "Error: failed to parse selected frames from select_bind_frame output" >&2
        exit 1
    fi
    IFS=',' read -r -a frames <<< "${frame_csv}"
else
    if [[ -z "${FRAME_INDEX}" ]]; then
        read -r -p "请输入帧号 (integer): " FRAME_INDEX
    fi
    frames=("${FRAME_INDEX}")
fi

if [[ "${#frames[@]}" -eq 0 ]]; then
    echo "Error: no anchor frames resolved." >&2
    exit 1
fi

for f in "${frames[@]}"; do
    if ! [[ "${f}" =~ ^[0-9]+$ ]]; then
        echo "Error: frame must be integer, got '${f}'" >&2
        exit 1
    fi
done

first_frame6="$(frame6 "${frames[0]}")"
if [[ "${#frames[@]}" -eq 1 ]]; then
    OUTPUT_PT="${SEQ_DATA_ROOT}/OUTPUT_PT/vggt_canonical_init_frame${first_frame6}.pt"
else
    OUTPUT_PT="${SEQ_DATA_ROOT}/OUTPUT_PT/vggt_canonical_init_${RUN_TAG}.pt"
fi
DEBUG_PLY="${RUN_DIR}/vggt_canonical_debug_${RUN_TAG}.ply"

NAME_CORE="${STAMP_MINUTE}_${DATASET}_${SEQUENCE}"
GARMENT_PT_VGGT="${VGGT_PT_DIR}/garment_${NAME_CORE}.pt"
GARMENT_PT_SEQ="${SEQ_PT_DIR}/garment_${NAME_CORE}.pt"
CANONICAL_PT_VGGT="${VGGT_PT_DIR}/canonical_${NAME_CORE}.pt"
CANONICAL_PT_SEQ="${SEQ_PT_DIR}/canonical_${NAME_CORE}.pt"

echo "==============================================="
echo "Dataset             : ${DATASET}"
echo "Sequence            : ${SEQUENCE}"
echo "Split/Mask          : ${SPLIT} / ${MASK_TYPE}"
echo "Anchor frames       : $(join_by_comma "${frames[@]}")"
echo "RUN_DIR             : ${RUN_DIR}"
echo "Bind model          : ${MODEL_TYPE} (${BODY_MODEL_PATH})"
echo "SeqAvatar PT        : ${OUTPUT_PT}"
echo "Garment PT          : ${GARMENT_PT_VGGT}"
echo "Canonical PT        : ${CANONICAL_PT_VGGT}"
echo "==============================================="

declare -a GARMENT_PLYS
declare -a NPZ_PATHS

for frame in "${frames[@]}"; do
    frame6_local="$(frame6 "${frame}")"
    ensure_extracted_for_frame "${frame}"

    frame_dir="${RUN_DIR}/frame_${frame6_local}"
    mkdir -p "${frame_dir}"

    garment_ply="${frame_dir}/vggt_garment_init_frame${frame6_local}.ply"
    npz_path_local="$(resolve_npz_for_frame "${frame}")"

    echo "--- Anchor frame ${frame6_local} ---"
    echo "Images : ${LAST_IMG_DIR}"
    echo "Masks  : ${LAST_MASK_DIR}"
    echo "NPZ    : ${npz_path_local}"
    echo "PLY    : ${garment_ply}"

    if [[ "${RUN_GARMENT_EXTRACT}" == "1" ]]; then
        (cd "${VGGT_ROOT}" && "${PY_BIN}" "${EXTRACT_GARMENT_PY}" \
            --image_dir "${LAST_IMG_DIR}" \
            --mask_dir "${LAST_MASK_DIR}" \
            --output_ply "${garment_ply}" \
            --cross_view_min_support "${CROSS_VIEW_MIN_SUPPORT}" \
            --depth_abs_tol "${DEPTH_ABS_TOL}" \
            --depth_rel_tol "${DEPTH_REL_TOL}" \
            --voxel_size "${VOXEL_SIZE}" \
            --radius_outlier_radius "${RADIUS_OUTLIER_RADIUS}" \
            --radius_outlier_min_neighbors "${RADIUS_OUTLIER_MIN_NEIGHBORS}" \
            --stat_nb_neighbors "${STAT_NB_NEIGHBORS}" \
            --stat_std_ratio "${STAT_STD_RATIO}" \
            --dbscan_eps "${DBSCAN_EPS}" \
            --dbscan_min_points "${DBSCAN_MIN_POINTS}")
    else
        echo "Skip garment extraction and reuse existing ply: ${garment_ply}"
        if [[ ! -f "${garment_ply}" ]]; then
            echo "Error: RUN_GARMENT_EXTRACT=0 but garment ply not found: ${garment_ply}" >&2
            exit 1
        fi
    fi

    GARMENT_PLYS+=("${garment_ply}")
    NPZ_PATHS+=("${npz_path_local}")
done

ply_csv="$(join_by_comma "${GARMENT_PLYS[@]}")"
npz_csv="$(join_by_comma "${NPZ_PATHS[@]}")"

(cd "${VGGT_ROOT}" && "${PY_BIN}" "${BIND_CANONICAL_PY}" \
    --model_type "${MODEL_TYPE}" \
    --smplx_model_path "${BODY_MODEL_PATH}" \
    --ply_path "${GARMENT_PLYS[0]}" \
    --npz_path "${NPZ_PATHS[0]}" \
    --ply_paths "${ply_csv}" \
    --npz_paths "${npz_csv}" \
    --output_pt "${OUTPUT_PT}" \
    --debug_ply "${DEBUG_PLY}" \
    --max_points "${MAX_POINTS}" \
    --sample_method "${SAMPLE_METHOD}" \
    --body_knn "${BODY_KNN}" \
    --patch_knn "${PATCH_KNN}" \
    --sigma_scale "${SIGMA_SCALE}" \
    --seqavatar_root "${SEQAVATAR_ROOT}" \
    --anchor_fuse_voxel_size "${ANCHOR_FUSE_VOXEL_SIZE}" \
    --min_anchor_support "${MIN_ANCHOR_SUPPORT}" \
    --inverse_lbs_eps "${INVERSE_LBS_EPS}" \
    --residual_weight "${RESIDUAL_WEIGHT}" \
    --chunk_size "${CHUNK_SIZE}" \
    --seed "${SEED}")

"${PY_BIN}" - "${GARMENT_PT_VGGT}" "${frames[*]}" "${GARMENT_PLYS[@]}" <<'PY'
import sys
import numpy as np
import torch
from plyfile import PlyData

out_pt = sys.argv[1]
frames = [int(x) for x in sys.argv[2].split() if x.strip()]
ply_paths = sys.argv[3:]

all_xyz = []
for p in ply_paths:
    ply = PlyData.read(p)
    x = np.asarray(ply.elements[0]["x"], dtype=np.float32)
    y = np.asarray(ply.elements[0]["y"], dtype=np.float32)
    z = np.asarray(ply.elements[0]["z"], dtype=np.float32)
    all_xyz.append(np.stack([x, y, z], axis=1))

if all_xyz:
    xyz = np.concatenate(all_xyz, axis=0)
else:
    xyz = np.zeros((0, 3), dtype=np.float32)

torch.save(
    {
        "xyz": torch.from_numpy(xyz),
        "source_plys": ply_paths,
        "anchor_frames": frames,
    },
    out_pt,
)
print(f"Saved garment pt: {out_pt}, points={xyz.shape[0]}, anchors={len(frames)}")
PY

cp -f "${GARMENT_PT_VGGT}" "${GARMENT_PT_SEQ}"
cp -f "${OUTPUT_PT}" "${CANONICAL_PT_VGGT}"
cp -f "${OUTPUT_PT}" "${CANONICAL_PT_SEQ}"

echo "Saved:"
echo "  ${GARMENT_PT_VGGT}"
echo "  ${GARMENT_PT_SEQ}"
echo "  ${CANONICAL_PT_VGGT}"
echo "  ${CANONICAL_PT_SEQ}"
echo "Done."
