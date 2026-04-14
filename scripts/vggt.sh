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
  FRAME_INDEX  integer frame id (e.g. 25)
  SPLIT        I3D only: train|novelview|novelpose (default=train)
  MASK_TYPE    ZJU only: mask|mask_cihp (default=mask)

Environment overrides:
  PYTHON_BIN      python executable
  NPZ_PATH        explicit bind input npz (skip auto resolution)
  MAX_POINTS      bind output max points (default 60000)
  SAMPLE_METHOD   random|fps (default fps)
  SEED            random seed (default 0)
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
if [[ -z "${FRAME_INDEX}" ]]; then
    read -r -p "请输入帧号 (integer): " FRAME_INDEX
fi
if ! [[ "${FRAME_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Error: FRAME_INDEX must be an integer, got '${FRAME_INDEX}'"
    exit 1
fi

FRAME6="$(printf "%06d" "${FRAME_INDEX}")"
FRAME_INT="$((10#${FRAME_INDEX}))"
MAX_POINTS="${MAX_POINTS:-60000}"
SAMPLE_METHOD="${SAMPLE_METHOD:-fps}"
SEED="${SEED:-0}"

IMG_DIR=""
MASK_DIR=""
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

EXTRACT_SEQ_ROOT="${EXTRACT_ROOT}/${EXTRACT_DATASET}/${EXTRACT_SEQ_KEY}"
IMG_DIR="${EXTRACT_SEQ_ROOT}/extracted_frame_${PREFIX}_${FRAME6}"
MASK_DIR="${EXTRACT_SEQ_ROOT}/extracted_frame_${PREFIX}_mask_${FRAME6}"

if [[ ! -d "${IMG_DIR}" ]]; then
    echo "Error: extracted image dir not found: ${IMG_DIR}"
    echo "请先运行 extracted.sh 生成该帧多视角图像。"
    exit 1
fi
if [[ ! -d "${MASK_DIR}" ]]; then
    echo "Error: extracted mask dir not found: ${MASK_DIR}"
    echo "请先运行 extracted.sh 生成该帧多视角 mask。"
    exit 1
fi
if [[ ! -f "${EXTRACT_GARMENT_PY}" || ! -f "${BIND_CANONICAL_PY}" ]]; then
    echo "Error: VGGT scripts not found under ${VGGT_ROOT}"
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

RUN_DIR="${EXTRACT_SEQ_ROOT}/vggt_outputs/frame_${FRAME6}"
mkdir -p "${RUN_DIR}"
mkdir -p "${SEQ_DATA_ROOT}/OUTPUT_PT"

GARMENT_PLY="${RUN_DIR}/vggt_garment_init_frame${FRAME6}.ply"
DEBUG_PLY="${RUN_DIR}/vggt_canonical_debug_frame${FRAME6}.ply"
OUTPUT_PT="${SEQ_DATA_ROOT}/OUTPUT_PT/vggt_canonical_init_frame${FRAME6}.pt"

STAMP_MINUTE="${STAMP_MINUTE:-$(date +%Y%m%d_%H%M)}"
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

NAME_CORE="${STAMP_MINUTE}_${DATASET}_${SEQUENCE}"
GARMENT_PT_VGGT="${VGGT_PT_DIR}/garment_${NAME_CORE}.pt"
GARMENT_PT_SEQ="${SEQ_PT_DIR}/garment_${NAME_CORE}.pt"
CANONICAL_PT_VGGT="${VGGT_PT_DIR}/canonical_${NAME_CORE}.pt"
CANONICAL_PT_SEQ="${SEQ_PT_DIR}/canonical_${NAME_CORE}.pt"

RESOLVED_NPZ="${NPZ_PATH:-}"
if [[ -z "${RESOLVED_NPZ}" ]]; then
    if [[ "${DATASET}" == "DNA" ]]; then
        CAND_NPZ="${SEQ_DATA_ROOT}/model/${FRAME6}.npz"
        if [[ -f "${CAND_NPZ}" ]]; then
            RESOLVED_NPZ="${CAND_NPZ}"
        else
            echo "Error: bind input npz not found: ${CAND_NPZ}"
            echo "你可以手动指定 NPZ_PATH=/abs/path/to/file.npz"
            exit 1
        fi
    elif [[ "${DATASET}" == "I3D" ]]; then
        CAND_NPZ="${SEQ_DATA_ROOT}/model/${FRAME6}.npz"
        if [[ -f "${CAND_NPZ}" ]]; then
            RESOLVED_NPZ="${CAND_NPZ}"
        else
            MESH_INFOS="${SEQ_DATA_ROOT}/mesh_infos.pkl"
            FRAMEID_POSE="${SEQ_DATA_ROOT}/frameid_pose.pkl"
            RESOLVED_NPZ="${RUN_DIR}/i3d_bind_input_frame${FRAME6}.npz"
            if [[ ! -f "${MESH_INFOS}" || ! -f "${FRAMEID_POSE}" ]]; then
                echo "Error: I3D bind input source files missing:"
                echo "  ${MESH_INFOS}"
                echo "  ${FRAMEID_POSE}"
                echo "你也可以手动指定 NPZ_PATH=/abs/path/to/file.npz"
                exit 1
            fi
            "${PY_BIN}" - "${MESH_INFOS}" "${FRAMEID_POSE}" "${RESOLVED_NPZ}" "${FRAME_INT}" "${SEQAVATAR_ROOT}" <<'PY'
import os
import pickle
import sys

import cv2
import numpy as np

mesh_infos_pkl, frameid_pose_pkl, out_npz, frame_idx_str, seqavatar_root = sys.argv[1:6]
frame_idx = int(frame_idx_str)

with open(mesh_infos_pkl, "rb") as f:
    mesh_infos = pickle.load(f)
with open(frameid_pose_pkl, "rb") as f:
    frameid_pose = pickle.load(f)

if frame_idx >= len(frameid_pose):
    raise IndexError(f"frame_idx={frame_idx} out of range for frameid_pose(len={len(frameid_pose)})")

frame_name = f"frame_{frame_idx:06d}_view_01"
mesh = mesh_infos.get(frame_name, None)
if mesh is None:
    raise KeyError(f"cannot find key in mesh_infos: {frame_name}")

rh = np.asarray(mesh["Rh"], dtype=np.float32).reshape(3)
th = np.asarray(mesh["Th"], dtype=np.float32).reshape(3)
R_mat = cv2.Rodrigues(rh[None, :])[0].astype(np.float32)

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

np.savez(out_npz, obs_xyz=obs_xyz, R=R_mat, Th=th, betas=betas)
print(f"Built I3D bind npz: {out_npz}")
PY
        fi
    elif [[ "${DATASET}" == "ZJU" ]]; then
        PARAM_NPY="${SEQ_DATA_ROOT}/new_params/${FRAME_INT}.npy"
        VERT_NPY="${SEQ_DATA_ROOT}/new_vertices/${FRAME_INT}.npy"
        RESOLVED_NPZ="${RUN_DIR}/zju_bind_input_frame${FRAME6}.npz"
        if [[ ! -f "${PARAM_NPY}" || ! -f "${VERT_NPY}" ]]; then
            echo "Error: ZJU frame params not found:"
            echo "  ${PARAM_NPY}"
            echo "  ${VERT_NPY}"
            exit 1
        fi
        "${PY_BIN}" - "${PARAM_NPY}" "${VERT_NPY}" "${RESOLVED_NPZ}" <<'PY'
import sys
import numpy as np

param_npy, vert_npy, out_npz = sys.argv[1:4]
param = np.load(param_npy, allow_pickle=True).item()
obs_xyz = np.load(vert_npy).astype(np.float32)
rh = np.asarray(param["Rh"], dtype=np.float32).reshape(-1, 3)[0]
th = np.asarray(param["Th"], dtype=np.float32).reshape(-1, 3)[0]
betas = np.asarray(param["shapes"], dtype=np.float32).reshape(-1)

theta = float(np.linalg.norm(rh))
if theta < 1e-12:
    rot = np.eye(3, dtype=np.float32)
else:
    axis = rh / theta
    kx, ky, kz = axis
    k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float32)
    rot = np.eye(3, dtype=np.float32) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)

np.savez(out_npz, obs_xyz=obs_xyz, R=rot, Th=th, betas=betas)
print(f"Built ZJU bind npz: {out_npz}")
PY
    fi
fi

echo "==============================================="
echo "Dataset         : ${DATASET}"
echo "Sequence        : ${SEQUENCE}"
echo "Frame           : ${FRAME6}"
echo "Images          : ${IMG_DIR}"
echo "Masks           : ${MASK_DIR}"
echo "VGGT ply output : ${GARMENT_PLY}"
echo "Bind npz        : ${RESOLVED_NPZ}"
echo "Bind model      : ${MODEL_TYPE} (${BODY_MODEL_PATH})"
echo "SeqAvatar PT    : ${OUTPUT_PT}"
echo "Garment PT      : ${GARMENT_PT_VGGT}"
echo "Canonical PT    : ${CANONICAL_PT_VGGT}"
echo "==============================================="

(cd "${VGGT_ROOT}" && "${PY_BIN}" "${EXTRACT_GARMENT_PY}" \
    --image_dir "${IMG_DIR}" \
    --mask_dir "${MASK_DIR}" \
    --output_ply "${GARMENT_PLY}")

"${PY_BIN}" - "${GARMENT_PLY}" "${GARMENT_PT_VGGT}" <<'PY'
import sys
import numpy as np
import torch
from plyfile import PlyData

ply_path, out_pt = sys.argv[1:3]
ply = PlyData.read(ply_path)
x = np.asarray(ply.elements[0]["x"], dtype=np.float32)
y = np.asarray(ply.elements[0]["y"], dtype=np.float32)
z = np.asarray(ply.elements[0]["z"], dtype=np.float32)
xyz = np.stack([x, y, z], axis=1)
torch.save({"xyz": torch.from_numpy(xyz), "source_ply": ply_path}, out_pt)
print(f"Saved garment pt: {out_pt}, points={xyz.shape[0]}")
PY
cp -f "${GARMENT_PT_VGGT}" "${GARMENT_PT_SEQ}"

(cd "${VGGT_ROOT}" && "${PY_BIN}" "${BIND_CANONICAL_PY}" \
    --model_type "${MODEL_TYPE}" \
    --smplx_model_path "${BODY_MODEL_PATH}" \
    --ply_path "${GARMENT_PLY}" \
    --npz_path "${RESOLVED_NPZ}" \
    --output_pt "${OUTPUT_PT}" \
    --debug_ply "${DEBUG_PLY}" \
    --max_points "${MAX_POINTS}" \
    --sample_method "${SAMPLE_METHOD}" \
    --seed "${SEED}")

cp -f "${OUTPUT_PT}" "${CANONICAL_PT_VGGT}"
cp -f "${OUTPUT_PT}" "${CANONICAL_PT_SEQ}"

echo "Saved:"
echo "  ${GARMENT_PT_VGGT}"
echo "  ${GARMENT_PT_SEQ}"
echo "  ${CANONICAL_PT_VGGT}"
echo "  ${CANONICAL_PT_SEQ}"
echo "Done."
