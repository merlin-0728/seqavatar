#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/media/image/mxz/human/SeqAvatar"
EXTRACT_PY="${ROOT_DIR}/extracted_frame/extract_frame.py"
OUTPUT_ROOT="${ROOT_DIR}/extracted_frame"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Error: neither python nor python3 is available in PATH."
        exit 1
    fi
fi

usage() {
    cat <<'EOF'
Usage:
  ./scripts/exps_vggt.sh <DATASET> <SEQUENCE> <FRAME_INDEX> [SPLIT] [MASK_TYPE]

Arguments:
  DATASET      DNA | I3D | ZJU
  SEQUENCE     e.g. 0007_04 / ID1_1 / CoreView_377
  FRAME_INDEX  integer frame id, e.g. 25
  SPLIT        (I3D only, optional) train|novelview|novelpose, default=train
  MASK_TYPE    (ZJU only, optional) mask|mask_cihp, default=mask

Examples:
  ./scripts/exps_vggt.sh DNA 0007_04 25
  ./scripts/exps_vggt.sh I3D ID1_1 301 train
  ./scripts/exps_vggt.sh ZJU CoreView_377 25 mask_cihp
EOF
}

if [[ $# -lt 3 ]]; then
    usage
    exit 1
fi

DATASET_RAW="$1"
SEQUENCE_RAW="$2"
FRAME_INDEX="$3"
SPLIT="${4:-train}"
MASK_TYPE="${5:-mask}"

if ! [[ "${FRAME_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Error: FRAME_INDEX must be an integer, got '${FRAME_INDEX}'"
    exit 1
fi

if [[ ! -f "${EXTRACT_PY}" ]]; then
    echo "Error: extractor not found: ${EXTRACT_PY}"
    exit 1
fi

DATASET="$(echo "${DATASET_RAW}" | tr '[:lower:]' '[:upper:]')"
FRAME6="$(printf "%06d" "${FRAME_INDEX}")"

IMAGE_BASE=""
MASK_BASE=""
OUT_DATASET=""
OUT_SEQ=""
OUT_PREFIX=""

case "${DATASET}" in
    DNA)
        IMAGE_BASE="${ROOT_DIR}/DNA-Rendering/${SEQUENCE_RAW}/images"
        MASK_BASE="${ROOT_DIR}/DNA-Rendering/${SEQUENCE_RAW}/bkgd_masks"
        OUT_DATASET="DNA_Rendering"
        OUT_SEQ="${SEQUENCE_RAW}"
        OUT_PREFIX="DNA"
        ;;
    I3D)
        if [[ "${SEQUENCE_RAW}" == *-train || "${SEQUENCE_RAW}" == *-novelview || "${SEQUENCE_RAW}" == *-novelpose ]]; then
            I3D_SEQ_DIR="${SEQUENCE_RAW}"
            OUT_SEQ="${SEQUENCE_RAW%-*}"
        else
            I3D_SEQ_DIR="${SEQUENCE_RAW}-${SPLIT}"
            OUT_SEQ="${SEQUENCE_RAW}"
        fi
        IMAGE_BASE="${ROOT_DIR}/I3D-Human/${I3D_SEQ_DIR}/images"
        MASK_BASE="${ROOT_DIR}/I3D-Human/${I3D_SEQ_DIR}/masks"
        OUT_DATASET="I3D-Human"
        OUT_PREFIX="I3D"
        ;;
    ZJU)
        if [[ "${MASK_TYPE}" != "mask" && "${MASK_TYPE}" != "mask_cihp" ]]; then
            echo "Error: ZJU MASK_TYPE must be 'mask' or 'mask_cihp', got '${MASK_TYPE}'"
            exit 1
        fi
        IMAGE_BASE="${ROOT_DIR}/ZJU-MoCap/${SEQUENCE_RAW}"
        MASK_BASE="${ROOT_DIR}/ZJU-MoCap/${SEQUENCE_RAW}/${MASK_TYPE}"
        OUT_DATASET="ZJU-MoCap"
        OUT_SEQ="${SEQUENCE_RAW}"
        OUT_PREFIX="ZJU"
        ;;
    *)
        echo "Error: unknown DATASET '${DATASET_RAW}'. Use DNA | I3D | ZJU."
        exit 1
        ;;
esac

if [[ ! -d "${IMAGE_BASE}" ]]; then
    echo "Error: image base not found: ${IMAGE_BASE}"
    exit 1
fi
if [[ ! -d "${MASK_BASE}" ]]; then
    echo "Error: mask base not found: ${MASK_BASE}"
    exit 1
fi

OUT_SEQ_DIR="${OUTPUT_ROOT}/${OUT_DATASET}/${OUT_SEQ}"
OUT_IMG_DIR="${OUT_SEQ_DIR}/extracted_frame_${OUT_PREFIX}_${FRAME6}"
OUT_MASK_DIR="${OUT_SEQ_DIR}/extracted_frame_${OUT_PREFIX}_mask_${FRAME6}"
mkdir -p "${OUT_SEQ_DIR}"

echo "==============================================="
echo "Dataset      : ${DATASET}"
echo "Sequence     : ${SEQUENCE_RAW}"
echo "Frame        : ${FRAME6}"
echo "Image source : ${IMAGE_BASE}"
echo "Mask source  : ${MASK_BASE}"
echo "Image output : ${OUT_IMG_DIR}"
echo "Mask output  : ${OUT_MASK_DIR}"
echo "==============================================="

"${PYTHON_BIN}" "${EXTRACT_PY}" --base_path "${IMAGE_BASE}" --frame "${FRAME_INDEX}" --output_dir "${OUT_IMG_DIR}"
"${PYTHON_BIN}" "${EXTRACT_PY}" --base_path "${MASK_BASE}" --frame "${FRAME_INDEX}" --output_dir "${OUT_MASK_DIR}"

echo "Done."
