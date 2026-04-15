#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/video.sh <renders_dir> <view_id> [fps] [output_mp4]

Arguments:
  renders_dir   渲染图片目录，例如:
                /media/image/mxz/human/SeqAvatar/output/DNA-Rendering/0007_04/novelview/ours_25000/renders_20260405_232352
  view_id       摄像头编号，例如 48 / 50 / 54
  fps           输出视频帧率，默认 24
  output_mp4    输出视频路径（可选）

Example:
  ./scripts/video.sh \
    /media/image/mxz/human/SeqAvatar/output/DNA-Rendering/0007_04/novelview/ours_25000/renders_20260405_232352 \
    54 24 \
    /media/image/mxz/human/SeqAvatar/output/DNA-Rendering/0007_04/novelview/ours_25000/view54.mp4
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

RENDERS_DIR="$1"
VIEW_RAW="$2"
FPS="${3:-24}"
OUTPUT_MP4="${4:-}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg 未安装或不在 PATH 中。"
    exit 1
fi

if [[ ! -d "${RENDERS_DIR}" ]]; then
    echo "Error: 渲染目录不存在: ${RENDERS_DIR}"
    exit 1
fi

if [[ "${VIEW_RAW}" =~ ^view_([0-9]+)$ ]]; then
    VIEW_ID="${BASH_REMATCH[1]}"
elif [[ "${VIEW_RAW}" =~ ^[0-9]+$ ]]; then
    VIEW_ID="${VIEW_RAW}"
else
    echo "Error: view_id 必须是数字（如 54）或 view_54。"
    exit 1
fi

if ! [[ "${FPS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: fps 必须是正数，当前: ${FPS}"
    exit 1
fi

if [[ -z "${OUTPUT_MP4}" ]]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_MP4="${RENDERS_DIR}/video_view_${VIEW_ID}_${STAMP}.mp4"
fi

WORK_DIR="$(mktemp -d /tmp/seqavatar_video_XXXXXX)"
cleanup() {
    rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

mapfile -t VIEW_FILES < <(
    python3 - "${RENDERS_DIR}" "${VIEW_ID}" <<'PY'
import os
import re
import sys

root = sys.argv[1]
view_id = int(sys.argv[2])
pat = re.compile(r"^frame_(\d+)_view_(\d+)\.png$")
items = []
for name in os.listdir(root):
    m = pat.match(name)
    if not m:
        continue
    frame_idx = int(m.group(1))
    cam_id = int(m.group(2))
    if cam_id != view_id:
        continue
    items.append((frame_idx, name))
items.sort(key=lambda x: x[0])
for _, name in items:
    print(name)
PY
)

if [[ "${#VIEW_FILES[@]}" -eq 0 ]]; then
    echo "Error: 在目录中没找到 view=${VIEW_ID} 的图片。"
    exit 1
fi

for i in "${!VIEW_FILES[@]}"; do
    printf -v IDX "%06d" "${i}"
    ln -s "${RENDERS_DIR}/${VIEW_FILES[$i]}" "${WORK_DIR}/${IDX}.png"
done

mkdir -p "$(dirname "${OUTPUT_MP4}")"
ffmpeg -y \
    -framerate "${FPS}" \
    -i "${WORK_DIR}/%06d.png" \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -crf 18 \
    -preset medium \
    "${OUTPUT_MP4}" >/dev/null 2>&1

echo "Done."
echo "view_id      : ${VIEW_ID}"
echo "frame_count  : ${#VIEW_FILES[@]}"
echo "fps          : ${FPS}"
echo "output_video : ${OUTPUT_MP4}"
