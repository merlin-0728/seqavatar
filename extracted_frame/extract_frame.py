import argparse
import os
import re
import shutil
from pathlib import Path

VALID_EXTS = (".png", ".jpg", ".jpeg")


def _is_camera_like(dirname: str) -> bool:
    return dirname.isdigit() or dirname.startswith("Camera_")


def _copy_camera_folder_mode(base_path: Path, frame_index: int, output_dir: Path) -> int:
    frame6 = f"{int(frame_index):06d}"
    all_dirs = sorted([p for p in base_path.iterdir() if p.is_dir()])
    camera_like_dirs = [p for p in all_dirs if _is_camera_like(p.name)]
    cam_folders = camera_like_dirs if camera_like_dirs else all_dirs
    count = 0

    for cam_dir in cam_folders:
        src = None
        for ext in VALID_EXTS:
            candidate = cam_dir / f"{frame6}{ext}"
            if candidate.exists():
                src = candidate
                break
        if src is None:
            print(f"警告: 摄像头 {cam_dir.name} 下未找到 {frame6}.(png/jpg/jpeg)")
            continue

        dest = output_dir / f"cam_{cam_dir.name}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        print(f"已提取: 摄像头 {cam_dir.name} -> {dest}")
        count += 1

    return count


def _copy_flat_i3d_mode(base_path: Path, frame_index: int, output_dir: Path) -> int:
    frame6 = f"{int(frame_index):06d}"
    pattern = re.compile(r"^frame_(\d+)_view_([A-Za-z0-9]+)\.(png|jpg|jpeg)$", re.IGNORECASE)
    files = sorted([p for p in base_path.iterdir() if p.is_file()])
    count = 0

    for file_path in files:
        m = pattern.match(file_path.name)
        if m is None:
            continue
        if int(m.group(1)) != int(frame_index):
            continue

        view_id = m.group(2)
        ext = "." + m.group(3).lower()
        dest = output_dir / f"cam_{view_id}{ext}"
        shutil.copy2(file_path, dest)
        print(f"已提取: 视角 {view_id} (frame {frame6}) -> {dest}")
        count += 1

    return count


def extract_specific_frame(base_path, frame_index, output_dir):
    """
    base_path: 数据根目录，支持两种格式
      1) camera folders: root/00/000025.png (DNA/ZJU)
      2) flat files: root/frame_000025_view_01.png (I3D)
    frame_index: 帧索引 (int 或数字字符串)
    output_dir: 输出目录
    """
    base = Path(base_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out}")

    if not base.exists():
        raise FileNotFoundError(f"输入目录不存在: {base}")

    has_subdirs = any(p.is_dir() for p in base.iterdir())
    if has_subdirs:
        count = _copy_camera_folder_mode(base, int(frame_index), out)
        mode = "camera-folders"
    else:
        count = _copy_flat_i3d_mode(base, int(frame_index), out)
        mode = "flat-files"

    if count == 0:
        raise RuntimeError(
            f"未提取到文件。base_path={base}, frame={int(frame_index):06d}, mode={mode}"
        )

    print(f"\n任务完成！共提取了 {count} 张图片到 {out} (mode={mode})")
    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Extract one frame across all views/cameras.")
    parser.add_argument("--base_path", required=True, help="Input path for images/masks")
    parser.add_argument("--frame", type=int, required=True, help="Frame index, e.g. 25")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_specific_frame(args.base_path, args.frame, args.output_dir)
