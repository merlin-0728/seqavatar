import numpy as np
import cv2
from pathlib import Path
import imageio.v2 as imageio

# 所有序列列表
sequences = ["0007_04", "0019_10", "0044_11", "0051_09", "0206_04", "0813_05"]

# 根路径
data_root = Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering")

# 筛选参数
window_size = 5  # 计算局部 std 的卷积窗口

for seq in sequences:
    fs_depth_dir = data_root / seq / "render_depth/novelview/npy"
    out_dir = data_root / seq / "render_depth/novelview/std_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_files = sorted(fs_depth_dir.glob("*.npy"))
    print(f"\nProcessing sequence {seq}, {len(fs_files)} depth files found.")

    for fs_file in fs_files:
        depth = np.load(fs_file).astype(np.float32)
        mask = np.isfinite(depth) & (depth > 0)
        H, W = depth.shape

        # 计算局部 std
        depth_masked = depth.copy()
        depth_masked[~mask] = 0
        kernel = np.ones((window_size, window_size), np.float32)
        mean_local = cv2.filter2D(depth_masked, -1, kernel/(window_size*window_size))
        sq_diff = (depth_masked - mean_local)**2
        std_local = np.sqrt(cv2.filter2D(sq_diff, -1, kernel/(window_size*window_size)))

        # 输出统计信息
        std_values = std_local[mask]
        if std_values.size > 0:
            mean_std = float(np.mean(std_values))
            max_std = float(np.max(std_values))
            min_std = float(np.min(std_values))
        else:
            mean_std = max_std = min_std = 0.0

        print(f"{seq}/{fs_file.name}: mean_std={mean_std:.4f}, min_std={min_std:.4f}, max_std={max_std:.4f}")

        # 可选：保存局部 std 可视化
        std_image = (std_local * 255 / np.max(std_local)).astype(np.uint8)
        color_std = cv2.applyColorMap(std_image, cv2.COLORMAP_TURBO)
        imageio.imwrite(out_dir / f"{fs_file.stem}_std.png", color_std)