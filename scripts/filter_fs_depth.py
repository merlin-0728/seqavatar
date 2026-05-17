import numpy as np
import cv2
from pathlib import Path

# 所有序列列表
sequences = ["0007_04", "0019_10", "0044_11", "0051_09", "0206_04", "0813_05"]
#sequences = ["0019_10"]
# 每个序列对应的 Mean_STD 阈值
mean_std_thresholds = {
    "0007_04": 0.32,
    "0019_10": 0.10,
    "0044_11": 0.38,
    "0051_09": 0.4,
    "0206_04": 0.22,  # None 表示不删
    "0813_05": 0.13,
}

# 根路径
data_root = Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering")

# 局部 std 计算参数
window_size = 5  # 局部窗口大小

for seq in sequences:
    fs_depth_dir = data_root / seq / "render_depth/novelview/npy"
    out_dir = data_root / seq / "render_depth/novelview/filtered_npy4" #这里改输出的位置
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_files = sorted(fs_depth_dir.glob("*.npy"))
    print(f"\nProcessing sequence {seq}, {len(fs_files)} depth files found.")

    threshold = mean_std_thresholds.get(seq, None)

    for fs_file in fs_files:
        depth = np.load(fs_file).astype(np.float32)
        mask = np.isfinite(depth) & (depth > 0)

        # 计算局部 std
        depth_masked = depth.copy()
        depth_masked[~mask] = 0
        kernel = np.ones((window_size, window_size), np.float32)
        mean_local = cv2.filter2D(depth_masked, -1, kernel / (window_size * window_size))
        sq_diff = (depth_masked - mean_local) ** 2
        std_local = np.sqrt(cv2.filter2D(sq_diff, -1, kernel / (window_size * window_size)))

        # 计算 Mean_STD
        std_values = std_local[mask]
        if std_values.size > 0:
            mean_std = float(np.mean(std_values))
        else:
            mean_std = 0.0

        # 判断是否保留
        if threshold is None:
            # 不删
            np.save(out_dir / fs_file.name, depth)
            print(f"{seq}/{fs_file.name}: retained (no threshold)")
        else:
            if mean_std < threshold:
                np.save(out_dir / fs_file.name, depth)
                print(f"{seq}/{fs_file.name}: retained (Mean_STD={mean_std:.4f})")
            else:
                print(f"{seq}/{fs_file.name}: discarded (Mean_STD={mean_std:.4f})")