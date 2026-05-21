import numpy as np
import cv2
from pathlib import Path
import shutil

# 所有序列列表
sequences = ["0007_04", "0019_10", "0044_11", "0051_09", "0206_04", "0813_05"]

# 根路径
data_root = Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering")

# 局部 std 计算参数
window_size = 5  # 局部窗口大小

# 保留比例：保留 Mean_STD 从小到大排序后的前 90%
keep_ratio = 0.70

for seq in sequences:
    fs_depth_dir = data_root / seq / "render_depth/train/npy"
    out_dir = data_root / seq / "render_depth/train/filtered_npy"

    # 如果你希望每次重新筛选时清空旧结果，保留下面两行
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_files = sorted(fs_depth_dir.glob("*.npy"))
    print(f"\nProcessing sequence {seq}, {len(fs_files)} depth files found.")

    if len(fs_files) == 0:
        print(f"[WARNING] No depth files found in {fs_depth_dir}")
        continue

    depth_stats = []

    # -------------------- 第一步：计算每张深度图的 Mean_STD --------------------
    for fs_file in fs_files:
        depth = np.load(fs_file).astype(np.float32)
        mask = np.isfinite(depth) & (depth > 0)

        if np.sum(mask) == 0:
            mean_std = float("inf")
            depth_stats.append({
                "file": fs_file,
                "mean_std": mean_std,
                "valid_pixels": 0,
            })
            print(f"{seq}/{fs_file.name}: Mean_STD=inf, valid_pixels=0")
            continue

        # 计算局部 std
        depth_masked = depth.copy()
        depth_masked[~mask] = 0

        kernel = np.ones((window_size, window_size), np.float32)
        mean_local = cv2.filter2D(
            depth_masked,
            -1,
            kernel / (window_size * window_size)
        )

        sq_diff = (depth_masked - mean_local) ** 2
        std_local = np.sqrt(
            cv2.filter2D(
                sq_diff,
                -1,
                kernel / (window_size * window_size)
            )
        )

        std_values = std_local[mask]
        mean_std = float(np.mean(std_values)) if std_values.size > 0 else float("inf")

        depth_stats.append({
            "file": fs_file,
            "mean_std": mean_std,
            "valid_pixels": int(np.sum(mask)),
        })

        print(f"{seq}/{fs_file.name}: Mean_STD={mean_std:.6f}, valid_pixels={int(np.sum(mask))}")

    # -------------------- 第二步：按 Mean_STD 从小到大排序 --------------------
    depth_stats_sorted = sorted(depth_stats, key=lambda x: x["mean_std"])

    total_num = len(depth_stats_sorted)
    keep_num = int(np.floor(total_num * keep_ratio))

    # 至少保留 1 张，避免极端情况下空目录
    keep_num = max(1, keep_num)

    kept_items = depth_stats_sorted[:keep_num]
    discarded_items = depth_stats_sorted[keep_num:]

    if kept_items:
        threshold_value = kept_items[-1]["mean_std"]
    else:
        threshold_value = None

    print(f"\n[{seq}] Total depth files: {total_num}")
    print(f"[{seq}] Keep ratio: {keep_ratio}")
    print(f"[{seq}] Kept files: {len(kept_items)}")
    print(f"[{seq}] Discarded files: {len(discarded_items)}")
    print(f"[{seq}] Auto Mean_STD threshold: {threshold_value}")

    # -------------------- 第三步：保存保留的深度图 --------------------
    kept_names = set()

    for item in kept_items:
        fs_file = item["file"]
        mean_std = item["mean_std"]

        depth = np.load(fs_file).astype(np.float32)
        np.save(out_dir / fs_file.name, depth)

        kept_names.add(fs_file.name)
        print(f"{seq}/{fs_file.name}: retained (Mean_STD={mean_std:.6f})")

    for item in discarded_items:
        fs_file = item["file"]
        mean_std = item["mean_std"]
        print(f"{seq}/{fs_file.name}: discarded (Mean_STD={mean_std:.6f})")

    # -------------------- 第四步：保存筛选统计文件 --------------------
    stat_file = out_dir / "mean_std_filter_summary.csv"
    with stat_file.open("w", encoding="utf-8") as f:
        f.write("sequence,file,mean_std,valid_pixels,status\n")

        for item in depth_stats_sorted:
            fs_file = item["file"]
            mean_std = item["mean_std"]
            valid_pixels = item["valid_pixels"]
            status = "retained" if fs_file.name in kept_names else "discarded"

            f.write(
                f"{seq},{fs_file.name},{mean_std},{valid_pixels},{status}\n"
            )

    print(f"[{seq}] Filtered depth saved to: {out_dir}")
    print(f"[{seq}] Summary saved to: {stat_file}")