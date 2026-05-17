#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


SEQAVATAR_ROOT = Path("/media/image/mxz/human/SeqAvatar")
DATA_ROOT = SEQAVATAR_ROOT / "DNA-Rendering"
OUT_ROOT = SEQAVATAR_ROOT / "output" / "DNA-Rendering"


def read_k_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        lines = [x.strip() for x in f if x.strip()]
    K = np.array(list(map(float, lines[0].split())), dtype=np.float32).reshape(3, 3)
    baseline = float(lines[1])
    return K, baseline


def safe_stats(x):
    if x.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "cv": None,
        }
    mean = float(np.mean(x))
    std = float(np.std(x))
    return {
        "count": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": mean,
        "std": std,
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "cv": float(std / (abs(mean) + 1e-8)),
    }


def colorize_scalar(arr, valid, cmap=cv2.COLORMAP_TURBO, invert=False):
    out = np.zeros(arr.shape, dtype=np.uint8)
    if np.any(valid):
        vals = arr[valid]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((arr - lo) / (hi - lo), 0, 1)
        if invert:
            norm = 1.0 - norm
        out[valid] = (norm[valid] * 255).astype(np.uint8)
    color = cv2.applyColorMap(out, cmap)
    color[~valid] = (0, 0, 0)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.45):
    if rgb is None:
        base = np.zeros((*mask.shape, 3), dtype=np.uint8)
    else:
        base = rgb.copy()
    if base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)
    color_img = np.zeros_like(base)
    color_img[..., 0] = color[0]
    color_img[..., 1] = color[1]
    color_img[..., 2] = color[2]
    out = base.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * color_img[mask]).astype(np.uint8)
    return out


def load_rgb(path: Path):
    if not path.exists():
        return None
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


def analyze_frame(
    name,
    depth_3dgs_path,
    depth_fs_path,
    k_path,
    left_path,
    right_path,
    out_dirs,
    args,
):
    d3 = np.load(depth_3dgs_path).astype(np.float32)
    dfs = np.load(depth_fs_path).astype(np.float32)

    if d3.shape != dfs.shape:
        raise RuntimeError(f"Shape mismatch for {name}: 3DGS {d3.shape}, FS {dfs.shape}")

    H, W = d3.shape
    K, baseline = read_k_file(k_path)
    fx = float(K[0, 0])

    left = load_rgb(left_path)
    right = load_rgb(right_path)

    shape_ok = True
    if left is not None and left.shape[:2] != d3.shape:
        shape_ok = False
    if right is not None and right.shape[:2] != d3.shape:
        shape_ok = False

    # 3DGS depth 在 render_dna_stereo_depth_batch.py 中已经按 alpha/bound_mask 置零，
    # 因此 d3>0 可以作为人体/有效渲染区域的近似 mask。
    mask_human = np.isfinite(d3) & (d3 > 0)

    mask_fs_valid = np.isfinite(dfs) & (dfs > 0)
    valid_human_fs = mask_human & mask_fs_valid
    valid_bg_fs = (~mask_human) & mask_fs_valid

    error_human = np.zeros_like(dfs, dtype=np.float32)
    error_human[valid_human_fs] = np.abs(d3[valid_human_fs] - dfs[valid_human_fs])

    high_error = valid_human_fs & (error_human > args.error_threshold)

    # 由 FS depth 反推 disparity，检查是否异常小或异常大。
    disp_from_depth = np.zeros_like(dfs, dtype=np.float32)
    disp_valid = valid_human_fs & (dfs > 1e-8)
    disp_from_depth[disp_valid] = fx * baseline / dfs[disp_valid]

    human_depth_stats = safe_stats(dfs[valid_human_fs])
    human_3dgs_stats = safe_stats(d3[mask_human])
    bg_depth_stats = safe_stats(dfs[valid_bg_fs])
    disp_stats = safe_stats(disp_from_depth[disp_valid])
    error_stats = safe_stats(error_human[valid_human_fs])

    human_pixels = int(mask_human.sum())
    fs_valid_human_pixels = int(valid_human_fs.sum())
    fs_valid_bg_pixels = int(valid_bg_fs.sum())

    human_ratio = human_pixels / float(H * W)
    fs_human_valid_ratio = fs_valid_human_pixels / float(human_pixels + 1e-8)
    fs_bg_valid_ratio = fs_valid_bg_pixels / float((H * W - human_pixels) + 1e-8)
    high_error_ratio = int(high_error.sum()) / float(fs_valid_human_pixels + 1e-8)

    # 诊断规则
    flags = []
    if not shape_ok:
        flags.append("left/right image shape is inconsistent with depth shape")
    if human_pixels == 0:
        flags.append("3DGS human mask is empty")
    if fs_human_valid_ratio < args.min_human_valid_ratio:
        flags.append("FS valid depth inside human mask is too sparse")
    if human_depth_stats["cv"] is not None and human_depth_stats["cv"] < args.constant_cv_threshold:
        flags.append("FS depth inside human mask is nearly constant")
    if bg_depth_stats["count"] > 0 and fs_bg_valid_ratio > args.max_bg_valid_ratio:
        flags.append("FS produces many valid depths in background")
    if disp_stats["p50"] is not None and disp_stats["p50"] < args.min_disp_px:
        flags.append("median disparity inside human mask is too small")
    if error_stats["mean"] is not None and error_stats["mean"] > args.large_error_m:
        flags.append("large human-region depth disagreement between 3DGS and FS")

    # 保存诊断可视化
    imageio.imwrite(out_dirs["mask"] / f"{name}_human_mask.png", (mask_human.astype(np.uint8) * 255))
    imageio.imwrite(out_dirs["mask"] / f"{name}_fs_valid_human_mask.png", (valid_human_fs.astype(np.uint8) * 255))
    imageio.imwrite(out_dirs["mask"] / f"{name}_high_error_mask.png", (high_error.astype(np.uint8) * 255))

    imageio.imwrite(
        out_dirs["vis"] / f"{name}_fs_depth_human.png",
        colorize_scalar(dfs, valid_human_fs, cv2.COLORMAP_TURBO, invert=True),
    )
    imageio.imwrite(
        out_dirs["vis"] / f"{name}_3dgs_depth_human.png",
        colorize_scalar(d3, mask_human, cv2.COLORMAP_TURBO, invert=True),
    )
    imageio.imwrite(
        out_dirs["vis"] / f"{name}_human_error_heatmap.png",
        colorize_scalar(error_human, valid_human_fs, cv2.COLORMAP_HOT, invert=False),
    )
    imageio.imwrite(
        out_dirs["vis"] / f"{name}_disp_from_fs_depth.png",
        colorize_scalar(disp_from_depth, disp_valid, cv2.COLORMAP_TURBO, invert=False),
    )

    if left is not None:
        imageio.imwrite(out_dirs["overlay"] / f"{name}_human_mask_overlay.png", overlay_mask(left, mask_human))
        imageio.imwrite(out_dirs["overlay"] / f"{name}_high_error_overlay.png", overlay_mask(left, high_error, color=(255, 255, 0)))

    np.save(out_dirs["npy"] / f"{name}_human_mask.npy", mask_human)
    np.save(out_dirs["npy"] / f"{name}_valid_human_fs.npy", valid_human_fs)
    np.save(out_dirs["npy"] / f"{name}_human_error.npy", error_human)
    np.save(out_dirs["npy"] / f"{name}_disp_from_fs_depth.npy", disp_from_depth)

    return {
        "file": f"{name}.npy",
        "shape": [int(H), int(W)],
        "fx": fx,
        "baseline": float(baseline),
        "shape_ok": bool(shape_ok),
        "human_pixels": human_pixels,
        "human_ratio": human_ratio,
        "fs_valid_human_pixels": fs_valid_human_pixels,
        "fs_valid_human_ratio": fs_human_valid_ratio,
        "fs_valid_bg_pixels": fs_valid_bg_pixels,
        "fs_valid_bg_ratio": fs_bg_valid_ratio,
        "high_error_pixels": int(high_error.sum()),
        "high_error_ratio_in_valid_human": high_error_ratio,
        "fs_depth_human": human_depth_stats,
        "depth_3dgs_human": human_3dgs_stats,
        "fs_depth_background": bg_depth_stats,
        "disp_from_fs_depth_human": disp_stats,
        "human_abs_error": error_stats,
        "flags": flags,
    }


def main():
    parser = argparse.ArgumentParser(description="Check FoundationStereo novelview depth quality inside human mask")
    parser.add_argument("--sequence", type=str, default="0007_04")
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=OUT_ROOT)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--error_threshold", type=float, default=0.02)
    parser.add_argument("--constant_cv_threshold", type=float, default=0.03)
    parser.add_argument("--min_human_valid_ratio", type=float, default=0.5)
    parser.add_argument("--max_bg_valid_ratio", type=float, default=0.05)
    parser.add_argument("--min_disp_px", type=float, default=1.0)
    parser.add_argument("--large_error_m", type=float, default=1.0)
    args = parser.parse_args()

    seq = args.sequence
    data_dir = args.data_root / seq
    model_out_dir = args.output_root / seq

    depth_3dgs_dir = data_dir / "depth" / "novelview" / "npy"
    depth_fs_dir = data_dir / "render_depth" / "novelview" / "npy"
    k_dir = data_dir / "render_depth" / "novelview" / "K"
    left_dir = data_dir / "render_depth" / "novelview" / "rgb"
    right_dir = data_dir / "render_depth" / "novelview" / "right"

    out_root = model_out_dir / "depth_map" / "fs_depth_diagnosis"
    out_dirs = {
        "vis": out_root / "vis",
        "mask": out_root / "mask",
        "overlay": out_root / "overlay",
        "npy": out_root / "npy",
    }
    for p in out_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    files = sorted(depth_fs_dir.glob("*.npy"))
    if args.max_frames is not None:
        files = files[: args.max_frames]

    records = []
    for fs_file in files:
        name = fs_file.stem
        d3_file = depth_3dgs_dir / f"{name}.npy"
        k_file = k_dir / f"{name}.txt"
        left_file = left_dir / f"{name}.png"
        right_file = right_dir / f"{name}.png"

        if not d3_file.exists():
            print(f"[SKIP] missing 3DGS depth: {d3_file}")
            continue
        if not k_file.exists():
            print(f"[SKIP] missing K file: {k_file}")
            continue

        try:
            rec = analyze_frame(
                name=name,
                depth_3dgs_path=d3_file,
                depth_fs_path=fs_file,
                k_path=k_file,
                left_path=left_file,
                right_path=right_file,
                out_dirs=out_dirs,
                args=args,
            )
            records.append(rec)
            flag_msg = "; ".join(rec["flags"]) if rec["flags"] else "OK"
            print(
                f"[{name}] "
                f"human_valid={rec['fs_valid_human_ratio']:.3f}, "
                f"human_std={rec['fs_depth_human']['std']}, "
                f"bg_valid={rec['fs_valid_bg_ratio']:.3f}, "
                f"err_mean={rec['human_abs_error']['mean']}, "
                f"flags={flag_msg}"
            )
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    if not records:
        raise RuntimeError("No frames processed")

    def mean_of(path):
        vals = []
        for r in records:
            cur = r
            for key in path:
                cur = cur.get(key, None) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None:
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else None

    summary = {
        "sequence": seq,
        "num_frames": len(records),
        "diagnosis_output_dir": str(out_root),
        "mean_human_ratio": mean_of(["human_ratio"]),
        "mean_fs_valid_human_ratio": mean_of(["fs_valid_human_ratio"]),
        "mean_fs_valid_bg_ratio": mean_of(["fs_valid_bg_ratio"]),
        "mean_high_error_ratio_in_valid_human": mean_of(["high_error_ratio_in_valid_human"]),
        "mean_fs_human_depth_std": mean_of(["fs_depth_human", "std"]),
        "mean_fs_human_depth_cv": mean_of(["fs_depth_human", "cv"]),
        "mean_human_abs_error": mean_of(["human_abs_error", "mean"]),
        "mean_disp_median": mean_of(["disp_from_fs_depth_human", "p50"]),
        "flag_counts": {},
        "frames": records,
    }

    for r in records:
        for flag in r["flags"]:
            summary["flag_counts"][flag] = summary["flag_counts"].get(flag, 0) + 1

    summary_path = out_root / "fs_depth_diagnosis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_root / "fs_depth_diagnosis_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "human_ratio",
            "fs_valid_human_ratio",
            "fs_valid_bg_ratio",
            "fs_human_min",
            "fs_human_max",
            "fs_human_mean",
            "fs_human_std",
            "fs_human_cv",
            "disp_p50",
            "error_mean",
            "error_std",
            "high_error_ratio",
            "flags",
        ])
        for r in records:
            writer.writerow([
                r["file"],
                r["human_ratio"],
                r["fs_valid_human_ratio"],
                r["fs_valid_bg_ratio"],
                r["fs_depth_human"]["min"],
                r["fs_depth_human"]["max"],
                r["fs_depth_human"]["mean"],
                r["fs_depth_human"]["std"],
                r["fs_depth_human"]["cv"],
                r["disp_from_fs_depth_human"]["p50"],
                r["human_abs_error"]["mean"],
                r["human_abs_error"]["std"],
                r["high_error_ratio_in_valid_human"],
                " | ".join(r["flags"]),
            ])

    print("\n[DONE]")
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")
    print(f"Visuals: {out_root}")
    print("\nKey means:")
    print(f"  mean_fs_valid_human_ratio = {summary['mean_fs_valid_human_ratio']}")
    print(f"  mean_fs_valid_bg_ratio    = {summary['mean_fs_valid_bg_ratio']}")
    print(f"  mean_fs_human_depth_std   = {summary['mean_fs_human_depth_std']}")
    print(f"  mean_fs_human_depth_cv    = {summary['mean_fs_human_depth_cv']}")
    print(f"  mean_human_abs_error      = {summary['mean_human_abs_error']}")
    print(f"  mean_disp_median          = {summary['mean_disp_median']}")
    print(f"  flag_counts               = {summary['flag_counts']}")


if __name__ == "__main__":
    main()