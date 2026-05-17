import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


SEQAVATAR_ROOT = Path("/media/image/mxz/human/SeqAvatar")
DNA_ROOT = SEQAVATAR_ROOT / "DNA-Rendering"
DEFAULT_SEQUENCES = ["0007_04", "0019_10", "0044_11", "0051_09", "0206_04", "0813_05"]


def discover_sequences():
    sequences = []
    for seq_dir in sorted(DNA_ROOT.iterdir()):
        if not seq_dir.is_dir():
            continue
        depth_dir = seq_dir / "depth" / "novelview" / "npy"
        fs_dir = seq_dir / "render_depth" / "novelview" / "npy"
        if depth_dir.is_dir() or fs_dir.is_dir():
            sequences.append(seq_dir.name)
    return sequences


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def load_depth(path):
    return np.load(path).astype(np.float32)


def compute_valid_mask(depth_3dgs, depth_fs):
    return (
        np.isfinite(depth_3dgs)
        & np.isfinite(depth_fs)
        & (depth_3dgs > 0)
        & (depth_fs > 0)
    )


def compute_metrics(depth_3dgs, depth_fs, valid_mask, relative_reference="fs"):
    if not np.any(valid_mask):
        return {
            "rmse_m": None,
            "mae_m": None,
            "relative_error_pct": None,
            "num_valid_pixels": 0,
            "num_high_error_pixels": 0,
        }

    diff = depth_3dgs[valid_mask] - depth_fs[valid_mask]
    abs_diff = np.abs(diff)

    if relative_reference == "3dgs":
        denom = depth_3dgs[valid_mask]
    else:
        denom = depth_fs[valid_mask]
    rel = abs_diff / np.maximum(denom, 1e-8)

    return {
        "rmse_m": float(np.sqrt(np.mean(diff ** 2))),
        "mae_m": float(np.mean(abs_diff)),
        "relative_error_pct": float(np.mean(rel) * 100.0),
        "num_valid_pixels": int(valid_mask.sum()),
        "num_high_error_pixels": None,
    }


def build_error_maps(depth_3dgs, depth_fs, valid_mask, threshold_m):
    abs_error = np.zeros_like(depth_3dgs, dtype=np.float32)
    if np.any(valid_mask):
        abs_error[valid_mask] = np.abs(depth_3dgs[valid_mask] - depth_fs[valid_mask])

    mask_high_error = np.zeros_like(depth_3dgs, dtype=np.uint8)
    mask_high_error[valid_mask] = (abs_error[valid_mask] > threshold_m).astype(np.uint8)
    return abs_error, mask_high_error


def save_heatmap(abs_error, valid_mask, out_path):
    heat = np.zeros(abs_error.shape, dtype=np.uint8)
    if np.any(valid_mask):
        vals = abs_error[valid_mask]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((abs_error - lo) / (hi - lo), 0.0, 1.0)
        heat[valid_mask] = (norm[valid_mask] * 255).astype(np.uint8)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    color[~valid_mask] = (0, 0, 0)
    cv2.imwrite(str(out_path), color)


def process_sequence(seq, threshold_m, relative_reference, output_root, output_dir=None, save_error_npy=False, max_frames=None):
    seq_root = DNA_ROOT / seq
    depth_dir = seq_root / "depth" / "novelview" / "npy"
    fs_dir = seq_root / "render_depth" / "novelview" / "npy"

    if not depth_dir.exists() and not fs_dir.exists():
        print(f"[WARN] {seq}: neither depth directory exists, skip")
        return None

    if output_dir is not None:
        out_root = output_dir
    elif output_root is None:
        out_root = seq_root / "depth_quantify" / "novelview"
    else:
        out_root = output_root / seq / "novelview"
    heatmap_dir = out_root / "error_heatmap"
    mask_dir = out_root / "mask_high_error"
    valid_dir = out_root / "valid_mask"
    error_dir = out_root / "abs_error"
    ensure_dir(heatmap_dir)
    ensure_dir(mask_dir)
    ensure_dir(valid_dir)
    if save_error_npy:
        ensure_dir(error_dir)

    frame_names = []
    if depth_dir.exists():
        frame_names = [p.name for p in sorted(depth_dir.glob("*.npy"))]
    elif fs_dir.exists():
        frame_names = [p.name for p in sorted(fs_dir.glob("*.npy"))]
    if max_frames is not None:
        frame_names = frame_names[:max_frames]

    per_frame = []
    skipped = 0

    for file_name in frame_names:
        path_3dgs = depth_dir / file_name
        path_fs = fs_dir / file_name
        if not path_3dgs.exists() or not path_fs.exists():
            print(f"[WARN] {seq}: missing pair for {file_name}, skip")
            skipped += 1
            continue

        depth_3dgs = load_depth(path_3dgs)
        depth_fs = load_depth(path_fs)
        if depth_3dgs.shape != depth_fs.shape:
            print(f"[WARN] {seq}: shape mismatch for {file_name}: {depth_3dgs.shape} vs {depth_fs.shape}, skip")
            skipped += 1
            continue

        valid_mask = compute_valid_mask(depth_3dgs, depth_fs)
        abs_error, mask_high_error = build_error_maps(depth_3dgs, depth_fs, valid_mask, threshold_m)
        metrics = compute_metrics(depth_3dgs, depth_fs, valid_mask, relative_reference=relative_reference)
        metrics["num_high_error_pixels"] = int(mask_high_error.sum())
        metrics["file"] = file_name
        metrics["sequence"] = seq
        metrics["threshold_m"] = float(threshold_m)
        metrics["relative_reference"] = relative_reference
        metrics["valid_pixel_ratio"] = float(valid_mask.mean())

        base = Path(file_name).stem
        np.save(mask_dir / f"{base}_mask_high_error.npy", mask_high_error.astype(np.uint8))
        np.save(valid_dir / f"{base}_valid_mask.npy", valid_mask.astype(np.uint8))
        if save_error_npy:
            np.save(error_dir / f"{base}_abs_error.npy", abs_error.astype(np.float32))
        save_heatmap(abs_error, valid_mask, heatmap_dir / f"{base}_error_heatmap.png")

        per_frame.append(metrics)
        print(
            f"[INFO] {seq}/{file_name} | "
            f"RMSE={metrics['rmse_m']:.4f} m, "
            f"MAE={metrics['mae_m']:.4f} m, "
            f"RelErr={metrics['relative_error_pct']:.2f}%, "
            f"HighErrorPixels={metrics['num_high_error_pixels']}"
        )

    aggregate = {
        "sequence": seq,
        "threshold_m": float(threshold_m),
        "relative_reference": relative_reference,
        "num_frames_total": len(frame_names),
        "num_frames_processed": len(per_frame),
        "num_frames_skipped": skipped,
    }
    if per_frame:
        aggregate["rmse_m_mean"] = float(np.mean([x["rmse_m"] for x in per_frame]))
        aggregate["mae_m_mean"] = float(np.mean([x["mae_m"] for x in per_frame]))
        rel_vals = [x["relative_error_pct"] for x in per_frame if x["relative_error_pct"] is not None]
        aggregate["relative_error_pct_mean"] = float(np.mean(rel_vals)) if rel_vals else None
        aggregate["high_error_pixels_total"] = int(sum(x["num_high_error_pixels"] for x in per_frame))
    else:
        aggregate["rmse_m_mean"] = None
        aggregate["mae_m_mean"] = None
        aggregate["relative_error_pct_mean"] = None
        aggregate["high_error_pixels_total"] = 0

    summary = {
        "sequence": seq,
        "threshold_m": float(threshold_m),
        "relative_reference": relative_reference,
        "aggregate": aggregate,
        "frames": per_frame,
    }
    ensure_dir(out_root)
    with (out_root / "depth_quantify_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare SeqAvatar 3DGS depth with FoundationStereo depth on novelview")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--threshold_m", type=float, default=0.02)
    parser.add_argument("--relative_reference", choices=["fs", "3dgs"], default="fs")
    parser.add_argument("--save_error_npy", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    sequences = args.sequences if args.sequences else discover_sequences()
    if not sequences:
        sequences = DEFAULT_SEQUENCES

    output_root = Path(args.output_root) if args.output_root is not None else None
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None and len(sequences) != 1:
        raise ValueError("--output_dir is an exact output directory and can only be used with one sequence")

    all_summaries = []
    for seq in sequences:
        summary = process_sequence(
            seq=seq,
            threshold_m=args.threshold_m,
            relative_reference=args.relative_reference,
            output_root=output_root,
            output_dir=output_dir,
            save_error_npy=args.save_error_npy,
            max_frames=args.max_frames,
        )
        if summary is not None:
            all_summaries.append(summary)

    if output_dir is not None:
        top_level = output_dir / "depth_quantify_summary.json"
    elif output_root is None:
        top_level = DNA_ROOT / "depth_quantify_summary.json"
    else:
        top_level = Path(output_root) / "depth_quantify_summary.json"
    with top_level.open("w", encoding="utf-8") as f:
        json.dump({"sequences": all_summaries}, f, indent=2)
    print(f"[INFO] Summary saved to {top_level}")


if __name__ == "__main__":
    main()
