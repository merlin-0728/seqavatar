import argparse
import csv
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


DEFAULT_SEQUENCES = ["0007_04", "0019_10", "0044_11", "0051_09", "0206_04", "0813_05"]
DATA_ROOT = Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering")
DETAIL_COLUMNS = [
    ("sequence", "Sequence"),
    ("frame", "Frame"),
    ("view", "View"),
    ("file", "File"),
    ("valid_pixels", "Valid Pixels"),
    ("valid_ratio", "Valid Ratio"),
    ("depth_min", "Depth Min"),
    ("depth_mean", "Depth Mean"),
    ("depth_median", "Depth Median"),
    ("depth_max", "Depth Max"),
    ("mean_std", "Mean STD"),
    ("min_std", "Min STD"),
    ("max_std", "Max STD"),
]
SUMMARY_COLUMNS = [
    ("sequence", "Sequence"),
    ("files", "Files"),
    ("valid_files", "Valid Files"),
    ("valid_pixels", "Valid Pixels"),
    ("mean_std_avg", "Mean STD Avg"),
    ("mean_std_median", "Mean STD Median"),
    ("mean_std_min", "Mean STD Min"),
    ("mean_std_max", "Mean STD Max"),
    ("std_min", "Pixel STD Min"),
    ("std_max", "Pixel STD Max"),
]


def compute_local_std(depth, mask, window_size):
    depth_masked = depth.copy()
    depth_masked[~mask] = 0
    kernel = np.ones((window_size, window_size), np.float32)
    mean_local = cv2.filter2D(depth_masked, -1, kernel / (window_size * window_size))
    sq_diff = (depth_masked - mean_local) ** 2
    return np.sqrt(cv2.filter2D(sq_diff, -1, kernel / (window_size * window_size)))


def save_std_visualization(std_local, out_path):
    max_value = float(np.max(std_local))
    if max_value > 0:
        std_image = np.clip(std_local * 255 / max_value, 0, 255).astype(np.uint8)
    else:
        std_image = np.zeros(std_local.shape, dtype=np.uint8)
    color_std = cv2.applyColorMap(std_image, cv2.COLORMAP_TURBO)
    imageio.imwrite(out_path, color_std)


def parse_frame_view(stem):
    match = re.match(r"frame_(\d+)_view_(\d+)$", stem)
    if match:
        return match.group(1), match.group(2)
    return "", ""


def analyze_sequence(seq, split, window_size, save_vis):
    fs_depth_dir = DATA_ROOT / seq / "render_depth" / split / "npy"
    out_dir = DATA_ROOT / seq / "render_depth" / split / "std_analysis"
    if save_vis:
        out_dir.mkdir(parents=True, exist_ok=True)

    fs_files = sorted(fs_depth_dir.glob("*.npy"))
    detail_rows = []
    mean_std_values = []
    min_std_values = []
    max_std_values = []
    valid_pixels_total = 0

    print(f"\nProcessing sequence {seq}, {len(fs_files)} depth files found.")

    for fs_file in fs_files:
        depth = np.load(fs_file).astype(np.float32)
        mask = np.isfinite(depth) & (depth > 0)
        std_local = compute_local_std(depth, mask, window_size)

        std_values = std_local[mask]
        depth_values = depth[mask]
        valid_pixels = int(std_values.size)
        valid_pixels_total += valid_pixels
        if valid_pixels > 0:
            depth_min = float(np.min(depth_values))
            depth_mean = float(np.mean(depth_values))
            depth_median = float(np.median(depth_values))
            depth_max = float(np.max(depth_values))
            mean_std = float(np.mean(std_values))
            max_std = float(np.max(std_values))
            min_std = float(np.min(std_values))
            mean_std_values.append(mean_std)
            min_std_values.append(min_std)
            max_std_values.append(max_std)
        else:
            depth_min = depth_mean = depth_median = depth_max = 0.0
            mean_std = max_std = min_std = 0.0

        frame, view = parse_frame_view(fs_file.stem)
        detail_rows.append(
            {
                "sequence": seq,
                "split": split,
                "frame": frame,
                "view": view,
                "file": fs_file.name,
                "valid_pixels": valid_pixels,
                "valid_ratio": float(valid_pixels / depth.size) if depth.size else 0.0,
                "depth_min": depth_min,
                "depth_mean": depth_mean,
                "depth_median": depth_median,
                "depth_max": depth_max,
                "mean_std": mean_std,
                "min_std": min_std,
                "max_std": max_std,
            }
        )

        if save_vis:
            save_std_visualization(std_local, out_dir / f"{fs_file.stem}_std.png")

    if mean_std_values:
        mean_std_array = np.asarray(mean_std_values, dtype=np.float64)
        return {
            "sequence": seq,
            "split": split,
            "files": len(fs_files),
            "valid_files": int(mean_std_array.size),
            "valid_pixels": int(valid_pixels_total),
            "mean_std_avg": float(np.mean(mean_std_array)),
            "mean_std_median": float(np.median(mean_std_array)),
            "mean_std_min": float(np.min(mean_std_array)),
            "mean_std_max": float(np.max(mean_std_array)),
            "std_min": float(np.min(min_std_values)),
            "std_max": float(np.max(max_std_values)),
        }, detail_rows

    return {
        "sequence": seq,
        "split": split,
        "files": len(fs_files),
        "valid_files": 0,
        "valid_pixels": 0,
        "mean_std_avg": 0.0,
        "mean_std_median": 0.0,
        "mean_std_min": 0.0,
        "mean_std_max": 0.0,
        "std_min": 0.0,
        "std_max": 0.0,
    }, detail_rows


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def make_markdown_table(rows, columns, title):
    widths = []
    for key, column_title in columns:
        width = len(column_title)
        for row in rows:
            width = max(width, len(format_value(row[key])))
        widths.append(width)

    header = "| " + " | ".join(column_title.ljust(widths[i]) for i, (_, column_title) in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    lines = [f"\n{title}", header, divider]
    for row in rows:
        values = [format_value(row[key]).ljust(widths[i]) for i, (key, _) in enumerate(columns)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def print_markdown_table(rows, columns, title):
    print(make_markdown_table(rows, columns, title))


def save_markdown_table(rows, columns, title, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(make_markdown_table(rows, columns, title).lstrip() + "\n", encoding="utf-8")


def save_csv(rows, output_path):
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze local STD of FoundationStereo depth and print per-frame tables.")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--split", default="train", choices=["train", "novelview"])
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--save_vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--print_table",
        choices=["detail", "summary", "both", "none"],
        default="detail",
        help="Which table to print to stdout. Detail keeps every depth file/frame.",
    )
    parser.add_argument(
        "--detail_csv",
        type=Path,
        default=None,
        help="Detailed per-depth-file CSV path. Defaults to render_depth/<split>_fs_depth_std_details.csv under DATA_ROOT.",
    )
    parser.add_argument(
        "--detail_md",
        type=Path,
        default=None,
        help="Detailed per-depth-file Markdown table path. Defaults to render_depth/<split>_fs_depth_std_details.md under DATA_ROOT.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Summary CSV path. Defaults to render_depth/<split>_fs_depth_std_summary.csv under DATA_ROOT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_rows = []
    detail_rows = []
    for seq in args.sequences:
        summary_row, seq_detail_rows = analyze_sequence(seq, args.split, args.window_size, args.save_vis)
        summary_rows.append(summary_row)
        detail_rows.extend(seq_detail_rows)

    if args.print_table in ("detail", "both"):
        print_markdown_table(detail_rows, DETAIL_COLUMNS, "Detailed Per-Frame Table")
    if args.print_table in ("summary", "both"):
        print_markdown_table(summary_rows, SUMMARY_COLUMNS, "Summary Table")

    detail_csv_path = args.detail_csv
    if detail_csv_path is None:
        detail_csv_path = DATA_ROOT / "render_depth" / f"{args.split}_fs_depth_std_details.csv"
    save_csv(detail_rows, detail_csv_path)

    detail_md_path = args.detail_md
    if detail_md_path is None:
        detail_md_path = DATA_ROOT / "render_depth" / f"{args.split}_fs_depth_std_details.md"
    save_markdown_table(detail_rows, DETAIL_COLUMNS, "Detailed Per-Frame Table", detail_md_path)

    summary_csv_path = args.summary_csv
    if summary_csv_path is None:
        summary_csv_path = DATA_ROOT / "render_depth" / f"{args.split}_fs_depth_std_summary.csv"
    save_csv(summary_rows, summary_csv_path)

    print(f"\nDetailed CSV saved to: {detail_csv_path}")
    print(f"Detailed Markdown saved to: {detail_md_path}")
    print(f"Summary CSV saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()
