#!/usr/bin/env python3
"""Validate DNA train-view optical-flow direction against SMPL vertex reprojection.

This script is diagnostic-only. It reads training-view RGB frames, cameras, and
SMPL observed vertices, then checks whether forward flow from t-1 to t moves
the projected t-1 vertex closer to the same vertex projected at t.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np


TRAIN_VIEWS = list(range(0, 48, 2))


def parse_list(raw: str) -> List[str]:
    return [item for item in re.split(r"[\s,]+", raw.strip()) if item]


def load_gray_scaled(path: Path, scale: float) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def project_points_to_image(points: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    w2c = np.linalg.inv(c2w)
    pts_cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pts_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    u = K[0, 0] * (pts_cam[:, 0] / safe_z) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / safe_z) + K[1, 2]
    return u, v, z


def sample_flow(flow: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    map_x = u.astype(np.float32).reshape(-1, 1)
    map_y = v.astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(
        flow,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled.reshape(-1, flow.shape[-1])


def load_c2w(cam_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cam = np.load(cam_path, allow_pickle=True)
    K = cam["K"]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = cam["RT"][:3, :3]
    c2w[:3, 3] = cam["RT"][:3, 3]
    return K, c2w


@dataclass
class RunningStats:
    count: int = 0
    flow_sum: float = 0.0
    no_sum: float = 0.0
    minus_sum: float = 0.0
    fb_sum: float = 0.0
    mag_sum: float = 0.0
    flow_sq_sum: float = 0.0
    no_sq_sum: float = 0.0
    minus_sq_sum: float = 0.0
    better_no_count: int = 0
    better_minus_count: int = 0
    samples_flow: list = None
    samples_no: list = None
    samples_minus: list = None
    samples_fb: list = None

    def __post_init__(self):
        self.samples_flow = [] if self.samples_flow is None else self.samples_flow
        self.samples_no = [] if self.samples_no is None else self.samples_no
        self.samples_minus = [] if self.samples_minus is None else self.samples_minus
        self.samples_fb = [] if self.samples_fb is None else self.samples_fb

    def update(self, flow_err, no_err, minus_err, fb_err, mag, sample_limit: int):
        n = int(flow_err.size)
        if n == 0:
            return
        self.count += n
        self.flow_sum += float(flow_err.sum())
        self.no_sum += float(no_err.sum())
        self.minus_sum += float(minus_err.sum())
        self.fb_sum += float(fb_err.sum())
        self.mag_sum += float(mag.sum())
        self.flow_sq_sum += float((flow_err * flow_err).sum())
        self.no_sq_sum += float((no_err * no_err).sum())
        self.minus_sq_sum += float((minus_err * minus_err).sum())
        self.better_no_count += int(np.count_nonzero(flow_err < no_err))
        self.better_minus_count += int(np.count_nonzero(flow_err < minus_err))
        if sample_limit > 0:
            step = max(1, int(math.ceil(n / float(sample_limit))))
            self.samples_flow.extend(flow_err[::step][:sample_limit].astype(np.float32).tolist())
            self.samples_no.extend(no_err[::step][:sample_limit].astype(np.float32).tolist())
            self.samples_minus.extend(minus_err[::step][:sample_limit].astype(np.float32).tolist())
            self.samples_fb.extend(fb_err[::step][:sample_limit].astype(np.float32).tolist())

    def finalize(self) -> Dict[str, float]:
        if self.count <= 0:
            return {"count": 0}
        flow_samples = np.asarray(self.samples_flow, dtype=np.float32)
        no_samples = np.asarray(self.samples_no, dtype=np.float32)
        minus_samples = np.asarray(self.samples_minus, dtype=np.float32)
        fb_samples = np.asarray(self.samples_fb, dtype=np.float32)

        def pct(arr, q):
            return None if arr.size == 0 else float(np.percentile(arr, q))

        return {
            "count": int(self.count),
            "flow_endpoint_mean_px": self.flow_sum / self.count,
            "no_flow_mean_px": self.no_sum / self.count,
            "minus_flow_mean_px": self.minus_sum / self.count,
            "fb_error_mean_px": self.fb_sum / self.count,
            "flow_mag_mean_px": self.mag_sum / self.count,
            "flow_endpoint_rmse_px": math.sqrt(self.flow_sq_sum / self.count),
            "no_flow_rmse_px": math.sqrt(self.no_sq_sum / self.count),
            "minus_flow_rmse_px": math.sqrt(self.minus_sq_sum / self.count),
            "flow_over_no_mean_ratio": self.flow_sum / max(self.no_sum, 1e-12),
            "flow_over_minus_mean_ratio": self.flow_sum / max(self.minus_sum, 1e-12),
            "flow_better_than_no_ratio": self.better_no_count / self.count,
            "flow_better_than_minus_ratio": self.better_minus_count / self.count,
            "flow_endpoint_p50_px": pct(flow_samples, 50),
            "flow_endpoint_p90_px": pct(flow_samples, 90),
            "flow_endpoint_p95_px": pct(flow_samples, 95),
            "no_flow_p50_px": pct(no_samples, 50),
            "minus_flow_p50_px": pct(minus_samples, 50),
            "fb_error_p50_px": pct(fb_samples, 50),
            "sample_count": int(flow_samples.size),
        }


def validate_sequence(
    data_root: Path,
    sequence: str,
    views: Sequence[int],
    pose_start: int,
    pose_end: int,
    scale: float,
    max_vertices: int,
    sample_limit: int,
) -> Dict[str, float]:
    seq_root = data_root / sequence
    stats = RunningStats()
    missing = 0
    checked_pairs = 0

    for pose_id in range(max(1, pose_start), pose_end + 1):
        prev_model_path = seq_root / "model" / f"{pose_id - 1:06d}.npz"
        curr_model_path = seq_root / "model" / f"{pose_id:06d}.npz"
        if not (prev_model_path.exists() and curr_model_path.exists()):
            missing += 1
            continue
        prev_xyz = np.load(prev_model_path, allow_pickle=True)["obs_xyz"].astype(np.float32, copy=False)
        curr_xyz = np.load(curr_model_path, allow_pickle=True)["obs_xyz"].astype(np.float32, copy=False)

        for view_id in views:
            prev_img = seq_root / "images" / f"{view_id:02d}" / f"{pose_id - 1:06d}.png"
            curr_img = seq_root / "images" / f"{view_id:02d}" / f"{pose_id:06d}.png"
            prev_cam = seq_root / "cameras" / f"{view_id:02d}" / f"{pose_id - 1:06d}.npz"
            curr_cam = seq_root / "cameras" / f"{view_id:02d}" / f"{pose_id:06d}.npz"
            if not (prev_img.exists() and curr_img.exists() and prev_cam.exists() and curr_cam.exists()):
                missing += 1
                continue

            prev_gray = load_gray_scaled(prev_img, scale)
            curr_gray = load_gray_scaled(curr_img, scale)
            flow_fwd = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            flow_bwd = cv2.calcOpticalFlowFarneback(
                curr_gray,
                prev_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            prev_K, prev_c2w = load_c2w(prev_cam)
            curr_K, curr_c2w = load_c2w(curr_cam)
            prev_u, prev_v, prev_z = project_points_to_image(prev_xyz, prev_K, prev_c2w)
            curr_u, curr_v, curr_z = project_points_to_image(curr_xyz, curr_K, curr_c2w)
            prev_u_s, prev_v_s = prev_u * scale, prev_v * scale
            curr_u_s, curr_v_s = curr_u * scale, curr_v * scale
            h, w = flow_fwd.shape[:2]
            valid = (
                (prev_z > 1e-6)
                & (curr_z > 1e-6)
                & (prev_u_s >= 0)
                & (prev_u_s <= w - 1)
                & (prev_v_s >= 0)
                & (prev_v_s <= h - 1)
                & (curr_u_s >= 0)
                & (curr_u_s <= w - 1)
                & (curr_v_s >= 0)
                & (curr_v_s <= h - 1)
            )
            valid_idx = np.where(valid)[0]
            if valid_idx.size == 0:
                continue
            if max_vertices > 0 and valid_idx.size > max_vertices:
                step = int(math.ceil(valid_idx.size / float(max_vertices)))
                valid_idx = valid_idx[::step][:max_vertices]

            flow = sample_flow(flow_fwd, prev_u_s[valid_idx], prev_v_s[valid_idx])
            end_u = prev_u_s[valid_idx] + flow[:, 0]
            end_v = prev_v_s[valid_idx] + flow[:, 1]
            bwd = sample_flow(flow_bwd, end_u, end_v)

            target_u = curr_u_s[valid_idx]
            target_v = curr_v_s[valid_idx]
            flow_err = np.sqrt((end_u - target_u) ** 2 + (end_v - target_v) ** 2)
            no_err = np.sqrt((prev_u_s[valid_idx] - target_u) ** 2 + (prev_v_s[valid_idx] - target_v) ** 2)
            minus_err = np.sqrt((prev_u_s[valid_idx] - flow[:, 0] - target_u) ** 2 + (prev_v_s[valid_idx] - flow[:, 1] - target_v) ** 2)
            fb_err = np.linalg.norm(flow + bwd, axis=-1)
            mag = np.linalg.norm(flow, axis=-1)
            stats.update(flow_err, no_err, minus_err, fb_err, mag, sample_limit)
            checked_pairs += 1

    result = stats.finalize()
    result.update(
        {
            "sequence": sequence,
            "checked_pose_view_pairs": checked_pairs,
            "missing_items": missing,
            "scale": scale,
            "pose_start": pose_start,
            "pose_end": pose_end,
            "views": list(views),
            "max_vertices": max_vertices,
        }
    )
    return result


def weighted_aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    valid = [r for r in rows if int(r.get("count", 0)) > 0]
    total = sum(int(r["count"]) for r in valid)
    if total <= 0:
        return {"count": 0}
    keys = [
        "flow_endpoint_mean_px",
        "no_flow_mean_px",
        "minus_flow_mean_px",
        "fb_error_mean_px",
        "flow_mag_mean_px",
        "flow_better_than_no_ratio",
        "flow_better_than_minus_ratio",
        "flow_over_no_mean_ratio",
        "flow_over_minus_mean_ratio",
    ]
    out = {"count": total}
    for key in keys:
        out[key] = sum(float(r[key]) * int(r["count"]) for r in valid) / total
    return out


def write_markdown(path: Path, rows: Sequence[Dict[str, float]], aggregate: Dict[str, float], args):
    lines = [
        "# Flow Correctness Validation",
        "",
        f"- Data root: `{args.data_root}`",
        f"- Sequences: `{', '.join(args.sequences)}`",
        f"- Views: train views only `{','.join(map(str, args.views))}`",
        f"- Pose range: `{args.pose_start}-{args.pose_end}`",
        f"- Scale: `{args.scale}`",
        f"- Max vertices per pose/view: `{args.max_vertices}`",
        f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`",
        "",
        "| Sequence | Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {sequence} | {count} | {flow_endpoint_mean_px:.4f} | {no_flow_mean_px:.4f} | "
            "{minus_flow_mean_px:.4f} | {flow_over_no_mean_ratio:.4f} | {flow_over_minus_mean_ratio:.4f} | "
            "{flow_better_than_no_ratio:.4f} | {flow_better_than_minus_ratio:.4f} | {fb_error_mean_px:.4f} | {flow_mag_mean_px:.4f} |".format(
                **row
            )
        )
    if int(aggregate.get("count", 0)) > 0:
        lines.extend(
            [
                "",
                "## Weighted Aggregate",
                "",
                "| Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| {count} | {flow_endpoint_mean_px:.4f} | {no_flow_mean_px:.4f} | {minus_flow_mean_px:.4f} | "
                "{flow_over_no_mean_ratio:.4f} | {flow_over_minus_mean_ratio:.4f} | "
                "{flow_better_than_no_ratio:.4f} | {flow_better_than_minus_ratio:.4f} | "
                "{fb_error_mean_px:.4f} | {flow_mag_mean_px:.4f} |".format(**aggregate),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering"))
    parser.add_argument("--sequences", default="0007_04 0019_10 0044_11 0051_09 0206_04 0813_05")
    parser.add_argument("--views", default=",".join(map(str, TRAIN_VIEWS)))
    parser.add_argument("--pose-start", type=int, default=1)
    parser.add_argument("--pose-end", type=int, default=99)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--max-vertices", type=int, default=4096)
    parser.add_argument("--sample-limit-per-pair", type=int, default=512)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    args.sequences = parse_list(args.sequences)
    args.views = [int(v) for v in parse_list(args.views)]
    rows = []
    for sequence in args.sequences:
        print(f"[FLOW-CHECK] sequence={sequence} views={len(args.views)} poses={args.pose_start}-{args.pose_end}")
        row = validate_sequence(
            args.data_root,
            sequence,
            args.views,
            args.pose_start,
            args.pose_end,
            args.scale,
            args.max_vertices,
            args.sample_limit_per_pair,
        )
        rows.append(row)
        print(
            "[FLOW-CHECK] {sequence}: flow={flow_endpoint_mean_px:.4f}px no={no_flow_mean_px:.4f}px "
            "minus={minus_flow_mean_px:.4f}px better_no={flow_better_than_no_ratio:.4f}".format(**row)
        )

    aggregate = weighted_aggregate(rows)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {"rows": rows, "aggregate": aggregate, "args": serializable_args}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    write_markdown(args.output_md, rows, aggregate, args)
    print(f"[FLOW-CHECK] wrote {args.output_json}")
    print(f"[FLOW-CHECK] wrote {args.output_md}")


if __name__ == "__main__":
    main()
