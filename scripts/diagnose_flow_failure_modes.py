#!/usr/bin/env python3
"""Diagnose why optical-flow conditions may not improve SeqAvatar.

This is a read-only diagnostic script. It measures four possible bottlenecks:
1) Multi-view vector averaging cancellation.
2) Gaussian KNN aggregation attenuation.
3) Farneback quality, including boundary vs non-boundary regions.
4) Existing trained flow_v2 metric response compared with flow_zero_v2.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree


TRAIN_VIEWS = list(range(0, 48, 2))
METRICS = ("PSNR", "SSIM", "LPIPS")


def parse_list(raw: str) -> List[str]:
    return [x for x in re.split(r"[\s,]+", raw.strip()) if x]


def load_gray_scaled(path: Path, scale: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def load_mask_scaled(path: Path, scale: float) -> np.ndarray:
    image = Image.open(path).convert("L")
    mask = np.asarray(image) != 0
    if scale != 1.0:
        h, w = mask.shape
        mask = cv2.resize(mask.astype(np.uint8), (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_NEAREST) != 0
    return mask


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(radius))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        shifted = []
        for dy in range(3):
            for dx in range(3):
                shifted.append(padded[dy : dy + out.shape[0], dx : dx + out.shape[1]])
        out = np.logical_or.reduce(shifted)
    return out


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    fg = mask.astype(bool)
    dilated = dilate_bool(fg, radius)
    eroded = ~dilate_bool(~fg, radius)
    return dilated & ~eroded


def project_points_to_image(points: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    w2c = np.linalg.inv(c2w)
    pts_cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pts_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    u = K[0, 0] * (pts_cam[:, 0] / safe_z) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / safe_z) + K[1, 2]
    return u, v, z


def sample_flow(flow: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    sampled = cv2.remap(
        flow,
        u.astype(np.float32).reshape(-1, 1),
        v.astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled.reshape(-1, flow.shape[-1])


def sample_bool(mask: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    out = np.zeros_like(valid, dtype=bool)
    out[valid] = mask[y[valid], x[valid]]
    return out


def load_c2w(cam_path: Path):
    cam = np.load(cam_path, allow_pickle=True)
    K = cam["K"]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = cam["RT"][:3, :3]
    c2w[:3, 3] = cam["RT"][:3, 3]
    return K, c2w


def load_ply_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)


class SumStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sq_sum = 0.0

    def update(self, values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sq_sum += float((arr * arr).sum())

    def mean(self):
        return None if self.count == 0 else self.sum / self.count

    def rmse(self):
        return None if self.count == 0 else math.sqrt(self.sq_sum / self.count)


def safe_mean(values) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def metric_response(repo_root: Path, sequence: str, run: str) -> Dict[str, float]:
    base = repo_root / "output" / "DNA-Rendering" / sequence / "flow_zero_v2" / run / "metrics" / "results_novelview_25000.json"
    cand = repo_root / "output" / "DNA-Rendering" / sequence / "flow_v2" / run / "metrics" / "results_novelview_25000.json"
    if not (base.exists() and cand.exists()):
        return {"has_metric_response": 0}
    b = json.loads(base.read_text())
    c = json.loads(cand.read_text())
    out = {"has_metric_response": 1}
    for metric in METRICS:
        out[f"flow_v2_delta_{metric}"] = float(c[metric]) - float(b[metric])
    return out


def diagnose_sequence(args, sequence: str) -> Dict[str, float]:
    seq_root = args.data_root / sequence
    first_model = np.load(seq_root / "model" / "000000.npz", allow_pickle=True)
    canon_xyz = first_model["obs_xyz"].astype(np.float32, copy=False)
    vertex_num = int(canon_xyz.shape[0])

    mv_ratios = []
    mv_counts = []
    vertex_to_gaussian_ratios = []

    flow_err_all = SumStats()
    no_err_all = SumStats()
    fb_all = SumStats()
    mag_all = SumStats()
    boundary_flow = SumStats()
    boundary_no = SumStats()
    nonboundary_flow = SumStats()
    nonboundary_no = SumStats()
    high_fb_flow = SumStats()
    low_fb_flow = SumStats()
    high_fb_count = 0
    total_obs = 0
    better_no = 0

    ply_path = (
        args.repo_root
        / "output"
        / "DNA-Rendering"
        / sequence
        / "flow_v2"
        / args.flow_run
        / "point_cloud"
        / "iteration_25000"
        / "point_cloud.ply"
    )
    gaussian_xyz = load_ply_xyz(ply_path) if ply_path.exists() else None
    tree = cKDTree(canon_xyz)
    knn_ids = None
    if gaussian_xyz is not None:
        _, knn_ids = tree.query(gaussian_xyz, k=min(args.seq_xyz_knn, vertex_num))
        if knn_ids.ndim == 1:
            knn_ids = knn_ids[:, None]

    checked_pairs = 0
    for pose_id in range(max(1, args.pose_start), args.pose_end + 1):
        prev_model = seq_root / "model" / f"{pose_id - 1:06d}.npz"
        curr_model = seq_root / "model" / f"{pose_id:06d}.npz"
        if not (prev_model.exists() and curr_model.exists()):
            continue
        prev_xyz = np.load(prev_model, allow_pickle=True)["obs_xyz"].astype(np.float32, copy=False)
        curr_xyz = np.load(curr_model, allow_pickle=True)["obs_xyz"].astype(np.float32, copy=False)

        vec_sum = np.zeros((vertex_num, 2), dtype=np.float64)
        mag_sum = np.zeros((vertex_num,), dtype=np.float64)
        obs_count = np.zeros((vertex_num,), dtype=np.float64)

        for view_id in args.views:
            prev_img = seq_root / "images" / f"{view_id:02d}" / f"{pose_id - 1:06d}.png"
            curr_img = seq_root / "images" / f"{view_id:02d}" / f"{pose_id:06d}.png"
            prev_cam = seq_root / "cameras" / f"{view_id:02d}" / f"{pose_id - 1:06d}.npz"
            curr_cam = seq_root / "cameras" / f"{view_id:02d}" / f"{pose_id:06d}.npz"
            prev_mask = seq_root / "bkgd_masks" / f"{view_id:02d}" / f"{pose_id - 1:06d}.png"
            if not (prev_img.exists() and curr_img.exists() and prev_cam.exists() and curr_cam.exists() and prev_mask.exists()):
                continue

            prev_gray = load_gray_scaled(prev_img, args.scale)
            curr_gray = load_gray_scaled(curr_img, args.scale)
            flow_fwd = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_bwd = cv2.calcOpticalFlowFarneback(
                curr_gray, prev_gray, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )

            prev_K, prev_c2w = load_c2w(prev_cam)
            curr_K, curr_c2w = load_c2w(curr_cam)
            prev_u, prev_v, prev_z = project_points_to_image(prev_xyz, prev_K, prev_c2w)
            curr_u, curr_v, curr_z = project_points_to_image(curr_xyz, curr_K, curr_c2w)
            prev_u_s, prev_v_s = prev_u * args.scale, prev_v * args.scale
            curr_u_s, curr_v_s = curr_u * args.scale, curr_v * args.scale
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
            ids = np.where(valid)[0]
            if ids.size == 0:
                continue
            if args.max_vertices > 0 and ids.size > args.max_vertices:
                step = int(math.ceil(ids.size / float(args.max_vertices)))
                ids = ids[::step][: args.max_vertices]

            flow = sample_flow(flow_fwd, prev_u_s[ids], prev_v_s[ids])
            end_u = prev_u_s[ids] + flow[:, 0]
            end_v = prev_v_s[ids] + flow[:, 1]
            bwd = sample_flow(flow_bwd, end_u, end_v)
            target_u = curr_u_s[ids]
            target_v = curr_v_s[ids]
            flow_err = np.sqrt((end_u - target_u) ** 2 + (end_v - target_v) ** 2)
            no_err = np.sqrt((prev_u_s[ids] - target_u) ** 2 + (prev_v_s[ids] - target_v) ** 2)
            fb_err = np.linalg.norm(flow + bwd, axis=-1)
            mag = np.linalg.norm(flow, axis=-1)

            flow_err_all.update(flow_err)
            no_err_all.update(no_err)
            fb_all.update(fb_err)
            mag_all.update(mag)
            high_fb = fb_err > args.high_fb_threshold
            high_fb_flow.update(flow_err[high_fb])
            low_fb_flow.update(flow_err[~high_fb])
            high_fb_count += int(np.count_nonzero(high_fb))
            total_obs += int(flow_err.size)
            better_no += int(np.count_nonzero(flow_err < no_err))

            fg = load_mask_scaled(prev_mask, args.scale)
            bd = boundary_band(fg, args.boundary_radius)
            bd_flag = sample_bool(bd, prev_u_s[ids], prev_v_s[ids])
            boundary_flow.update(flow_err[bd_flag])
            boundary_no.update(no_err[bd_flag])
            nonboundary_flow.update(flow_err[~bd_flag])
            nonboundary_no.update(no_err[~bd_flag])

            vec_sum[ids] += flow.astype(np.float64)
            mag_sum[ids] += mag.astype(np.float64)
            obs_count[ids] += 1.0
            checked_pairs += 1

        has = obs_count >= args.min_views
        if np.any(has):
            mean_vec = vec_sum[has] / obs_count[has, None]
            mean_mag = mag_sum[has] / obs_count[has]
            ratio = np.linalg.norm(mean_vec, axis=-1) / np.maximum(mean_mag, 1e-8)
            mv_ratios.extend(ratio.astype(np.float32).tolist())
            mv_counts.extend(obs_count[has].astype(np.float32).tolist())

            if knn_ids is not None:
                vertex_ratio = np.zeros((vertex_num,), dtype=np.float32)
                vertex_ratio[has] = ratio.astype(np.float32)
                vertex_vec = np.zeros((vertex_num, 2), dtype=np.float32)
                vertex_vec[has] = mean_vec.astype(np.float32)
                vertex_mag = np.zeros((vertex_num,), dtype=np.float32)
                vertex_mag[has] = mean_mag.astype(np.float32)
                knn_vec = vertex_vec[knn_ids]
                knn_mag = vertex_mag[knn_ids]
                valid_knn = knn_mag.mean(axis=1) > 1e-8
                if np.any(valid_knn):
                    agg_vec_norm = np.linalg.norm(knn_vec[valid_knn].mean(axis=1), axis=-1)
                    neigh_mag_mean = knn_mag[valid_knn].mean(axis=1)
                    vertex_to_gaussian_ratios.extend((agg_vec_norm / np.maximum(neigh_mag_mean, 1e-8)).astype(np.float32).tolist())

    mv = np.asarray(mv_ratios, dtype=np.float32)
    ga = np.asarray(vertex_to_gaussian_ratios, dtype=np.float32)
    result = {
        "sequence": sequence,
        "checked_pose_view_pairs": checked_pairs,
        "observation_count": int(total_obs),
        "flow_endpoint_mean_px": flow_err_all.mean(),
        "no_flow_mean_px": no_err_all.mean(),
        "flow_over_no_ratio": None if no_err_all.mean() in (None, 0) else flow_err_all.mean() / no_err_all.mean(),
        "flow_better_than_no_ratio": None if total_obs == 0 else better_no / total_obs,
        "fb_error_mean_px": fb_all.mean(),
        "flow_mag_mean_px": mag_all.mean(),
        "high_fb_ratio": None if total_obs == 0 else high_fb_count / total_obs,
        "high_fb_flow_error_px": high_fb_flow.mean(),
        "low_fb_flow_error_px": low_fb_flow.mean(),
        "boundary_flow_error_px": boundary_flow.mean(),
        "boundary_no_flow_error_px": boundary_no.mean(),
        "boundary_flow_over_no_ratio": None if boundary_no.mean() in (None, 0) else boundary_flow.mean() / boundary_no.mean(),
        "nonboundary_flow_error_px": nonboundary_flow.mean(),
        "nonboundary_no_flow_error_px": nonboundary_no.mean(),
        "nonboundary_flow_over_no_ratio": None if nonboundary_no.mean() in (None, 0) else nonboundary_flow.mean() / nonboundary_no.mean(),
        "multiview_consistency_mean": safe_mean(mv),
        "multiview_consistency_p25": None if mv.size == 0 else float(np.percentile(mv, 25)),
        "multiview_consistency_low_ratio": None if mv.size == 0 else float(np.mean(mv < 0.5)),
        "visible_views_mean": safe_mean(np.asarray(mv_counts, dtype=np.float32)),
        "gaussian_knn_consistency_mean": safe_mean(ga),
        "gaussian_knn_consistency_p25": None if ga.size == 0 else float(np.percentile(ga, 25)),
        "gaussian_knn_low_ratio": None if ga.size == 0 else float(np.mean(ga < 0.5)),
        "gaussian_count": None if gaussian_xyz is None else int(gaussian_xyz.shape[0]),
    }
    result.update(metric_response(args.repo_root, sequence, args.flow_run))
    return result


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    total = sum(int(r.get("observation_count", 0)) for r in rows)
    out = {"observation_count": total}
    weighted_keys = [
        "flow_endpoint_mean_px",
        "no_flow_mean_px",
        "fb_error_mean_px",
        "flow_mag_mean_px",
        "flow_better_than_no_ratio",
        "high_fb_ratio",
        "boundary_flow_error_px",
        "boundary_no_flow_error_px",
        "nonboundary_flow_error_px",
        "nonboundary_no_flow_error_px",
        "multiview_consistency_mean",
        "gaussian_knn_consistency_mean",
    ]
    for key in weighted_keys:
        vals = [(r.get(key), int(r.get("observation_count", 0))) for r in rows if r.get(key) is not None]
        denom = sum(w for _, w in vals)
        out[key] = None if denom == 0 else sum(float(v) * w for v, w in vals) / denom
    out["flow_over_no_ratio"] = (
        None
        if not out.get("flow_endpoint_mean_px") or not out.get("no_flow_mean_px")
        else out["flow_endpoint_mean_px"] / out["no_flow_mean_px"]
    )
    out["boundary_flow_over_no_ratio"] = (
        None
        if not out.get("boundary_flow_error_px") or not out.get("boundary_no_flow_error_px")
        else out["boundary_flow_error_px"] / out["boundary_no_flow_error_px"]
    )
    out["nonboundary_flow_over_no_ratio"] = (
        None
        if not out.get("nonboundary_flow_error_px") or not out.get("nonboundary_no_flow_error_px")
        else out["nonboundary_flow_error_px"] / out["nonboundary_no_flow_error_px"]
    )
    for metric in METRICS:
        key = f"flow_v2_delta_{metric}"
        vals = [r[key] for r in rows if r.get("has_metric_response") and key in r]
        out[key] = safe_mean(vals)
    return out


def fmt(x, digits=4):
    if x is None:
        return "-"
    return f"{float(x):.{digits}f}"


def write_markdown(path: Path, rows: Sequence[Dict[str, float]], agg: Dict[str, float], args):
    lines = [
        "# Flow Failure Mode Diagnostics",
        "",
        f"- Sequences: `{', '.join(args.sequences)}`",
        f"- Views: train views only `{','.join(map(str, args.views))}`",
        f"- Pose range: `{args.pose_start}-{args.pose_end}`",
        f"- Scale: `{args.scale}`",
        f"- Flow run for metric response / PLY: `{args.flow_run}`",
        "",
        "## Direction / Farneback / Boundary",
        "",
        "| Sequence | Flow/No | Better No | FB err | High-FB ratio | Boundary Flow/No | NonBoundary Flow/No | dPSNR flow_v2-zero | dLPIPS*1000 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['sequence']} | {fmt(r.get('flow_over_no_ratio'))} | {fmt(r.get('flow_better_than_no_ratio'))} | "
            f"{fmt(r.get('fb_error_mean_px'))} | {fmt(r.get('high_fb_ratio'))} | "
            f"{fmt(r.get('boundary_flow_over_no_ratio'))} | {fmt(r.get('nonboundary_flow_over_no_ratio'))} | "
            f"{fmt(r.get('flow_v2_delta_PSNR'))} | {fmt((r.get('flow_v2_delta_LPIPS') or 0) * 1000.0)} |"
        )
    lines.extend(
        [
            "",
            "## Multi-view and Gaussian KNN Attenuation",
            "",
            "| Sequence | Multi-view consistency mean | MV p25 | MV low<0.5 | Gaussian KNN consistency | KNN p25 | KNN low<0.5 | Gaussian count |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['sequence']} | {fmt(r.get('multiview_consistency_mean'))} | {fmt(r.get('multiview_consistency_p25'))} | "
            f"{fmt(r.get('multiview_consistency_low_ratio'))} | {fmt(r.get('gaussian_knn_consistency_mean'))} | "
            f"{fmt(r.get('gaussian_knn_consistency_p25'))} | {fmt(r.get('gaussian_knn_low_ratio'))} | "
            f"{r.get('gaussian_count') or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "```json",
            json.dumps(agg, indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/media/image/mxz/human/SeqAvatar"))
    parser.add_argument("--data-root", type=Path, default=Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering"))
    parser.add_argument("--sequences", default="0007_04 0019_10 0044_11 0051_09 0206_04 0813_05")
    parser.add_argument("--views", default=",".join(map(str, TRAIN_VIEWS)))
    parser.add_argument("--pose-start", type=int, default=1)
    parser.add_argument("--pose-end", type=int, default=99)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--max-vertices", type=int, default=4096)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--seq-xyz-knn", type=int, default=8)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--high-fb-threshold", type=float, default=1.0)
    parser.add_argument("--flow-run", default="20260706_010043")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    args.sequences = parse_list(args.sequences)
    args.views = [int(v) for v in parse_list(args.views)]

    rows = []
    for seq in args.sequences:
        print(f"[FLOW-DIAG] sequence={seq}")
        row = diagnose_sequence(args, seq)
        rows.append(row)
        print(
            "[FLOW-DIAG] {sequence}: flow/no={flow_over_no_ratio:.4f} mv={multiview_consistency_mean:.4f} "
            "knn={gaussian_knn_consistency_mean:.4f} dPSNR={flow_v2_delta_PSNR:.4f}".format(**row)
        )
    agg = aggregate(rows)
    serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    payload = {"rows": rows, "aggregate": agg, "args": serializable_args}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    write_markdown(args.output_md, rows, agg, args)
    print(f"[FLOW-DIAG] wrote {args.output_json}")
    print(f"[FLOW-DIAG] wrote {args.output_md}")


if __name__ == "__main__":
    main()
