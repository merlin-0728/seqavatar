#!/usr/bin/env python3
"""Fast headroom diagnostic for view-wise optical-flow tokens.

This script does not train a model and does not use novel/test views. It splits
DNA train views into source views and held-out validation views. Source
view-wise flow observations are converted to a 3D displacement by projection
Jacobian least squares, then evaluated by reprojection endpoint error on held
out train views.

The comparison is:
  - no_motion: previous vertex projection vs current vertex projection.
  - collapsed: average source-view flow/Jacobian first, then solve dX.
  - viewwise: keep per-view tokens, stack per-view Jacobians, then solve dX.
  - best_conf: use the single source view with highest FB-confidence.
  - smpl_oracle: use current_xyz - prev_xyz, an upper bound for SMPL vertices.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np


TRAIN_VIEWS = list(range(0, 48, 2))
METHODS = ("no_motion", "collapsed", "viewwise", "best_conf", "smpl_oracle")


def parse_list(raw: str) -> List[str]:
    return [x for x in re.split(r"[\s,]+", raw.strip()) if x]


def load_gray_scaled(path: Path, scale: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def load_camera(path: Path, scale: float):
    cam = np.load(path, allow_pickle=True)
    K = cam["K"].astype(np.float64).copy()
    K[:2, :] *= float(scale)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = cam["RT"][:3, :3]
    c2w[:3, 3] = cam["RT"][:3, 3]
    w2c = np.linalg.inv(c2w)
    return K, w2c


def project(points: np.ndarray, K: np.ndarray, w2c: np.ndarray):
    pts_cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pts_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    u = K[0, 0] * pts_cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / safe_z + K[1, 2]
    return u, v, z, pts_cam


def projection_jacobian(points: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    _, _, z, pts_cam = project(points, K, w2c)
    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
    safe_z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    fx = K[0, 0]
    fy = K[1, 1]
    j_cam = np.zeros((points.shape[0], 2, 3), dtype=np.float64)
    j_cam[:, 0, 0] = fx / safe_z
    j_cam[:, 0, 2] = -fx * x / (safe_z * safe_z)
    j_cam[:, 1, 1] = fy / safe_z
    j_cam[:, 1, 2] = -fy * y / (safe_z * safe_z)
    return np.einsum("nij,jk->nik", j_cam, w2c[:3, :3])


def sample_flow(flow: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    sampled = cv2.remap(
        flow,
        u.astype(np.float32).reshape(-1, 1),
        v.astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled.reshape(-1, flow.shape[-1]).astype(np.float64)


class ErrorStats:
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


def solve_ridge(a: np.ndarray, b: np.ndarray, valid: np.ndarray, ridge: float) -> np.ndarray:
    out = np.zeros((a.shape[0], 3), dtype=np.float64)
    ids = np.where(valid)[0]
    if ids.size == 0:
        return out
    eye = np.eye(3, dtype=np.float64)[None, :, :] * float(ridge)
    try:
        out[ids] = np.linalg.solve(a[ids] + eye, b[ids])
    except np.linalg.LinAlgError:
        out[ids] = np.linalg.lstsq(a[ids] + eye, b[ids], rcond=None)[0]
    return out


def evaluate_sequence(args, sequence: str) -> Dict[str, object]:
    seq_root = args.data_root / sequence
    source_views = [int(v) for v in args.source_views]
    val_views = [int(v) for v in args.val_views]

    first = np.load(seq_root / "model" / "000000.npz", allow_pickle=True)
    vertex_num = int(first["obs_xyz"].shape[0])
    vertex_ids = np.arange(vertex_num)
    if args.max_vertices > 0 and vertex_ids.size > args.max_vertices:
        step = int(math.ceil(vertex_ids.size / float(args.max_vertices)))
        vertex_ids = vertex_ids[::step][: args.max_vertices]
    m = int(vertex_ids.size)

    stats = {name: ErrorStats() for name in METHODS}
    source_counts = ErrorStats()
    best_conf_values = ErrorStats()

    checked_poses = 0
    for pose_id in range(max(1, args.pose_start), args.pose_end + 1, args.pose_step):
        prev_path = seq_root / "model" / f"{pose_id - 1:06d}.npz"
        curr_path = seq_root / "model" / f"{pose_id:06d}.npz"
        if not (prev_path.exists() and curr_path.exists()):
            continue
        prev_xyz_all = np.load(prev_path, allow_pickle=True)["obs_xyz"].astype(np.float64, copy=False)
        curr_xyz_all = np.load(curr_path, allow_pickle=True)["obs_xyz"].astype(np.float64, copy=False)
        prev_xyz = prev_xyz_all[vertex_ids]
        curr_xyz = curr_xyz_all[vertex_ids]

        a_view = np.zeros((m, 3, 3), dtype=np.float64)
        b_view = np.zeros((m, 3), dtype=np.float64)
        j_sum = np.zeros((m, 2, 3), dtype=np.float64)
        f_sum = np.zeros((m, 2), dtype=np.float64)
        count = np.zeros((m,), dtype=np.float64)
        best_conf = np.full((m,), -1.0, dtype=np.float64)
        best_j = np.zeros((m, 2, 3), dtype=np.float64)
        best_f = np.zeros((m, 2), dtype=np.float64)

        for view_id in source_views:
            prev_img = seq_root / "images" / f"{view_id:02d}" / f"{pose_id - 1:06d}.png"
            curr_img = seq_root / "images" / f"{view_id:02d}" / f"{pose_id:06d}.png"
            cam_path = seq_root / "cameras" / f"{view_id:02d}" / f"{pose_id - 1:06d}.npz"
            if not (prev_img.exists() and curr_img.exists() and cam_path.exists()):
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
            K, w2c = load_camera(cam_path, args.scale)
            u, v, z, _ = project(prev_xyz, K, w2c)
            h, w = flow_fwd.shape[:2]
            valid = (z > 1e-6) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
            if not np.any(valid):
                continue
            flow = sample_flow(flow_fwd, u, v)
            end_u = u + flow[:, 0]
            end_v = v + flow[:, 1]
            bwd = sample_flow(flow_bwd, end_u, end_v)
            fb_error = np.linalg.norm(flow + bwd, axis=-1)
            conf = np.exp(-fb_error / 2.0)
            if not args.use_conf_weight:
                conf = np.ones_like(conf)
            j = projection_jacobian(prev_xyz, K, w2c)

            ids = np.where(valid)[0]
            wgt = conf[ids]
            ji = j[ids]
            fi = flow[ids]
            a_view[ids] += wgt[:, None, None] * np.einsum("nij,nik->njk", ji, ji)
            b_view[ids] += wgt[:, None] * np.einsum("nij,ni->nj", ji, fi)
            j_sum[ids] += ji
            f_sum[ids] += fi
            count[ids] += 1.0
            better = valid & (conf > best_conf)
            best_conf[better] = conf[better]
            best_j[better] = j[better]
            best_f[better] = flow[better]

        has = count >= args.min_source_views
        if not np.any(has):
            continue
        source_counts.update(count[has])
        best_conf_values.update(best_conf[has])
        d_view = solve_ridge(a_view, b_view, has, args.ridge)

        j_mean = np.zeros_like(j_sum)
        f_mean = np.zeros_like(f_sum)
        j_mean[has] = j_sum[has] / count[has, None, None]
        f_mean[has] = f_sum[has] / count[has, None]
        a_collapsed = np.einsum("nij,nik->njk", j_mean, j_mean)
        b_collapsed = np.einsum("nij,ni->nj", j_mean, f_mean)
        d_collapsed = solve_ridge(a_collapsed, b_collapsed, has, args.ridge)

        a_best = np.einsum("nij,nik->njk", best_j, best_j)
        b_best = np.einsum("nij,ni->nj", best_j, best_f)
        d_best = solve_ridge(a_best, b_best, has, args.ridge)
        d_smpl = curr_xyz - prev_xyz

        for view_id in val_views:
            cam_path = seq_root / "cameras" / f"{view_id:02d}" / f"{pose_id - 1:06d}.npz"
            curr_cam_path = seq_root / "cameras" / f"{view_id:02d}" / f"{pose_id:06d}.npz"
            if not (cam_path.exists() and curr_cam_path.exists()):
                continue
            K_prev, w2c_prev = load_camera(cam_path, args.scale)
            K_curr, w2c_curr = load_camera(curr_cam_path, args.scale)
            prev_u, prev_v, prev_z, _ = project(prev_xyz, K_prev, w2c_prev)
            curr_u, curr_v, curr_z, _ = project(curr_xyz, K_curr, w2c_curr)
            # DNA cameras have fixed image sizes; load a small image only to get bounds.
            img_path = seq_root / "images" / f"{view_id:02d}" / f"{pose_id - 1:06d}.png"
            gray = load_gray_scaled(img_path, args.scale)
            h, w = gray.shape[:2]
            val_valid = (
                has
                & (prev_z > 1e-6)
                & (curr_z > 1e-6)
                & (prev_u >= 0)
                & (prev_u <= w - 1)
                & (prev_v >= 0)
                & (prev_v <= h - 1)
                & (curr_u >= 0)
                & (curr_u <= w - 1)
                & (curr_v >= 0)
                & (curr_v <= h - 1)
            )
            if not np.any(val_valid):
                continue

            def endpoint_error(delta):
                pred_u, pred_v, pred_z, _ = project(prev_xyz + delta, K_curr, w2c_curr)
                ok = val_valid & (pred_z > 1e-6)
                err = np.sqrt((pred_u[ok] - curr_u[ok]) ** 2 + (pred_v[ok] - curr_v[ok]) ** 2)
                return err

            stats["no_motion"].update(np.sqrt((prev_u[val_valid] - curr_u[val_valid]) ** 2 + (prev_v[val_valid] - curr_v[val_valid]) ** 2))
            stats["collapsed"].update(endpoint_error(d_collapsed))
            stats["viewwise"].update(endpoint_error(d_view))
            stats["best_conf"].update(endpoint_error(d_best))
            stats["smpl_oracle"].update(endpoint_error(d_smpl))
        checked_poses += 1

    row = {
        "sequence": sequence,
        "checked_poses": checked_poses,
        "source_views": source_views,
        "val_views": val_views,
        "mean_source_view_count": source_counts.mean(),
        "mean_best_conf": best_conf_values.mean(),
    }
    for method, st in stats.items():
        row[f"{method}_mean_px"] = st.mean()
        row[f"{method}_rmse_px"] = st.rmse()
        row[f"{method}_count"] = st.count
    if row["no_motion_mean_px"]:
        for method in ("collapsed", "viewwise", "best_conf", "smpl_oracle"):
            row[f"{method}_over_no"] = row[f"{method}_mean_px"] / row["no_motion_mean_px"]
    if row["collapsed_mean_px"]:
        row["viewwise_over_collapsed"] = row["viewwise_mean_px"] / row["collapsed_mean_px"]
    return row


def aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    total = sum(int(r.get("no_motion_count", 0)) for r in rows)
    out["count"] = total
    for method in METHODS:
        key = f"{method}_mean_px"
        vals = [(r.get(key), int(r.get(f"{method}_count", 0))) for r in rows if r.get(key) is not None]
        denom = sum(w for _, w in vals)
        out[key] = None if denom == 0 else sum(float(v) * w for v, w in vals) / denom
    if out.get("no_motion_mean_px"):
        for method in ("collapsed", "viewwise", "best_conf", "smpl_oracle"):
            out[f"{method}_over_no"] = out[f"{method}_mean_px"] / out["no_motion_mean_px"]
    if out.get("collapsed_mean_px"):
        out["viewwise_over_collapsed"] = out["viewwise_mean_px"] / out["collapsed_mean_px"]
    return out


def fmt(x, digits=4):
    return "-" if x is None else f"{float(x):.{digits}f}"


def write_md(path: Path, rows: Sequence[Dict[str, object]], agg: Dict[str, object], args):
    lines = [
        "# Flow View-wise Token Headroom",
        "",
        f"- Sequences: `{', '.join(args.sequences)}`",
        f"- Source train views: `{','.join(map(str, args.source_views))}`",
        f"- Held-out validation train views: `{','.join(map(str, args.val_views))}`",
        f"- Pose range/step: `{args.pose_start}-{args.pose_end}/{args.pose_step}`",
        f"- Max vertices: `{args.max_vertices}`",
        "",
        "| Seq | No px | Collapsed px | Viewwise px | Best-conf px | SMPL oracle px | Viewwise/Collapsed | Viewwise/No | Source views |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['sequence']} | {fmt(r.get('no_motion_mean_px'))} | {fmt(r.get('collapsed_mean_px'))} | "
            f"{fmt(r.get('viewwise_mean_px'))} | {fmt(r.get('best_conf_mean_px'))} | "
            f"{fmt(r.get('smpl_oracle_mean_px'))} | {fmt(r.get('viewwise_over_collapsed'))} | "
            f"{fmt(r.get('viewwise_over_no'))} | {fmt(r.get('mean_source_view_count'))} |"
        )
    lines.extend(["", "## Aggregate", "", "```json", json.dumps(agg, indent=2), "```"])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/media/image/mxz/human/SeqAvatar/DNA-Rendering"))
    parser.add_argument("--sequences", default="0007_04 0044_11")
    parser.add_argument("--source-views", default=",".join(map(str, TRAIN_VIEWS[::2])))
    parser.add_argument("--val-views", default=",".join(map(str, TRAIN_VIEWS[1::2])))
    parser.add_argument("--pose-start", type=int, default=1)
    parser.add_argument("--pose-end", type=int, default=99)
    parser.add_argument("--pose-step", type=int, default=5)
    parser.add_argument("--max-vertices", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--min-source-views", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--use-conf-weight", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    args.sequences = parse_list(args.sequences)
    args.source_views = [int(v) for v in parse_list(args.source_views)]
    args.val_views = [int(v) for v in parse_list(args.val_views)]

    rows = []
    for seq in args.sequences:
        print(f"[FLOW-VIEW-HEADROOM] sequence={seq}", flush=True)
        row = evaluate_sequence(args, seq)
        rows.append(row)
        print(
            "[FLOW-VIEW-HEADROOM] {sequence}: no={no_motion_mean_px:.4f} collapsed={collapsed_mean_px:.4f} "
            "viewwise={viewwise_mean_px:.4f} ratio={viewwise_over_collapsed:.4f}".format(**row),
            flush=True,
        )
    agg = aggregate(rows)
    payload = {
        "rows": rows,
        "aggregate": agg,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    write_md(args.output_md, rows, agg, args)
    print(f"[FLOW-VIEW-HEADROOM] wrote {args.output_json}", flush=True)
    print(f"[FLOW-VIEW-HEADROOM] wrote {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
