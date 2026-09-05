"""Audit VGGT candidate projections against DNA images and baseline renders."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def project(points, camera_path):
    camera = np.load(camera_path, allow_pickle=False)
    world_to_camera = np.linalg.inv(camera["RT"])
    camera_xyz = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    projected, _ = cv2.projectPoints(
        camera_xyz.astype(np.float64),
        np.zeros(3),
        np.zeros(3),
        camera["K"].astype(np.float64),
        camera["D"].astype(np.float64),
    )
    return projected.reshape(-1, 2), camera_xyz[:, 2]


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def error_mask(original_model, sequence, frame_id, view_id, shape):
    if original_model is None:
        return None
    render_path = (
        original_model / "novelview" / "ours_25000" / "renders"
        / f"frame_{frame_id:06d}_view_{view_id:02d}.png"
    )
    gt_path = (
        original_model / "novelview" / "ours_25000" / "gt"
        / f"frame_{frame_id:06d}_view_{view_id:02d}.png"
    )
    if not render_path.exists() or not gt_path.exists():
        return None
    render = load_rgb(render_path).astype(np.float32) / 255.0
    gt = load_rgb(gt_path).astype(np.float32) / 255.0
    if render.shape[:2] != shape or gt.shape[:2] != shape:
        return None
    error = np.abs(render - gt).mean(axis=2)
    positive = error[error > 1e-6]
    threshold = np.percentile(positive, 80.0) if positive.size else np.inf
    return error >= threshold


def draw_overlay(image, xy, scores, path):
    output = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(output)
    order = np.argsort(scores)
    for index in order:
        x, y = xy[index]
        color = (255, int(255 * (1.0 - scores[index])), 32)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    output.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--candidate", default="")
    parser.add_argument("--original-model", default="")
    parser.add_argument("--output", default="output/vggt_candidate_audit")
    parser.add_argument("--surface-threshold", type=float, default=0.018)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sequence_root = repo_root / "DNA-Rendering" / args.sequence
    candidate_path = Path(args.candidate) if args.candidate else (
        repo_root / "output" / "vggt_garment_candidates" / args.sequence / "candidates.npz"
    )
    original_model = Path(args.original_model).resolve() if args.original_model else None
    output_root = repo_root / args.output / args.sequence
    overlay_root = output_root / "overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)

    data = np.load(candidate_path, allow_pickle=False)
    points = data["reference_xyz"].astype(np.float32)
    scores = data["score"].astype(np.float32)
    source_views = data["source_view"].astype(np.int32)
    frame_ids = data["frame_id"].astype(np.int32)
    smpl_obs = np.asarray(np.load(sequence_root / "model" / "000000.npz")["obs_xyz"], dtype=np.float32)
    body_extent = float(np.linalg.norm(smpl_obs.max(axis=0) - smpl_obs.min(axis=0)))

    rows = []
    for frame_id, view_id in zip(frame_ids, source_views):
        camera_path = sequence_root / "cameras" / f"{view_id:02d}" / f"{frame_id:06d}.npz"
        image_path = sequence_root / "images" / f"{view_id:02d}" / f"{frame_id:06d}.png"
        mask_path = sequence_root / "bkgd_masks" / f"{view_id:02d}" / f"{frame_id:06d}.png"
        if not camera_path.exists() or not image_path.exists() or not mask_path.exists():
            rows.append({"valid": False, "reason": "missing_source_files"})
            continue
        # Process one point at a time here because the cache can mix frames/views.
        index = len(rows)
        xy, depth = project(points[index:index + 1], camera_path)
        image = load_rgb(image_path)
        mask = load_mask(mask_path)
        height, width = image.shape[:2]
        x = int(np.rint(xy[0, 0]))
        y = int(np.rint(xy[0, 1]))
        valid = bool(depth[0] > 0 and 0 <= x < width and 0 <= y < height)
        row = {
            "valid": valid,
            "frame_id": int(frame_id),
            "source_view": int(view_id),
            "x": x,
            "y": y,
            "score": float(scores[index]),
        }
        if valid:
            boundary = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)
            ) > 0
            row["foreground"] = bool(mask[y, x])
            row["boundary_3px"] = bool(boundary[y, x])
            row["high_error"] = None
            high_error = error_mask(original_model, args.sequence, int(frame_id), int(view_id), image.shape[:2])
            if high_error is not None:
                row["high_error"] = bool(high_error[y, x])
        else:
            row.update({"foreground": False, "boundary_3px": False, "high_error": None})
        rows.append(row)

    # The nearest SMPL distance is computed in the same aligned reference frame
    # used by candidate generation; it is a selection diagnostic, not a garment GT.
    nearest = np.asarray(data["nearest_smpl_vertex"], dtype=np.int64)
    smpl_by_frame = {}
    for frame_id in np.unique(frame_ids):
        smpl_by_frame[int(frame_id)] = np.asarray(
            np.load(sequence_root / "model" / f"{int(frame_id):06d}.npz")["obs_xyz"], dtype=np.float32
        )
    body_distance = np.linalg.norm(
        points - np.stack([smpl_by_frame[int(f)][int(v)] for f, v in zip(frame_ids, nearest)]), axis=1
    )
    threshold = args.surface_threshold * body_extent
    for row, distance in zip(rows, body_distance):
        row["smpl_distance"] = float(distance)
        row["outside_smpl_threshold"] = bool(distance >= threshold)

    def fraction(key, subset=None):
        values = [r[key] for r in rows if r.get("valid") and r.get(key) is not None and (subset is None or subset(r))]
        return float(np.mean(values)) if values else None

    stats = {
        "sequence": args.sequence,
        "candidate_count": len(rows),
        "valid_projection_fraction": fraction("valid"),
        "foreground_fraction": fraction("foreground"),
        "boundary_3px_fraction": fraction("boundary_3px"),
        "high_error_fraction": fraction("high_error"),
        "high_error_evaluable_count": sum(r.get("high_error") is not None for r in rows),
        "outside_smpl_threshold_fraction": float(np.mean([r["outside_smpl_threshold"] for r in rows])),
        "surface_threshold_world": float(threshold),
        "source_views": sorted(set(int(v) for v in source_views)),
        "frames": sorted(set(int(f) for f in frame_ids)),
    }
    (output_root / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (output_root / "points.json").write_text(json.dumps(rows, indent=2) + "\n")

    for frame_id, view_id in sorted(set(zip(frame_ids.tolist(), source_views.tolist()))):
        indices = np.where((frame_ids == frame_id) & (source_views == view_id))[0]
        image_path = sequence_root / "images" / f"{view_id:02d}" / f"{frame_id:06d}.png"
        if not image_path.exists():
            continue
        image = load_rgb(image_path)
        xy, depth = project(points[indices], sequence_root / "cameras" / f"{view_id:02d}" / f"{frame_id:06d}.npz")
        height, width = image.shape[:2]
        visible = (depth > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        draw_overlay(
            image,
            xy[visible],
            scores[indices][visible],
            overlay_root / f"frame_{frame_id:06d}_view_{view_id:02d}.png",
        )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
