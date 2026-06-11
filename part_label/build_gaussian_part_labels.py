#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from common import (
    LABELS,
    SOURCE_LABELS,
    default_part_label_dir,
    label_counts,
    load_cfg_args,
    model_point_cloud_path,
    namespace_to_dict,
    parse_camera_image_name,
    read_ply_xyz,
    setup_repo,
    write_colored_ply,
    write_json,
)


def parse_ints(values):
    if values is None:
        return None
    return {int(v) for v in values}


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def tensor_mask_to_numpy(mask):
    if mask is None:
        return None
    if hasattr(mask, "detach"):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = np.squeeze(arr)
    return arr != 0


def load_seqavatar_model(cfg, iteration):
    from gaussian_renderer import GaussianModel
    from scene import Scene

    gaussians = GaussianModel(cfg.sh_degree, cfg.smpl_type, cfg.motion_offset_flag, cfg.actor_gender, cfg)
    scene = Scene(cfg, gaussians, load_iteration=iteration, shuffle=False)
    return gaussians, scene


def select_cameras(cameras, views=None, frames=None, max_cameras=None, debug_one_frame=False, debug_view=0, debug_frame=0):
    selected = []
    for view in cameras:
        camera_view, frame = parse_camera_image_name(view)
        if debug_one_frame:
            if camera_view == debug_view and frame == debug_frame:
                return [view]
            continue
        if views is not None and camera_view not in views:
            continue
        if frames is not None and frame not in frames:
            continue
        selected.append(view)
        if max_cameras is not None and len(selected) >= max_cameras:
            break
    return selected


def load_smpl_rot(model_path, iteration):
    path = Path(model_path) / "smpl_rot" / f"iteration_{int(iteration)}" / "smpl_rot.pickle"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        return pickle.load(f), path


def render_one(view, gaussians, pipeline, background, smpl_rot=None, split_name="train"):
    from gaussian_renderer import render

    kwargs = {}
    if smpl_rot is not None and split_name in smpl_rot:
        by_pose = smpl_rot[split_name]
        pose_id = int(view.pose_id)
        if pose_id in by_pose:
            cached = by_pose[pose_id]
            kwargs = {
                "transforms": cached.get("transforms"),
                "translation": cached.get("translation"),
                "d_nonrigid": cached.get("d_nonrigid"),
            }
    return render(view, gaussians, pipeline, background, **kwargs)


def project_points(view, points, radii=None):
    import torch

    height = int(view.image_height)
    width = int(view.image_width)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)

    clip = points_h @ view.full_proj_transform
    w = clip[:, 3]
    safe_w = torch.where(torch.abs(w) > 1e-8, w, torch.ones_like(w) * 1e-8)
    ndc = clip[:, :3] / safe_w[:, None]
    px = torch.round((ndc[:, 0] + 1.0) * 0.5 * (width - 1)).long()
    py = torch.round((ndc[:, 1] + 1.0) * 0.5 * (height - 1)).long()

    cam = points_h @ view.world_view_transform
    z = cam[:, 2]

    valid = torch.isfinite(ndc).all(dim=1)
    valid &= torch.abs(w) > 1e-8
    valid &= z > 0
    valid &= px >= 0
    valid &= px < width
    valid &= py >= 0
    valid &= py < height
    if radii is not None:
        valid &= radii > 0

    return (
        px.detach().cpu().numpy(),
        py.detach().cpu().numpy(),
        z.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
    )


def accumulate_view_votes(vote_internal, render_pkg, view, semantic_mask, zbuffer_eps, stats):
    points = render_pkg["deformed_means3D"]
    radii = render_pkg.get("radii")
    px, py, z, valid = project_points(view, points, radii=radii)
    num_points = points.shape[0]

    height, width = semantic_mask.shape[:2]
    if height != int(view.image_height) or width != int(view.image_width):
        raise RuntimeError(
            f"semantic mask size {semantic_mask.shape[:2]} != camera image {(int(view.image_height), int(view.image_width))}"
        )

    stats["total_projected_points"] += int(num_points)
    stats["valid_in_image_points"] += int(np.sum(valid))

    bkgd_mask = tensor_mask_to_numpy(getattr(view, "bkgd_mask", None))
    bound_mask = tensor_mask_to_numpy(getattr(view, "bound_mask", None))
    if bkgd_mask is not None:
        valid &= bkgd_mask[py.clip(0, height - 1), px.clip(0, width - 1)]
    if bound_mask is not None:
        valid &= bound_mask[py.clip(0, height - 1), px.clip(0, width - 1)]
    stats["valid_fg_points"] += int(np.sum(valid))

    labels = np.zeros(num_points, dtype=np.uint8)
    labels[valid] = semantic_mask[py[valid], px[valid]]
    valid &= labels > 0
    valid &= labels <= 7
    stats["valid_semantic_points"] += int(np.sum(valid))

    if not np.any(valid):
        return labels

    valid_idx = np.where(valid)[0]
    lin = py[valid_idx] * width + px[valid_idx]
    z_valid = z[valid_idx]
    min_z = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(min_z, lin, z_valid)
    zkeep = z_valid <= (min_z[lin] + float(zbuffer_eps))
    keep_idx = valid_idx[zkeep]
    stats["valid_zbuffer_points"] += int(keep_idx.shape[0])

    kept_labels = labels[keep_idx]
    np.add.at(vote_internal, (keep_idx, kept_labels), 1)
    return labels


def fold_internal_votes(vote_internal, smpl_prior):
    vote = vote_internal[:, :7].copy()
    if vote_internal.shape[1] <= 7:
        return vote, 0

    hand_votes = vote_internal[:, 7]
    has_hand = hand_votes > 0
    left = has_hand & (smpl_prior == 2)
    right = has_hand & (smpl_prior == 3)
    other = has_hand & ~(left | right)
    vote[left, 2] += hand_votes[left]
    vote[right, 3] += hand_votes[right]
    vote[other, 1] += hand_votes[other]
    return vote, int(np.sum(hand_votes))


def fuse_labels(vote, smpl_prior, smpl_dist, min_votes, conf_thr, max_smpl_dist, prior_only=False):
    n = smpl_prior.shape[0]
    final = np.zeros(n, dtype=np.uint8)
    conf = np.zeros(n, dtype=np.float32)
    source = np.zeros(n, dtype=np.uint8)

    prior_valid = (smpl_prior > 0) & (smpl_prior <= 4) & (smpl_dist < max_smpl_dist)
    final[prior_valid] = smpl_prior[prior_valid]
    conf[prior_valid] = np.clip(1.0 - smpl_dist[prior_valid] / max_smpl_dist, 0.0, 1.0)
    source[prior_valid] = 1

    if prior_only:
        return final, conf, source

    semantic_votes = vote[:, 1:7]
    vote_sum = semantic_votes.sum(axis=1)
    vote_arg = np.argmax(semantic_votes, axis=1) + 1
    vote_max = np.max(semantic_votes, axis=1)
    vote_conf = np.divide(vote_max, vote_sum, out=np.zeros_like(vote_max, dtype=np.float32), where=vote_sum > 0)
    strong = (vote_sum >= min_votes) & (vote_conf >= conf_thr)

    for i in np.where(strong)[0]:
        label = int(vote_arg[i])
        final[i] = label
        conf[i] = float(vote_conf[i])
        source[i] = 2

        if label == 5:
            source[i] = 3
            continue
        if label == 6:
            source[i] = 4
            continue
        if label == 4:
            if smpl_prior[i] in (1, 4) or not prior_valid[i]:
                final[i] = 4
            else:
                final[i] = smpl_prior[i]
                source[i] = 1
            continue
        if label == 1:
            if prior_valid[i] and smpl_prior[i] in (2, 3, 4):
                final[i] = smpl_prior[i]
                source[i] = 1
            else:
                final[i] = 1
            continue
        if label in (2, 3):
            if prior_valid[i] and smpl_prior[i] in (2, 3):
                final[i] = smpl_prior[i]
                source[i] = 5
            else:
                final[i] = label

    return final, conf, source


def save_debug_projection(path, view, semantic_mask, render_pkg, sampled_labels=None, max_points=8000):
    image = view.original_image[:3].detach().cpu().numpy()
    image = np.transpose(np.clip(image, 0.0, 1.0), (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    overlay = image.copy()
    palette = np.array(
        [
            [0, 0, 0],
            [220, 130, 70],
            [90, 190, 40],
            [70, 70, 230],
            [130, 185, 245],
            [55, 220, 245],
            [210, 210, 70],
            [180, 120, 230],
        ],
        dtype=np.uint8,
    )
    sem_color = palette[np.clip(semantic_mask, 0, 7)]
    overlay = cv2.addWeighted(overlay, 0.65, sem_color, 0.35, 0.0)

    points = render_pkg["deformed_means3D"]
    radii = render_pkg.get("radii")
    px, py, _z, valid = project_points(view, points, radii=radii)
    idx = np.where(valid)[0]
    if idx.shape[0] > max_points:
        rng = np.random.default_rng(1234)
        idx = rng.choice(idx, size=max_points, replace=False)
    for i in idx:
        color = (255, 255, 255)
        if sampled_labels is not None:
            label = int(sampled_labels[i])
            color = tuple(int(v) for v in palette[np.clip(label, 0, 7)].tolist())
        cv2.circle(overlay, (int(px[i]), int(py[i])), 1, color, -1, lineType=cv2.LINE_AA)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def main():
    parser = argparse.ArgumentParser(description="Build final canonical Gaussian part labels from semantic votes and SMPL-X prior.")
    parser.add_argument("--source_path", default="/media/image/mxz/human/SeqAvatar/DNA-Rendering/0007_04")
    parser.add_argument("--model_path", default="/media/image/mxz/human/SeqAvatar/output/DNA-Rendering/0007_04/no_depth_no_split/20260523_164556")
    parser.add_argument("--iteration", type=int, default=25000)
    parser.add_argument("--semantic_root", default=None)
    parser.add_argument("--smpl_prior_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--max_cameras", type=int, default=None)
    parser.add_argument("--min_votes", type=int, default=3)
    parser.add_argument("--conf_thr", type=float, default=0.55)
    parser.add_argument("--max_smpl_dist", type=float, default=0.08)
    parser.add_argument("--zbuffer_eps", type=float, default=0.02)
    parser.add_argument("--debug_one_frame", action="store_true")
    parser.add_argument("--debug_view", type=int, default=0)
    parser.add_argument("--debug_frame", type=int, default=0)
    parser.add_argument("--prior_only", action="store_true", help="Save final labels from SMPL-X prior only; use only when semantic masks are unavailable.")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ignore_smpl_rot_cache", action="store_true")
    args = parser.parse_args()

    setup_repo(args.gpu)
    cfg = load_cfg_args(args.model_path, args.source_path)
    out_dir = Path(args.out_dir) if args.out_dir else default_part_label_dir(cfg.model_path, args.iteration)
    prior_dir = Path(args.smpl_prior_dir) if args.smpl_prior_dir else out_dir
    semantic_root = Path(args.semantic_root) if args.semantic_root else Path(cfg.source_path) / "semantic_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_path = out_dir / "gaussian_part_label.npy"
    if label_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists, pass --overwrite: {label_path}")

    smpl_prior_path = prior_dir / "gaussian_smpl_prior_label.npy"
    smpl_dist_path = prior_dir / "gaussian_smpl_dist.npy"
    if not smpl_prior_path.exists() or not smpl_dist_path.exists():
        raise FileNotFoundError(f"Missing SMPL prior files in {prior_dir}. Run build_smplx_nn_prior.py first.")

    smpl_prior = np.load(smpl_prior_path).astype(np.uint8)
    smpl_dist = np.load(smpl_dist_path).astype(np.float32)
    ply_xyz = read_ply_xyz(model_point_cloud_path(cfg.model_path, args.iteration))
    n = ply_xyz.shape[0]
    if smpl_prior.shape[0] != n or smpl_dist.shape[0] != n:
        raise RuntimeError(f"SMPL prior length mismatch: {smpl_prior.shape[0]}, {smpl_dist.shape[0]}, point cloud {n}")

    if args.prior_only:
        vote = np.zeros((n, 7), dtype=np.uint32)
        final, conf, source = fuse_labels(
            vote,
            smpl_prior,
            smpl_dist,
            args.min_votes,
            args.conf_thr,
            args.max_smpl_dist,
            prior_only=True,
        )
        stats = {
            "mode": "prior_only",
            "total_projected_points": 0,
            "valid_in_image_points": 0,
            "valid_fg_points": 0,
            "valid_semantic_points": 0,
            "valid_zbuffer_points": 0,
        }
        smpl_rot_path = None
        selected_count = 0
        generic_hand_votes = 0
    else:
        if not semantic_root.exists():
            raise FileNotFoundError(
                f"Missing semantic masks: {semantic_root}. "
                "Generate real masks first, or explicitly pass --prior_only."
            )

        import torch
        from arguments import PipelineParams
        from argparse import ArgumentParser

        pipe_parser = ArgumentParser()
        pipeline = PipelineParams(pipe_parser).extract(pipe_parser.parse_args([]))
        gaussians, scene = load_seqavatar_model(cfg, args.iteration)
        if gaussians.get_xyz.shape[0] != n:
            raise RuntimeError(f"Loaded Gaussian count {gaussians.get_xyz.shape[0]} != point cloud {n}")

        train_cameras = scene.getTrainCameras()
        selected = select_cameras(
            train_cameras,
            views=parse_ints(args.views),
            frames=parse_ints(args.frames),
            max_cameras=args.max_cameras,
            debug_one_frame=args.debug_one_frame,
            debug_view=args.debug_view,
            debug_frame=args.debug_frame,
        )
        if not selected:
            raise RuntimeError("No train cameras selected")

        smpl_rot = None
        smpl_rot_path = None
        if not args.ignore_smpl_rot_cache:
            smpl_rot, smpl_rot_path = load_smpl_rot(cfg.model_path, args.iteration)

        background = torch.tensor([1, 1, 1] if cfg.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        vote_internal = np.zeros((n, 8), dtype=np.uint32)
        stats = {
            "mode": "semantic_vote",
            "total_projected_points": 0,
            "valid_in_image_points": 0,
            "valid_fg_points": 0,
            "valid_semantic_points": 0,
            "valid_zbuffer_points": 0,
        }

        debug_saved = False
        for idx, view in enumerate(selected):
            camera_view, frame = parse_camera_image_name(view)
            sem_path = semantic_root / f"{int(camera_view):02d}" / f"{int(frame):06d}.png"
            semantic_mask = load_mask(sem_path)
            render_pkg = render_one(view, gaussians, pipeline, background, smpl_rot=smpl_rot, split_name="train")
            if render_pkg["deformed_means3D"].shape[0] != n:
                raise RuntimeError(f"deformed_means3D count mismatch on {view.image_name}")
            sampled_labels = accumulate_view_votes(vote_internal, render_pkg, view, semantic_mask, args.zbuffer_eps, stats)
            if args.debug_one_frame or not debug_saved:
                save_debug_projection(
                    out_dir / f"debug_projection_view{int(camera_view):02d}_frame{int(frame):06d}.png",
                    view,
                    semantic_mask,
                    render_pkg,
                    sampled_labels=sampled_labels,
                )
                debug_saved = True
            if (idx + 1) % 25 == 0:
                print(f"processed {idx + 1}/{len(selected)} cameras")

        vote, generic_hand_votes = fold_internal_votes(vote_internal, smpl_prior)
        final, conf, source = fuse_labels(
            vote,
            smpl_prior,
            smpl_dist,
            args.min_votes,
            args.conf_thr,
            args.max_smpl_dist,
            prior_only=False,
        )
        selected_count = len(selected)
        vote_arg = np.argmax(vote[:, 1:7], axis=1) + 1
        vote_sum = vote[:, 1:7].sum(axis=1)
        vote_debug = np.where(vote_sum > 0, vote_arg, 0).astype(np.uint8)
        write_colored_ply(out_dir / "gaussian_vote_debug.ply", ply_xyz, vote_debug)

    np.save(out_dir / "gaussian_part_label.npy", final)
    np.save(out_dir / "gaussian_part_conf.npy", conf)
    np.save(out_dir / "gaussian_part_vote.npy", vote)
    np.save(out_dir / "gaussian_part_source.npy", source)
    write_colored_ply(out_dir / "gaussian_part_debug.ply", ply_xyz, final)

    write_json(
        out_dir / "gaussian_part_meta.json",
        {
            "source_path": cfg.source_path,
            "model_path": cfg.model_path,
            "iteration": args.iteration,
            "semantic_root": str(semantic_root),
            "smpl_prior_dir": str(prior_dir),
            "out_dir": str(out_dir),
            "num_gaussians": int(n),
            "labels": LABELS,
            "source_labels": SOURCE_LABELS,
            "label_counts": label_counts(final),
            "source_counts": {SOURCE_LABELS[i]: int(np.sum(source == i)) for i in SOURCE_LABELS},
            "params": {
                "min_votes": args.min_votes,
                "conf_thr": args.conf_thr,
                "max_smpl_dist": args.max_smpl_dist,
                "zbuffer_eps": args.zbuffer_eps,
                "prior_only": args.prior_only,
                "views": sorted(parse_ints(args.views)) if args.views is not None else "all_train",
                "frames": sorted(parse_ints(args.frames)) if args.frames is not None else "all_train",
                "max_cameras": args.max_cameras,
                "debug_one_frame": args.debug_one_frame,
            },
            "stats": stats,
            "selected_camera_count": selected_count,
            "generic_hand_votes_folded": generic_hand_votes,
            "smpl_rot_cache": str(smpl_rot_path) if smpl_rot_path else None,
            "cfg_args": namespace_to_dict(cfg),
        },
    )

    print(f"saved gaussian part labels: {out_dir}")
    print(label_counts(final))


if __name__ == "__main__":
    main()
