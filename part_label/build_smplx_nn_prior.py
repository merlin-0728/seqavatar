#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import (
    LABELS,
    default_part_label_dir,
    label_counts,
    load_cfg_args,
    model_point_cloud_path,
    namespace_to_dict,
    read_ply_xyz,
    setup_repo,
    smplx_lbs_vertex_labels,
    write_colored_ply,
    write_json,
)


def normalize_part_name(name):
    return str(name).lower().replace("-", "_").replace(" ", "_")


def part_name_to_label(name):
    n = normalize_part_name(name)
    if any(k in n for k in ("left_hand", "lefthand", "l_hand", "left_index", "left_middle", "left_ring", "left_pinky", "left_thumb")):
        return 2
    if any(k in n for k in ("right_hand", "righthand", "r_hand", "right_index", "right_middle", "right_ring", "right_pinky", "right_thumb")):
        return 3
    if any(k in n for k in ("face", "head", "jaw", "eye")):
        return 4
    if any(k in n for k in ("hair",)):
        return 5
    if any(k in n for k in ("cloth", "clothes", "shirt", "pants", "dress", "skirt", "coat")):
        return 6
    return 1


def load_vertex_labels_from_json(path, num_vertices):
    data = json.loads(Path(path).read_text())
    labels = np.ones(num_vertices, dtype=np.uint8)

    if isinstance(data, list):
        if len(data) != num_vertices:
            raise ValueError(f"Vertex label list length {len(data)} != {num_vertices}")
        for i, value in enumerate(data):
            labels[i] = int(value) if isinstance(value, int) else part_name_to_label(value)
        return labels, {"method": "smplx_vertex_segmentation_json_list", "path": str(path)}

    if not isinstance(data, dict):
        raise ValueError("Unsupported vertex segmentation json format")

    # Common format: {"leftHand": [0, 1, ...], "rightHand": [...]}.
    if all(isinstance(v, list) for v in data.values()):
        for part_name, ids in data.items():
            label = part_name_to_label(part_name)
            ids = np.asarray(ids, dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < num_vertices)]
            labels[ids] = label
        return labels, {"method": "smplx_vertex_segmentation_json_part_lists", "path": str(path)}

    # Fallback format: {"0": "body", "1": "left_hand", ...}.
    for key, value in data.items():
        idx = int(key)
        if 0 <= idx < num_vertices:
            labels[idx] = int(value) if isinstance(value, int) else part_name_to_label(value)
    return labels, {"method": "smplx_vertex_segmentation_json_index_map", "path": str(path)}


def load_seqavatar_model(cfg, iteration):
    from gaussian_renderer import GaussianModel
    from scene import Scene

    gaussians = GaussianModel(cfg.sh_degree, cfg.smpl_type, cfg.motion_offset_flag, cfg.actor_gender, cfg)
    scene = Scene(cfg, gaussians, load_iteration=iteration, shuffle=False)
    return gaussians, scene


def main():
    parser = argparse.ArgumentParser(description="Build canonical Gaussian -> SMPL-X nearest-neighbor part prior.")
    parser.add_argument("--source_path", default="/media/image/mxz/human/SeqAvatar/DNA-Rendering/0007_04")
    parser.add_argument("--model_path", default="/media/image/mxz/human/SeqAvatar/output/DNA-Rendering/0007_04/no_depth_no_split/20260523_164556")
    parser.add_argument("--iteration", type=int, default=25000)
    parser.add_argument("--smplx_vertex_seg", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_repo(args.gpu)
    cfg = load_cfg_args(args.model_path, args.source_path)
    out_dir = Path(args.out_dir) if args.out_dir else default_part_label_dir(cfg.model_path, args.iteration)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_path = out_dir / "gaussian_smpl_prior_label.npy"
    dist_path = out_dir / "gaussian_smpl_dist.npy"
    if label_path.exists() and dist_path.exists() and not args.overwrite:
        print(f"Existing prior found: {out_dir}")
        return

    gaussians, _scene = load_seqavatar_model(cfg, args.iteration)
    gaussian_xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    canon_vertices = gaussians.canon_vertices.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)

    ply_path = model_point_cloud_path(cfg.model_path, args.iteration)
    ply_xyz = read_ply_xyz(ply_path)
    if ply_xyz.shape[0] != gaussian_xyz.shape[0]:
        raise RuntimeError(f"PLY Gaussian count {ply_xyz.shape[0]} != loaded gaussians {gaussian_xyz.shape[0]}")
    if not np.allclose(ply_xyz, gaussian_xyz, atol=1e-6):
        max_err = float(np.max(np.abs(ply_xyz - gaussian_xyz)))
        raise RuntimeError(f"PLY xyz order/content differs from gaussians.get_xyz, max error={max_err}")

    if args.smplx_vertex_seg:
        vertex_labels, seg_meta = load_vertex_labels_from_json(args.smplx_vertex_seg, canon_vertices.shape[0])
    else:
        vertex_labels, seg_meta = smplx_lbs_vertex_labels(gaussians.SMPL_NEUTRAL)

    tree = cKDTree(canon_vertices)
    smpl_dist, vert_ids = tree.query(gaussian_xyz, k=1)
    smpl_prior_label = vertex_labels[vert_ids].astype(np.uint8)
    smpl_dist = smpl_dist.astype(np.float32)

    np.save(label_path, smpl_prior_label)
    np.save(dist_path, smpl_dist)
    write_colored_ply(out_dir / "debug_smpl_prior.ply", gaussian_xyz, smpl_prior_label)

    write_json(
        out_dir / "smpl_prior_meta.json",
        {
            "source_path": cfg.source_path,
            "model_path": cfg.model_path,
            "iteration": args.iteration,
            "point_cloud": str(ply_path),
            "out_dir": str(out_dir),
            "num_gaussians": int(gaussian_xyz.shape[0]),
            "num_smplx_vertices": int(canon_vertices.shape[0]),
            "labels": LABELS,
            "label_counts": label_counts(smpl_prior_label),
            "distance_stats": {
                "min": float(np.min(smpl_dist)),
                "mean": float(np.mean(smpl_dist)),
                "median": float(np.median(smpl_dist)),
                "max": float(np.max(smpl_dist)),
            },
            "segmentation": seg_meta,
            "cfg_args": namespace_to_dict(cfg),
        },
    )

    print(f"saved SMPL-X prior: {out_dir}")
    print(label_counts(smpl_prior_label))


if __name__ == "__main__":
    main()
