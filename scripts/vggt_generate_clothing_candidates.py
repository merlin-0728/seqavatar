"""Generate VGGT-proposed garment candidates for DNA-Rendering.

The output is an offline proposal cache.  It is not a geometry ground truth:
SeqAvatar still decides whether accepted candidates are useful through its
normal RGB, mask, SSIM, LPIPS and AIAP losses.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VGGT_ROOT = REPO_ROOT / "third_party" / "vggt"
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.spatial import cKDTree

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from smpl_model.smplx.body_models import SMPLX
from utils.smpl_utils import (
    SMPL_to_tensor,
    batch_rodrigues,
    create_canonical_vertices,
    get_transform_params_torch,
    read_pickle,
)


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def load_pad_mask(path, target_size=518):
    image = Image.open(path).convert("L")
    width, height = image.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
    canvas = Image.new("L", (target_size, target_size), 0)
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    canvas.paste(image, (left, top))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def load_pad_rgb(path, target_size=518):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    canvas.paste(image, (left, top))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def mask_boundary(mask):
    image = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    dilated = np.asarray(image.filter(ImageFilter.MaxFilter(5))) > 0
    eroded = np.asarray(image.filter(ImageFilter.MinFilter(5))) > 0
    return (dilated & ~eroded).astype(np.float32)


def umeyama_similarity(source, target):
    """Return row-vector Sim(3): target ~= source @ rotation.T * scale + translation."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("At least three paired camera centers are required for Sim(3) alignment")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = np.sum(source_centered * source_centered) / source.shape[0]
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_similarity(points, scale, rotation, translation):
    points = np.asarray(points, dtype=np.float64)
    return (scale * (points @ rotation.T) + translation).astype(np.float32)


def camera_center_from_c2w(path):
    params = np.load(path, allow_pickle=True)
    c2w = np.asarray(params["RT"], dtype=np.float64)
    return c2w[:3, 3]


def load_smpl_context(repo_root, sequence, frame_id, device):
    sequence_path = repo_root / "DNA-Rendering" / sequence
    model_data = np.load(sequence_path / "model" / f"{frame_id:06d}.npz", allow_pickle=True)
    gender = "neutral"
    smc_path = sequence_path / f"{sequence}.smc"
    try:
        from utils.SMCReader import SMCReader

        gender = str(SMCReader(str(smc_path)).actor_info["gender"])
    except Exception:
        pass
    body_model = SMPLX(
        str(repo_root / "smpl_model" / "models"),
        smpl_type="smplx",
        gender=gender,
        use_face_contour=True,
        flat_hand_mean=False,
        use_pca=False,
        num_pca_comps=24,
        num_betas=10,
        num_expression_coeffs=10,
        ext="pkl",
    )
    with torch.no_grad():
        canonical_params, canonical_vertices = create_canonical_vertices(body_model, "smplx")
    neutral_path = repo_root / "smpl_model" / "models" / f"SMPLX_{gender.upper()}.pkl"
    smpl = SMPL_to_tensor(read_pickle(str(neutral_path)), device=device)
    canonical_params = {
        key: torch.as_tensor(value, device=device)
        for key, value in canonical_params.items()
    }
    target_params = {
        key: torch.as_tensor(model_data[key], device=device)
        for key in model_data.files
        if key in {"poses", "shapes", "R", "Th"}
    }
    target_params["rot_mats"] = batch_rodrigues(target_params["poses"].reshape(-1, 3)).view(1, -1, 3, 3)
    with torch.no_grad():
        canonical_a, _, _, _ = get_transform_params_torch(smpl, canonical_params)
        target_a, global_r, global_th, _ = get_transform_params_torch(
            smpl, target_params, rot_mats=target_params["rot_mats"]
        )
    return {
        "canonical_vertices": np.asarray(canonical_vertices, dtype=np.float32),
        "obs_xyz": np.asarray(model_data["obs_xyz"], dtype=np.float32),
        "weights": smpl["weights"].detach().cpu().numpy().astype(np.float32),
        "canonical_a": canonical_a[0].detach().cpu().numpy().astype(np.float32),
        "target_a": target_a[0].detach().cpu().numpy().astype(np.float32),
        "global_r": global_r.detach().cpu().numpy().astype(np.float32),
        "global_th": global_th[0].detach().cpu().numpy().astype(np.float32),
    }


def canonicalize_points(points, context, nearest_ids, eta, max_offset):
    obs_xyz = context["obs_xyz"][nearest_ids]
    offset_world = points - obs_xyz
    # SeqAvatar applies world = smpl @ inv(global_R) for row vectors.
    # Undo that global transform before inverting the blended pose transform.
    offset_smpl = offset_world @ context["global_r"]
    weights = context["weights"][nearest_ids]
    target_a = context["target_a"][..., :3, :3]
    canonical_a = context["canonical_a"][..., :3, :3]
    target_rot = np.einsum("nk,kij->nij", weights, target_a)
    canonical_rot = np.einsum("nk,kij->nij", weights, canonical_a)
    # coarse_deform_c2source maps canonical offsets with
    #   target_rot @ inv(canonical_rot)
    # in column-vector form.  Invert that exact row-vector equivalent here.
    # The confidence controls how much VGGT offset is transferred; the image
    # losses refine it afterwards.
    offset_target_inverse = np.einsum(
        "nij,nj->ni", np.linalg.inv(target_rot), offset_smpl
    )
    offset_canonical = np.einsum(
        "ni,nij->nj",
        offset_target_inverse,
        canonical_rot.transpose(0, 2, 1),
    )
    lengths = np.linalg.norm(offset_canonical, axis=1, keepdims=True)
    canonical_xyz = context["canonical_vertices"][nearest_ids] + offset_canonical * np.minimum(
        1.0, max_offset / np.maximum(lengths, 1e-8)
    ) * eta[:, None]
    return canonical_xyz.astype(np.float32)


def robust_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(values, [5, 95]) if values.size else (0.0, 1.0)
    return np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def process_frame(model, repo_root, sequence, frame_id, view_ids, device, args):
    sequence_path = repo_root / "DNA-Rendering" / sequence
    image_paths = [
        str(sequence_path / "images" / f"{view_id:02d}" / f"{frame_id:06d}.png")
        for view_id in view_ids
    ]
    mask_paths = [
        str(sequence_path / "bkgd_masks" / f"{view_id:02d}" / f"{frame_id:06d}.png")
        for view_id in view_ids
    ]
    images = load_and_preprocess_images(image_paths, mode="pad").to(device)
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            prediction = model(images)
    points = prediction["world_points"][0].float().cpu().numpy()
    confidence = prediction["world_points_conf"][0].float().cpu().numpy()
    pose_enc = prediction["pose_enc"].float()
    predicted_extrinsics, _ = pose_encoding_to_extri_intri(
        pose_enc, images.shape[-2:], build_intrinsics=False
    )
    predicted_extrinsics = predicted_extrinsics[0].detach().cpu().numpy()
    predicted_centers = []
    known_centers = []
    for view_id, extrinsic in zip(view_ids, predicted_extrinsics):
        pred_c2w = np.eye(4, dtype=np.float64)
        pred_c2w[:3, :4] = np.linalg.inv(np.vstack([extrinsic, [0, 0, 0, 1]]))[:3, :4]
        predicted_centers.append(pred_c2w[:3, 3])
        known_centers.append(camera_center_from_c2w(
            sequence_path / "cameras" / f"{view_id:02d}" / f"{frame_id:06d}.npz"
        ))
    scale, rotation, translation = umeyama_similarity(predicted_centers, known_centers)
    points = apply_similarity(points.reshape(-1, 3), scale, rotation, translation).reshape(points.shape)

    context = load_smpl_context(repo_root, sequence, frame_id, device)
    body_tree = cKDTree(context["obs_xyz"])
    all_points = []
    all_conf = []
    all_boundary = []
    all_views = []
    all_colors = []
    for local_view, mask_path in enumerate(mask_paths):
        mask = load_pad_mask(mask_path, points.shape[2])
        boundary = mask_boundary(mask)
        rgb = load_pad_rgb(image_paths[local_view], points.shape[2])
        view_points = points[local_view].reshape(-1, 3)
        view_conf = confidence[local_view].reshape(-1)
        fg = mask.reshape(-1) > 0.5
        valid = fg & np.isfinite(view_points).all(axis=1) & np.isfinite(view_conf)
        ids = np.flatnonzero(valid)
        if ids.size > args.max_points_per_view:
            order = np.argsort(view_conf[ids])[-args.max_points_per_view:]
            ids = ids[order]
        all_points.append(view_points[ids])
        all_conf.append(view_conf[ids])
        all_boundary.append(boundary.reshape(-1)[ids])
        all_views.append(np.full(ids.shape, view_ids[local_view], dtype=np.int32))
        all_colors.append(rgb.reshape(-1, 3)[ids])
    points = np.concatenate(all_points, axis=0)
    confidence = np.concatenate(all_conf, axis=0)
    boundary = np.concatenate(all_boundary, axis=0)
    source_views = np.concatenate(all_views, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    _, nearest_ids = body_tree.query(points, k=1)
    body_distance = np.linalg.norm(points - context["obs_xyz"][nearest_ids], axis=1)
    body_extent = np.linalg.norm(context["obs_xyz"].max(axis=0) - context["obs_xyz"].min(axis=0))
    clothing_offset = np.clip(
        (body_distance - args.surface_threshold * body_extent)
        / max(args.surface_threshold * body_extent, 1e-6),
        0.0,
        1.0,
    )

    # A point is more trustworthy when another input view explains a nearby
    # point in the aligned cloud.  This is a proposal confidence, not a loss.
    tree = cKDTree(points)
    nearest_dist, nearest_idx = tree.query(points, k=min(12, len(points)))
    consistency = np.zeros(len(points), dtype=np.float32)
    for row, (dists, neighbors) in enumerate(zip(nearest_dist, nearest_idx)):
        other = source_views[neighbors] != source_views[row]
        if np.any(other):
            consistency[row] = np.exp(-float(dists[other][0]) / max(args.consistency_scale * body_extent, 1e-6))
    score = (
        robust_normalize(confidence)
        * (0.25 + 0.75 * consistency)
        * (0.25 + 0.75 * clothing_offset)
        * (0.50 + 0.50 * boundary)
    )
    keep = (
        confidence >= np.percentile(confidence, args.confidence_percentile)
        ) & (body_distance >= args.surface_threshold * body_extent)
    if not np.any(keep):
        return None
    points = points[keep]
    score = score[keep]
    confidence = confidence[keep]
    source_views = source_views[keep]
    nearest_ids = nearest_ids[keep]
    colors = colors[keep]
    eta = np.clip(0.25 + 0.75 * robust_normalize(confidence), 0.25, 1.0) * args.offset_transfer
    canonical_xyz = canonicalize_points(
        points, context, nearest_ids, eta, max_offset=args.max_offset_ratio * body_extent
    )
    return {
        "canonical_xyz": canonical_xyz,
        "reference_xyz": points.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "score": score.astype(np.float32),
        "source_view": source_views.astype(np.int32),
        "nearest_smpl_vertex": nearest_ids.astype(np.int32),
        "frame_id": np.full(len(points), frame_id, dtype=np.int32),
        # RGB is only an initialization copied from the proposal pixel; it is
        # never used as a VGGT supervision target.
        "colors": colors.astype(np.float32),
        "alignment_scale": np.asarray([scale], dtype=np.float32),
        "alignment_rotation": rotation.astype(np.float32),
        "alignment_translation": translation.astype(np.float32),
    }


def voxel_deduplicate(data, voxel_size):
    if data is None or len(data["canonical_xyz"]) == 0:
        return data
    coords = np.floor(data["canonical_xyz"] / max(voxel_size, 1e-6)).astype(np.int64)
    _, unique = np.unique(coords, axis=0, return_index=True)
    order = unique[np.argsort(data["score"][unique])[::-1]]
    return {key: value[order] for key, value in data.items() if isinstance(value, np.ndarray) and value.shape[0] == len(data["canonical_xyz"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", default="0,20,40,60,80")
    parser.add_argument("--views", default="0,8,16,24,32,40,48,56")
    parser.add_argument("--output", default="output/vggt_garment_candidates")
    parser.add_argument("--model-name", default="facebook/VGGT-1B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-points-per-view", type=int, default=4096)
    parser.add_argument("--max-candidates", type=int, default=2048)
    parser.add_argument("--confidence-percentile", type=float, default=60.0)
    parser.add_argument("--surface-threshold", type=float, default=0.018)
    parser.add_argument("--consistency-scale", type=float, default=0.035)
    parser.add_argument("--offset-transfer", type=float, default=0.45)
    parser.add_argument("--max-offset-ratio", type=float, default=0.12)
    parser.add_argument("--voxel-size-ratio", type=float, default=0.012)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    device = torch.device(args.device)
    model = VGGT.from_pretrained(args.model_name).to(device).eval()
    merged = []
    for frame_id in parse_int_list(args.frames):
        result = process_frame(
            model, repo_root, args.sequence, frame_id, parse_int_list(args.views), device, args
        )
        if result is not None:
            merged.append(result)
        print(f"[VGGT] sequence={args.sequence} frame={frame_id} candidates={0 if result is None else len(result['canonical_xyz'])}", flush=True)
    if not merged:
        raise RuntimeError("VGGT produced no garment candidates")
    keys = ["canonical_xyz", "reference_xyz", "confidence", "score", "source_view", "nearest_smpl_vertex", "frame_id", "colors"]
    data = {key: np.concatenate([item[key] for item in merged], axis=0) for key in keys}
    data = voxel_deduplicate(data, args.voxel_size_ratio * np.linalg.norm(
        data["canonical_xyz"].max(axis=0) - data["canonical_xyz"].min(axis=0)
    ))
    order = np.argsort(data["score"])[::-1][:args.max_candidates]
    data = {key: value[order] for key, value in data.items()}
    output_dir = repo_root / args.output / args.sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "candidates.npz", **data)
    metadata = {
        "sequence": args.sequence,
        "frames": parse_int_list(args.frames),
        "views": parse_int_list(args.views),
        "model_name": args.model_name,
        "candidate_count": int(len(data["canonical_xyz"])),
        "proposal_only": True,
        "rgb_mask_losses_decide_validity": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[VGGT] saved={output_dir / 'candidates.npz'} count={metadata['candidate_count']}", flush=True)


if __name__ == "__main__":
    main()
