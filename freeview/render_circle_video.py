#!/usr/bin/env python3
"""
Render one selected SeqAvatar pose as a circle-view grid video.

This script keeps the camera and visual composition behavior close to
freeview/render_circle.py, but packs the circle views into one grid frame.
"""

import argparse
import copy
import json
import math
import os
import pickle
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# ================= User Settings =================
GPU_ID = "2"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
PYTHON_BIN_DIR = Path(sys.executable).resolve().parent
os.environ["PATH"] = f"{PYTHON_BIN_DIR}:{os.environ.get('PATH', '')}"

from arguments import ModelParams, PipelineParams, get_combined_args
from freeview import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_ROOT,
    build_dataset_args,
    build_pipeline_args,
    prepare_output_dir,
    read_metric,
    resolve_sequence,
    select_run,
)
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2
from visual_effects import (
    alpha_to_l_pil,
    compose_torch_render,
    fit_background,
    make_default_stage_background,
    visual_metadata,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "freeview" / "DNA-Rendering"
DEFAULT_SEQUENCES = ["0019"]
DEFAULT_BASE_YAW_OFFSET = 180.0


def serializable_args(args):
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def get_look_at_rotation(camera_position, target_position, up_vector=np.array([0, 1, 0])):
    z_axis = target_position - camera_position
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0, 0, 1], dtype=np.float64)
    else:
        z_axis = z_axis / z_norm

    x_axis = np.cross(up_vector, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        x_axis = np.array([1, 0, 0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    rotation = np.eye(3, dtype=np.float64)
    rotation[0, :] = x_axis
    rotation[1, :] = y_axis
    rotation[2, :] = z_axis
    return rotation.transpose()


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def make_tile_background(tile_size, background_path=None, fit_mode="cover"):
    if background_path:
        background = Image.open(background_path).convert("RGB")
    else:
        background = make_default_stage_background()
    if fit_mode == "stretch":
        return background.resize(tile_size, Image.Resampling.LANCZOS)
    return fit_background(background, tile_size, fit=fit_mode)


def resize_to_tile(pil_img, tile_size, fit_mode="contain", padding_background=None):
    tile_w, tile_h = tile_size
    src_w, src_h = pil_img.size
    if src_w <= 0 or src_h <= 0:
        return padding_background.copy() if padding_background is not None else Image.new("RGB", tile_size, (0, 0, 0))

    if fit_mode == "stretch":
        return pil_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)

    scale = max(tile_w / src_w, tile_h / src_h) if fit_mode == "cover" else min(tile_w / src_w, tile_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    pil_img = pil_img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

    if fit_mode == "cover":
        left = max(0, (resized_w - tile_w) // 2)
        top = max(0, (resized_h - tile_h) // 2)
        return pil_img.crop((left, top, left + tile_w, top + tile_h))

    canvas = padding_background.copy() if padding_background is not None else Image.new("RGB", tile_size, (0, 0, 0))
    canvas.paste(pil_img, ((tile_w - resized_w) // 2, (tile_h - resized_h) // 2))
    return canvas


def crop_with_margin(pil_img, alpha_pil, margin=0.12, threshold=8):
    if alpha_pil is None:
        return pil_img, alpha_pil

    alpha_arr = np.asarray(alpha_pil)
    ys, xs = np.where(alpha_arr > int(threshold))
    if xs.size == 0 or ys.size == 0:
        return pil_img, alpha_pil

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    pad_x = int(round(box_w * float(margin)))
    pad_y = int(round(box_h * float(margin)))

    width, height = pil_img.size
    bbox = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    return pil_img.crop(bbox), alpha_pil.crop(bbox)


def resolve_tile_size(grid_rows, grid_cols, output_width, output_height, tile_width, tile_height):
    if output_width is not None and output_height is not None:
        return max(1, int(round(output_width / grid_cols))), max(1, int(round(output_height / grid_rows)))
    return max(1, int(tile_width)), max(1, int(tile_height))


def make_grid(
    pil_images,
    grid_rows,
    grid_cols,
    tile_size,
    output_size=None,
    fit_mode="contain",
    alpha_images=None,
    crop_background=False,
    crop_margin=0.12,
    crop_threshold=8,
    tile_padding="black",
    tile_background_path=None,
    tile_background_fit="stretch",
):
    tile_w, tile_h = tile_size
    padding_background = None
    if tile_padding == "background":
        padding_background = make_tile_background(tile_size, tile_background_path, tile_background_fit)

    grid_bg = padding_background if padding_background is not None else Image.new("RGB", tile_size, (0, 0, 0))
    grid = Image.new("RGB", (tile_w * grid_cols, tile_h * grid_rows), (0, 0, 0))
    for row in range(grid_rows):
        for col in range(grid_cols):
            grid.paste(grid_bg, (col * tile_w, row * tile_h))

    if alpha_images is None:
        alpha_images = [None] * len(pil_images)
    for idx, pil_img in enumerate(pil_images[: grid_rows * grid_cols]):
        row = idx // grid_cols
        col = idx % grid_cols
        alpha_pil = alpha_images[idx] if idx < len(alpha_images) else None
        if crop_background:
            pil_img, alpha_pil = crop_with_margin(
                pil_img,
                alpha_pil,
                margin=crop_margin,
                threshold=crop_threshold,
            )
        tile = resize_to_tile(
            pil_img,
            (tile_w, tile_h),
            fit_mode=fit_mode,
            padding_background=padding_background,
        )
        grid.paste(tile, (col * tile_w, row * tile_h))
    if output_size is not None and grid.size != output_size:
        grid = grid.resize(output_size, Image.Resampling.LANCZOS)
    return grid


def get_split_views(scene, split):
    if split == "train":
        views = scene.getTrainCameras()
        if isinstance(views, dict):
            key = list(views.keys())[0]
            return "train", views[key]
        return "train", views

    test_views = scene.getTestCameras()
    if isinstance(test_views, dict):
        key = list(test_views.keys())[0]
        return key, test_views[key]
    return "test", test_views


def unique_pose_views(views):
    selected = []
    seen = set()
    for view in sorted(views, key=lambda item: int(item.pose_id)):
        pose_id = int(view.pose_id)
        if pose_id not in seen:
            selected.append(view)
            seen.add(pose_id)
    return selected


def select_pose_view(views, pose_id=None, frame_index=None):
    unique_views = unique_pose_views(views)
    if not unique_views:
        raise RuntimeError("No cameras available for pose selection.")

    if frame_index is not None:
        if frame_index < 0 or frame_index >= len(unique_views):
            raise IndexError(f"frame_index {frame_index} out of range 0..{len(unique_views)-1}")
        return unique_views[frame_index]

    pose_id = 0 if pose_id is None else int(pose_id)
    for view in unique_views:
        if int(view.pose_id) == pose_id:
            return view
    available = [int(view.pose_id) for view in unique_views[:10]]
    raise KeyError(f"pose_id {pose_id} not found. First available pose ids: {available}")


def write_metadata(out_dir, metadata):
    out_dir = Path(out_dir)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    lines = [
        "SeqAvatar render_circle_video render",
        "",
        f"sequence: {metadata.get('sequence')}",
        f"dataset: {metadata.get('dataset')}",
        f"model: {metadata.get('model')}",
        f"experiment: {metadata.get('experiment')}",
        f"run: {metadata.get('run')}",
        f"iteration: {metadata.get('iteration')}",
        "",
        f"reference_split: {metadata.get('reference_split')}",
        f"reference_camera: {metadata.get('reference_camera')}",
        f"motion_split: {metadata.get('motion_split')}",
        f"pose_id: {metadata.get('pose_id')}",
        f"frame_index: {metadata.get('frame_index')}",
        f"num_views: {metadata.get('num_views')}",
        f"angle_start: {metadata.get('angle_start')}",
        f"angle_end: {metadata.get('angle_end')}",
        f"radius: {metadata.get('radius')}",
        f"radius_scale: {metadata.get('radius_scale')}",
        f"height: {metadata.get('height')}",
        f"height_offset: {metadata.get('height_offset')}",
        f"base_angle: {metadata.get('base_angle')}",
        f"base_yaw_offset: {metadata.get('base_yaw_offset')}",
        f"grid_rows: {metadata.get('grid_rows')}",
        f"grid_cols: {metadata.get('grid_cols')}",
        f"tile_width: {metadata.get('tile_width')}",
        f"tile_height: {metadata.get('tile_height')}",
        f"tile_fit_mode: {metadata.get('tile_fit_mode')}",
        f"tile_padding: {metadata.get('tile_padding')}",
        f"tile_background_fit: {metadata.get('tile_background_fit')}",
        f"crop_background: {metadata.get('crop_background')}",
        f"crop_margin: {metadata.get('crop_margin')}",
        f"crop_threshold: {metadata.get('crop_threshold')}",
        "",
        f"grid_frame: {metadata.get('grid_frame')}",
        f"video: {metadata.get('video')}",
        f"view_frames_dir: {metadata.get('view_frames_dir')}",
        "",
        "command:",
        f"  {metadata.get('command')}",
    ]
    with (out_dir / "render_info.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def apply_dna_defaults(args):
    def set_default(name, value):
        if getattr(args, name, None) is None:
            setattr(args, name, value)

    if args.iteration in (None, -1):
        args.iteration = 25000
    set_default("sh_degree", 3)
    set_default("images", "images")
    set_default("resolution", -1)
    set_default("white_background", False)
    set_default("data_device", "cuda")
    set_default("eval", True)
    set_default("smpl_type", "smplx")
    set_default("actor_gender", "neutral")
    set_default("motion_offset_flag", True)
    set_default("non_rigid_flag", True)
    set_default("nonrigid_poseconds_flag", True)
    set_default("nonrigid_deltaposeconds_flag", True)
    set_default("nonrigid_deltaxyzconds_flag", True)
    set_default("seq_len", 8)
    set_default("seq_xyz_knn", 8)
    set_default("time_step_num", 3)
    set_default("max_time_step", 3)
    set_default("minimal_time_step", 1)


def render_circle_grid(dataset, iteration, pipeline, background, scene, gaussians):
    reference_data_name, reference_views = get_split_views(scene, dataset.render_reference_split)
    motion_data_name, motion_views = get_split_views(scene, dataset.render_motion_split)

    reference_index = int(dataset.render_reference_index)
    if reference_index < 0 or reference_index >= len(reference_views):
        raise IndexError(f"reference_index {reference_index} out of range 0..{len(reference_views)-1}")
    ref_cam = reference_views[reference_index]
    motion_cam = select_pose_view(motion_views, dataset.render_pose_id, dataset.render_frame_index)
    pose_id = int(motion_cam.pose_id)

    output_dir = Path(dataset.render_output_dir)
    frames_dir = output_dir / "frames"
    view_frames_dir = output_dir / "view_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset.render_save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if dataset.render_save_view_frames:
        view_frames_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(dataset.model_path) / "smpl_rot" / f"iteration_{iteration}" / "smpl_rot.pickle", "rb") as handle:
        smpl_rot = pickle.load(handle)
    cache_data_name = motion_data_name
    if cache_data_name not in smpl_rot:
        cache_data_name = None

    cached_pose = smpl_rot.get(cache_data_name, {}).get(pose_id) if cache_data_name is not None else None
    cache_used = cached_pose is not None
    if cache_used:
        d_nonrigid = cached_pose["d_nonrigid"]
        transforms = cached_pose["transforms"]
        translation = cached_pose["translation"]
    else:
        print(f"[Circle Grid] No cached smpl_rot for {motion_data_name} pose_id={pose_id}; rendering deformation on the fly.")
        d_nonrigid = None
        transforms = None
        translation = None

    ref_center = tensor_to_numpy(ref_cam.camera_center).astype(np.float64)
    radius = np.linalg.norm(np.array([ref_center[0], ref_center[2]], dtype=np.float64))
    radius = radius * float(dataset.render_radius_scale)
    height = float(ref_center[1]) + float(dataset.render_height_offset)
    target_pos = np.asarray(dataset.render_target, dtype=np.float64)
    base_angle = math.degrees(math.atan2(float(ref_center[0]), float(ref_center[2]))) + float(dataset.render_base_yaw_offset)

    num_views = int(dataset.render_num_views)
    grid_rows = int(dataset.render_grid_rows)
    grid_cols = int(dataset.render_grid_cols)
    if grid_rows * grid_cols < num_views:
        raise ValueError(f"grid_rows * grid_cols must be >= num_views, got {grid_rows}x{grid_cols} for {num_views}")
    angles = np.linspace(float(dataset.render_angle_start), float(dataset.render_angle_end), num_views)

    tile_w, tile_h = resolve_tile_size(
        grid_rows,
        grid_cols,
        dataset.render_output_width,
        dataset.render_output_height,
        dataset.render_tile_width,
        dataset.render_tile_height,
    )
    output_size = None
    if dataset.render_output_width is not None and dataset.render_output_height is not None:
        output_size = (int(dataset.render_output_width), int(dataset.render_output_height))

    video_path = output_dir / dataset.render_video_name
    grid_frame_path = frames_dir / f"circle_grid_pose{pose_id:06d}.png"

    metadata = {
        "sequence": dataset.render_sequence,
        "dataset": dataset.source_path,
        "model": dataset.model_path,
        "experiment": dataset.render_experiment,
        "run": dataset.render_run,
        "iteration": iteration,
        "renderer": "render_circle_video",
        "status": "started",
        "reference_split": reference_data_name,
        "reference_camera": ref_cam.image_name,
        "reference_index": reference_index,
        "motion_split": motion_data_name,
        "smpl_rot_cache_key": cache_data_name,
        "smpl_rot_cache_used": cache_used,
        "pose_id": pose_id,
        "frame_index": dataset.render_frame_index,
        "num_views": num_views,
        "angle_start": float(angles[0]),
        "angle_end": float(angles[-1]),
        "radius": float(radius),
        "radius_scale": float(dataset.render_radius_scale),
        "height": float(height),
        "height_offset": float(dataset.render_height_offset),
        "target": list(target_pos),
        "base_angle": float(base_angle),
        "base_yaw_offset": float(dataset.render_base_yaw_offset),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "output_width": output_size[0] if output_size else tile_w * grid_cols,
        "output_height": output_size[1] if output_size else tile_h * grid_rows,
        "tile_width": tile_w,
        "tile_height": tile_h,
        "tile_fit_mode": dataset.render_tile_fit_mode,
        "tile_padding": dataset.render_tile_padding,
        "tile_background_fit": dataset.render_tile_background_fit,
        "crop_background": bool(dataset.render_crop_background),
        "crop_margin": float(dataset.render_crop_margin),
        "crop_threshold": int(dataset.render_crop_threshold),
        "fps": dataset.render_fps,
        "repeat_frames": dataset.render_repeat_frames,
        "grid_frame": str(grid_frame_path) if dataset.render_save_frames else None,
        "video": str(video_path) if dataset.render_make_video else None,
        "view_frames_dir": str(view_frames_dir) if dataset.render_save_view_frames else None,
        "metric": dataset.render_metric,
        "parameters": dataset.render_parameters,
        "visual_effects": dataset.visual_effects,
        "command": dataset.render_command,
    }
    write_metadata(output_dir, metadata)

    print(f"\n[Circle Grid] sequence={dataset.render_sequence} pose_id={pose_id}")
    print(f"[Circle Grid] reference={reference_data_name}:{ref_cam.image_name}")
    print(f"[Circle Grid] output={output_dir}")
    print(f"[Circle Grid] radius={radius:.4f}, height={height:.4f}, base_angle={base_angle:.2f}")

    pil_views = []
    alpha_views = []
    with torch.no_grad():
        for view_idx, angle_deg in enumerate(tqdm(angles, desc="Rendering circle grid views")):
            current_angle = base_angle + float(angle_deg)
            rad = math.radians(current_angle)
            camera_position = np.array(
                [
                    radius * math.sin(rad),
                    height,
                    radius * math.cos(rad),
                ],
                dtype=np.float64,
            )

            view_cam = copy.deepcopy(ref_cam)
            view_cam.uid = 300000 + view_idx
            view_cam.pose_id = pose_id
            view_cam.image_name = f"circle_grid_pose{pose_id:06d}_view{view_idx:03d}_angle{current_angle:07.2f}"
            view_cam.R = get_look_at_rotation(camera_position, target_pos)
            view_cam.T = -np.dot(view_cam.R.transpose(), camera_position)
            view_cam.world_view_transform = torch.tensor(
                getWorld2View2(view_cam.R, view_cam.T, view_cam.trans, view_cam.scale),
                dtype=torch.float32,
                device="cuda",
            ).transpose(0, 1)
            view_cam.full_proj_transform = (
                view_cam.world_view_transform.unsqueeze(0).bmm(view_cam.projection_matrix.unsqueeze(0))
            ).squeeze(0)
            view_cam.camera_center = view_cam.world_view_transform.inverse()[3, :3]

            render_pkg = render(
                view_cam,
                gaussians,
                pipeline,
                background,
                transforms=transforms,
                translation=translation,
                d_nonrigid=d_nonrigid,
            )
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            render_alpha = render_pkg.get("render_alpha")
            frame = compose_torch_render(
                image,
                render_alpha,
                background_path=dataset.visual_background_path,
                enable_background=dataset.visual_background,
                enable_shadow=dataset.visual_shadow,
                background_fit=dataset.visual_background_fit,
                shadow_offset=dataset.visual_shadow_offset,
                shadow_blur=dataset.visual_shadow_blur,
                shadow_opacity=dataset.visual_shadow_opacity,
            )
            pil_frame = Image.fromarray(frame)
            alpha_pil = alpha_to_l_pil(render_alpha, pil_frame.size) if render_alpha is not None else None
            pil_views.append(pil_frame)
            alpha_views.append(alpha_pil)
            if dataset.render_save_view_frames:
                pil_frame.save(view_frames_dir / f"{view_cam.image_name}.png", optimize=True)

    grid = make_grid(
        pil_views,
        grid_rows,
        grid_cols,
        (tile_w, tile_h),
        output_size=output_size,
        fit_mode=dataset.render_tile_fit_mode,
        alpha_images=alpha_views,
        crop_background=dataset.render_crop_background,
        crop_margin=dataset.render_crop_margin,
        crop_threshold=dataset.render_crop_threshold,
        tile_padding=dataset.render_tile_padding,
        tile_background_path=dataset.visual_background_path,
        tile_background_fit=dataset.render_tile_background_fit,
    )
    if dataset.render_save_frames:
        grid.save(grid_frame_path, optimize=True)

    if dataset.render_make_video:
        writer = imageio.get_writer(str(video_path), fps=dataset.render_fps, quality=dataset.render_quality, macro_block_size=None)
        for _ in range(max(1, int(dataset.render_repeat_frames))):
            writer.append_data(np.asarray(grid))
        writer.close()

    metadata["status"] = "complete"
    write_metadata(output_dir, metadata)
    print(f"[Circle Grid] done: {output_dir}")


def render_sets(dataset, iteration, pipeline):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_circle_grid(dataset, iteration, pipeline, background, scene, gaussians)


def run_sequence_mode(args, model, pipeline):
    args.dataset_root = Path(args.dataset_root).resolve()
    args.model_root = Path(args.model_root).resolve()
    args.output_root = Path(args.output_root).resolve()
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    apply_dna_defaults(args)
    sequences = [resolve_sequence(sequence, args.dataset_root) for sequence in args.sequences]
    print("Resolved sequences:", ", ".join(sequences))

    selected = []
    for sequence in sequences:
        run_dir = select_run(sequence, args)
        selected.append((sequence, run_dir))
        print(f"[dry] {sequence}: {run_dir}")
    if args.dry_run:
        return

    safe_state(args.quiet)
    for sequence, run_dir in selected:
        source_path = args.dataset_root / sequence
        out_dir = prepare_output_dir(sequence, run_dir, args)
        metric = read_metric(run_dir, args.iteration)
        dataset = build_dataset_args(args, source_path, run_dir)
        dataset.render_output_dir = str(out_dir)
        dataset.render_sequence = sequence
        dataset.render_experiment = args.experiment
        dataset.render_run = run_dir.name
        dataset.render_metric = metric
        dataset.render_parameters = serializable_args(args)
        dataset.render_command = " ".join([Path(sys.executable).name] + sys.argv)

        dataset.render_video_name = args.video_name
        dataset.render_fps = args.fps
        dataset.render_quality = args.quality
        dataset.render_save_frames = args.save_frames
        dataset.render_make_video = args.make_video
        dataset.render_save_view_frames = args.save_view_frames
        dataset.render_repeat_frames = args.repeat_frames

        dataset.render_reference_split = args.reference_split
        dataset.render_motion_split = args.motion_split
        dataset.render_reference_index = args.reference_index
        dataset.render_pose_id = args.pose_id
        dataset.render_frame_index = args.frame_index
        dataset.render_num_views = args.num_views
        dataset.render_grid_rows = args.grid_rows
        dataset.render_grid_cols = args.grid_cols
        dataset.render_angle_start = args.angle_start
        dataset.render_angle_end = args.angle_end
        dataset.render_base_yaw_offset = args.base_yaw_offset
        dataset.render_radius_scale = args.radius_scale
        dataset.render_height_offset = args.height_offset
        dataset.render_target = args.target
        dataset.render_output_width = args.output_width
        dataset.render_output_height = args.output_height
        dataset.render_tile_width = args.tile_width
        dataset.render_tile_height = args.tile_height
        dataset.render_tile_fit_mode = args.tile_fit_mode
        dataset.render_tile_padding = args.tile_padding
        dataset.render_tile_background_fit = args.tile_background_fit
        dataset.render_crop_background = args.crop_background
        dataset.render_crop_margin = args.crop_margin
        dataset.render_crop_threshold = args.crop_threshold

        dataset.visual_background = args.visual_background
        dataset.visual_background_path = str(args.visual_background_path) if args.visual_background_path else None
        dataset.visual_background_fit = args.visual_background_fit
        dataset.visual_shadow = args.visual_shadow
        dataset.visual_shadow_offset = args.visual_shadow_offset
        dataset.visual_shadow_blur = args.visual_shadow_blur
        dataset.visual_shadow_opacity = args.visual_shadow_opacity
        dataset.visual_effects = visual_metadata(args)

        print(f"\n[{sequence}] dataset: {source_path}")
        print(f"[{sequence}] model:   {run_dir}")
        print(f"[{sequence}] output:  {out_dir}")
        render_sets(dataset, args.iteration, build_pipeline_args(args))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser():
    parser = ArgumentParser(description="Render selected SeqAvatar pose as a circle-view grid video.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_subdir", default="render_circle_video")
    parser.add_argument("--experiment", default="no_depth_no_split")
    parser.add_argument("--run", default="auto_best")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--gpu", default=GPU_ID)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--video_name", default="render_circle_video.mp4")
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_view_frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repeat_frames", type=int, default=1)

    parser.add_argument("--reference_split", choices=["train", "novelview"], default="novelview")
    parser.add_argument("--motion_split", choices=["train", "novelview"], default="train")
    parser.add_argument("--reference_index", type=int, default=0)
    parser.add_argument("--pose_id", type=int, default=0)
    parser.add_argument("--frame_index", type=int, default=None)

    parser.add_argument("--num_views", type=int, default=30)
    parser.add_argument("--grid_rows", type=int, default=3)
    parser.add_argument("--grid_cols", type=int, default=10)
    parser.add_argument("--angle_start", type=float, default=-30.0)
    parser.add_argument("--angle_end", type=float, default=30.0)
    parser.add_argument("--base_yaw_offset", type=float, default=DEFAULT_BASE_YAW_OFFSET)
    parser.add_argument("--radius_scale", type=float, default=0.65)
    parser.add_argument("--height_offset", type=float, default=1)
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 0.0])

    parser.add_argument("--output_width", type=int, default=None)
    parser.add_argument("--output_height", type=int, default=None)
    parser.add_argument("--tile_width", type=int, default=216)
    parser.add_argument("--tile_height", type=int, default=384)
    parser.add_argument("--tile_fit_mode", choices=["contain", "cover", "stretch"], default="contain")
    parser.add_argument("--tile_padding", choices=["background", "black"], default="black")
    parser.add_argument("--tile_background_fit", choices=["stretch", "cover", "contain"], default="stretch")
    parser.add_argument("--crop_background", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--crop_margin", type=float, default=0.12)
    parser.add_argument("--crop_threshold", type=int, default=8)

    parser.add_argument("--visual_background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual_background_path", type=Path, default=None)
    parser.add_argument("--visual_background_fit", choices=["cover", "contain"], default="cover")
    parser.add_argument("--visual_shadow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual_shadow_offset", nargs=2, type=int, default=[46, 30])
    parser.add_argument("--visual_shadow_blur", type=float, default=14.0)
    parser.add_argument("--visual_shadow_opacity", type=float, default=0.33)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser, model, pipeline


if __name__ == "__main__":
    parser, model, pipeline = build_parser()
    raw_args = parser.parse_args()
    if raw_args.sequences is not None or (not raw_args.source_path and not raw_args.model_path):
        if raw_args.sequences is None:
            raw_args.sequences = DEFAULT_SEQUENCES
        run_sequence_mode(raw_args, None, None)
        sys.exit(0)

    args = get_combined_args(parser)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
