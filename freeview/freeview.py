#!/usr/bin/env python3
"""
Free-view orbit video renderer for SeqAvatar DNA-Rendering sequences.

Edit the settings below, then run:
    python /media/image/mxz/human/SeqAvatar/freeview/freeview.py
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from visual_effects import compose_torch_render, visual_metadata


# ================= User Settings =================
GPU_ID = "2"
DEFAULT_SEQUENCES = ["0019", "0044", "0051", "0206", "0813"]
DEFAULT_EXPERIMENT = "part_moe_leg"
DEFAULT_RUN = "latest"
DEFAULT_ITERATION = 25000

DEFAULT_SPLIT = "train"
DEFAULT_REFERENCE_SPLIT = "train"
DEFAULT_REFERENCE_INDEX = 0
DEFAULT_TARGET_MODE = "smpl_center"
DEFAULT_TARGET = [0.0, 0.0, 0.0]
DEFAULT_MAX_FRAMES = 100

DEFAULT_YAW_START = 0.0
DEFAULT_YAW_END = 360.0
DEFAULT_INCLUDE_ENDPOINT = False
DEFAULT_RADIUS = None
DEFAULT_RADIUS_SCALE = 0.65 #人物大小
DEFAULT_HEIGHT = None
DEFAULT_HEIGHT_OFFSET = 0.6
DEFAULT_BASE_YAW_OFFSET = 0.0

DEFAULT_FPS = 24
DEFAULT_QUALITY = 8
DEFAULT_SAVE_FRAMES = True
DEFAULT_MAKE_VIDEO = True
DEFAULT_OVERWRITE = False

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "DNA-Rendering"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "output" / "DNA-Rendering"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "freeview" / "DNA-Rendering"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SeqAvatar free-view orbit videos.")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_subdir", default="freeview")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run", default=DEFAULT_RUN, help="auto_best, latest, or an explicit run folder.")
    parser.add_argument("--iteration", type=int, default=DEFAULT_ITERATION)
    parser.add_argument("--gpu", default=GPU_ID)

    parser.add_argument("--split", choices=["train", "novelview"], default=DEFAULT_SPLIT)
    parser.add_argument("--reference_split", choices=["train", "novelview"], default=DEFAULT_REFERENCE_SPLIT)
    parser.add_argument("--reference_index", type=int, default=DEFAULT_REFERENCE_INDEX)
    parser.add_argument("--pose_start", type=int, default=None)
    parser.add_argument("--pose_end", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)

    parser.add_argument("--yaw_start", type=float, default=DEFAULT_YAW_START)
    parser.add_argument("--yaw_end", type=float, default=DEFAULT_YAW_END)
    parser.add_argument("--include_endpoint", action="store_true", default=DEFAULT_INCLUDE_ENDPOINT)
    parser.add_argument("--base_yaw_offset", type=float, default=DEFAULT_BASE_YAW_OFFSET)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--radius_scale", type=float, default=DEFAULT_RADIUS_SCALE)
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT)
    parser.add_argument("--height_offset", type=float, default=DEFAULT_HEIGHT_OFFSET)
    parser.add_argument("--target", nargs=3, type=float, default=DEFAULT_TARGET)
    parser.add_argument("--target_mode", choices=["origin", "smpl_center"], default=DEFAULT_TARGET_MODE)

    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--video_name", default="freeview.mp4")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=DEFAULT_SAVE_FRAMES)
    parser.add_argument("--make_video", action=argparse.BooleanOptionalAction, default=DEFAULT_MAKE_VIDEO)
    parser.add_argument("--visual_background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual_background_path", type=Path, default=None)
    parser.add_argument("--visual_background_fit", choices=["cover", "contain"], default="cover")
    parser.add_argument("--visual_shadow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual_shadow_offset", nargs=2, type=int, default=[46, 30])
    parser.add_argument("--visual_shadow_blur", type=float, default=14.0)
    parser.add_argument("--visual_shadow_opacity", type=float, default=0.33)
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # SeqAvatar model/pipeline defaults matching DNA scripts.
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--white_background", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smpl_type", default="smplx")
    parser.add_argument("--actor_gender", default="neutral")
    parser.add_argument("--motion_offset_flag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--non_rigid_flag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nonrigid_poseconds_flag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nonrigid_deltaposeconds_flag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nonrigid_deltaxyzconds_flag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seq_xyz_knn", type=int, default=8)
    parser.add_argument("--time_step_num", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--max_time_step", type=int, default=3)
    parser.add_argument("--minimal_time_step", type=int, default=1)
    parser.add_argument("--convert_SHs_python", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compute_cov3D_python", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def import_seqavatar_modules():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    python_bin_dir = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin_dir}:{os.environ.get('PATH', '')}"
    os.chdir(PROJECT_ROOT)

    import imageio.v2 as imageio
    import torch
    from gaussian_renderer import GaussianModel, render
    from scene import Scene
    from utils.general_utils import safe_state
    from utils.graphics_utils import getWorld2View2

    return imageio, torch, GaussianModel, render, Scene, safe_state, getWorld2View2


def resolve_sequence(sequence: str, dataset_root: Path) -> str:
    exact = dataset_root / sequence
    if exact.is_dir():
        return sequence
    matches = sorted(path.name for path in dataset_root.glob(f"{sequence}_*") if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No sequence matching {sequence} under {dataset_root}")
    raise ValueError(f"Ambiguous sequence prefix {sequence}: {', '.join(matches)}")


def checkpoint_exists(run_dir: Path, iteration: int) -> bool:
    return (
        (run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply").exists()
        and (run_dir / "mlp_ckpt" / f"iteration_{iteration}" / "ckpt.pth").exists()
    )


def read_metric(run_dir: Path, iteration: int) -> dict | None:
    metric_path = run_dir / "metrics" / f"results_novelview_{iteration}.json"
    if not metric_path.exists():
        return None
    try:
        with metric_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def select_run(sequence: str, args: argparse.Namespace) -> Path:
    exp_dir = args.model_root / sequence / args.experiment
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Missing experiment directory: {exp_dir}")

    candidates = [path for path in sorted(exp_dir.iterdir()) if path.is_dir()]
    candidates = [path for path in candidates if checkpoint_exists(path, args.iteration)]
    if not candidates:
        raise FileNotFoundError(f"No run under {exp_dir} has iteration {args.iteration}")

    if args.run not in {"auto_best", "latest"}:
        run_dir = exp_dir / args.run
        if not checkpoint_exists(run_dir, args.iteration):
            raise FileNotFoundError(f"Run lacks iteration {args.iteration}: {run_dir}")
        return run_dir

    if args.run == "latest":
        return sorted(candidates, key=lambda path: (path.name, path.stat().st_mtime))[-1]

    scored = []
    for run_dir in candidates:
        metric = read_metric(run_dir, args.iteration)
        if metric is None:
            continue
        psnr = float(metric.get("PSNR", float("-inf")))
        ssim = float(metric.get("SSIM", float("-inf")))
        lpips = float(metric.get("LPIPS", float("inf")))
        scored.append((psnr, ssim, -lpips, run_dir.name, run_dir))

    if scored:
        return max(scored, key=lambda item: (item[0], item[1], item[2], item[3]))[4]
    return sorted(candidates, key=lambda path: (path.name, path.stat().st_mtime))[-1]


def load_model_cfg_args(model_path: Path) -> Namespace | None:
    cfg_path = Path(model_path) / "cfg_args"
    if not cfg_path.exists():
        return None
    text = cfg_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    expr = ast.parse(text, mode="eval")
    if not isinstance(expr.body, ast.Call) or getattr(expr.body.func, "id", "") != "Namespace":
        raise ValueError(f"Unsupported cfg_args format: {cfg_path}")
    values = {}
    for keyword in expr.body.keywords:
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return Namespace(**values)


def build_dataset_args(args: argparse.Namespace, source_path: Path, model_path: Path) -> Namespace:
    cfg = load_model_cfg_args(model_path)
    if cfg is None:
        cfg = Namespace(
            sh_degree=args.sh_degree,
            images=args.images,
            resolution=args.resolution,
            white_background=args.white_background,
            data_device=args.data_device,
            eval=args.eval,
            exp_name="",
            smpl_type=args.smpl_type,
            actor_gender=args.actor_gender,
            motion_offset_flag=args.motion_offset_flag,
            non_rigid_flag=args.non_rigid_flag,
            nonrigid_poseconds_flag=args.nonrigid_poseconds_flag,
            nonrigid_deltaposeconds_flag=args.nonrigid_deltaposeconds_flag,
            nonrigid_deltaxyzconds_flag=args.nonrigid_deltaxyzconds_flag,
            seq_xyz_knn=args.seq_xyz_knn,
            time_step_num=args.time_step_num,
            seq_len=args.seq_len,
            max_time_step=args.max_time_step,
            minimal_time_step=args.minimal_time_step,
        )
    cfg.source_path = str(source_path.resolve())
    cfg.model_path = str(model_path)
    return cfg


def build_pipeline_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        convert_SHs_python=args.convert_SHs_python,
        compute_cov3D_python=args.compute_cov3D_python,
        debug=args.debug,
    )


def flatten_views(views) -> list:
    if isinstance(views, dict):
        merged = []
        for key in sorted(views.keys()):
            merged.extend(views[key])
        return merged
    return list(views)


def get_views(scene, split: str) -> list:
    if split == "train":
        return flatten_views(scene.getTrainCameras())
    test_views = scene.getTestCameras()
    if isinstance(test_views, dict) and "novelview" in test_views:
        return list(test_views["novelview"])
    return flatten_views(test_views)


def unique_pose_views(views: Iterable, args: argparse.Namespace) -> list:
    unique = []
    seen = set()
    for view in sorted(views, key=lambda item: int(item.pose_id)):
        pose_id = int(view.pose_id)
        if pose_id in seen:
            continue
        if args.pose_start is not None and pose_id < args.pose_start:
            continue
        if args.pose_end is not None and pose_id > args.pose_end:
            continue
        seen.add(pose_id)
        unique.append(view)

    unique = unique[:: max(1, int(args.frame_stride))]
    if args.max_frames is not None and args.max_frames > 0:
        unique = unique[: args.max_frames]
    return unique


def get_look_at_rotation(camera_position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
    up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = target_position - camera_position
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_axis = z_axis / z_norm

    x_axis = np.cross(up_vector, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    rotation = np.eye(3, dtype=np.float64)
    rotation[0, :] = x_axis
    rotation[1, :] = y_axis
    rotation[2, :] = z_axis
    return rotation.transpose()


def tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pose_target(gaussians, pose_id: int, args: argparse.Namespace) -> np.ndarray:
    if getattr(args, "target_mode", "origin") == "origin":
        return np.asarray(args.target, dtype=np.float64)

    smpl_params = gaussians.smpl_params_dict.get(pose_id)
    if smpl_params is None or "obs_xyz" not in smpl_params:
        return np.asarray(args.target, dtype=np.float64)

    obs_xyz = tensor_to_numpy(smpl_params["obs_xyz"]).reshape(-1, 3)
    return obs_xyz.mean(axis=0).astype(np.float64) + np.asarray(args.target, dtype=np.float64)


def prepare_output_dir(sequence: str, run_dir: Path, args: argparse.Namespace) -> Path:
    output_subdir = getattr(args, "output_subdir", None)
    if output_subdir:
        out_dir = args.output_root / sequence / output_subdir / args.timestamp
    else:
        out_dir = args.output_root / sequence / args.timestamp

    if out_dir.exists() and args.overwrite:
        for path in sorted(out_dir.glob("**/*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_render_info(out_dir: Path, metadata: dict) -> None:
    lines = [
        "SeqAvatar free-view render",
        "",
        f"sequence: {metadata['sequence']}",
        f"dataset: {metadata['dataset']}",
        f"model: {metadata['model']}",
        f"experiment: {metadata['experiment']}",
        f"run: {metadata['run']}",
        f"iteration: {metadata['iteration']}",
        "",
        f"split: {metadata['split']}",
        f"reference_split: {metadata['reference_split']}",
        f"reference_index: {metadata['reference_index']}",
        f"target_mode: {metadata['target_mode']}",
        f"target: {metadata['target']}",
        "",
        f"frame_count: {metadata['frame_count']}",
        f"fps: {metadata['fps']}",
        f"yaw_start: {metadata['yaw_start']}",
        f"yaw_end: {metadata['yaw_end']}",
        f"base_yaw: {metadata['base_yaw']}",
        f"radius: {metadata['radius']}",
        f"radius_scale: {metadata.get('radius_scale')}",
        f"height: {metadata['height']}",
        f"height_offset: {metadata['height_offset']}",
        "",
        f"video: {metadata['video']}",
        f"frames_dir: {metadata['frames_dir']}",
        "",
        "metrics:",
    ]
    metric = metadata.get("metric")
    if metric:
        for key in sorted(metric.keys()):
            lines.append(f"  {key}: {metric[key]}")
    else:
        lines.append("  unavailable")
    lines.extend(["", "command:", f"  {metadata['command']}"])
    with (out_dir / "render_info.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def render_sequence(
    sequence: str,
    args: argparse.Namespace,
    imageio,
    torch,
    GaussianModel,
    render,
    Scene,
    getWorld2View2,
) -> Path:
    source_path = args.dataset_root / sequence
    run_dir = select_run(sequence, args)
    dataset = build_dataset_args(args, source_path, run_dir)
    pipeline = build_pipeline_args(args)

    print(f"\n[{sequence}] dataset: {source_path}")
    print(f"[{sequence}] model:   {run_dir}")
    metric = read_metric(run_dir, args.iteration)
    if metric:
        print(
            f"[{sequence}] metric@{args.iteration}: "
            f"PSNR={metric.get('PSNR'):.4f} SSIM={metric.get('SSIM'):.5f} LPIPS={metric.get('LPIPS'):.5f}"
        )

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.smpl_type,
        dataset.motion_offset_flag,
        dataset.actor_gender,
        dataset,
    )
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    motion_views = unique_pose_views(get_views(scene, args.split), args)
    reference_views = get_views(scene, args.reference_split)
    if not motion_views:
        raise RuntimeError(f"No motion frames selected for {sequence}")
    if not reference_views:
        raise RuntimeError(f"No reference cameras selected for {sequence}")
    if args.reference_index < 0 or args.reference_index >= len(reference_views):
        raise IndexError(f"reference_index {args.reference_index} out of range 0..{len(reference_views)-1}")

    reference_cam = reference_views[args.reference_index]
    ref_center = tensor_to_numpy(reference_cam.camera_center).astype(np.float64)
    ref_target = pose_target(gaussians, int(reference_cam.pose_id), args)
    ref_delta = ref_center - ref_target
    base_radius = math.hypot(float(ref_delta[0]), float(ref_delta[2]))
    if base_radius < 1e-8:
        base_radius = 1.0
    radius = args.radius if args.radius is not None else base_radius * args.radius_scale
    base_height_offset = float(ref_delta[1])
    base_yaw = math.degrees(math.atan2(float(ref_delta[0]), float(ref_delta[2]))) + args.base_yaw_offset

    frame_count = len(motion_views)
    yaws = np.linspace(args.yaw_start, args.yaw_end, frame_count, endpoint=args.include_endpoint)

    out_dir = prepare_output_dir(sequence, run_dir, args)
    frames_dir = out_dir / "frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / args.video_name

    bg_color = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    writer = None
    if args.make_video:
        writer = imageio.get_writer(str(video_path), fps=args.fps, quality=args.quality, macro_block_size=None)

    from tqdm import tqdm

    print(
        f"[{sequence}] rendering {frame_count} frames, radius={radius:.4f}, "
        f"base_yaw={base_yaw:.2f}, output={out_dir}"
    )

    with torch.no_grad():
        for frame_idx, (base_view, yaw_offset) in enumerate(
            tqdm(list(zip(motion_views, yaws)), desc=f"Freeview {sequence}")
        ):
            pose_id = int(base_view.pose_id)
            target = pose_target(gaussians, pose_id, args)
            yaw = math.radians(base_yaw + float(yaw_offset))
            pos_y = args.height if args.height is not None else target[1] + base_height_offset
            camera_position = np.array(
                [
                    target[0] + radius * math.sin(yaw),
                    pos_y + args.height_offset,
                    target[2] + radius * math.cos(yaw),
                ],
                dtype=np.float64,
            )

            view = copy.deepcopy(base_view)
            view.uid = 900000 + frame_idx
            view.image_name = f"freeview_{frame_idx:04d}_pose{pose_id:06d}_yaw{base_yaw + float(yaw_offset):07.2f}"
            view.R = get_look_at_rotation(camera_position, target)
            view.T = -np.dot(view.R.transpose(), camera_position)
            view.world_view_transform = torch.tensor(
                getWorld2View2(view.R, view.T, view.trans, view.scale),
                dtype=torch.float32,
                device="cuda",
            ).transpose(0, 1)
            view.full_proj_transform = (
                view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))
            ).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3, :3]

            render_pkg = render(view, gaussians, pipeline, background)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            frame = compose_torch_render(
                image,
                render_pkg.get("render_alpha"),
                background_path=args.visual_background_path,
                enable_background=args.visual_background,
                enable_shadow=args.visual_shadow,
                background_fit=args.visual_background_fit,
                shadow_offset=args.visual_shadow_offset,
                shadow_blur=args.visual_shadow_blur,
                shadow_opacity=args.visual_shadow_opacity,
            )

            if args.save_frames:
                imageio.imwrite(str(frames_dir / f"{view.image_name}.png"), frame)
            if writer is not None:
                writer.append_data(frame)

    if writer is not None:
        writer.close()

    metadata = {
        "sequence": sequence,
        "dataset": str(source_path),
        "model": str(run_dir),
        "experiment": args.experiment,
        "run": run_dir.name,
        "iteration": args.iteration,
        "split": args.split,
        "reference_split": args.reference_split,
        "reference_index": args.reference_index,
        "target_mode": args.target_mode,
        "target": list(args.target),
        "radius": radius,
        "radius_scale": args.radius_scale,
        "base_yaw": base_yaw,
        "yaw_start": args.yaw_start,
        "yaw_end": args.yaw_end,
        "height": args.height,
        "height_offset": args.height_offset,
        "frame_count": frame_count,
        "fps": args.fps,
        "video": str(video_path) if args.make_video else None,
        "frames_dir": str(frames_dir) if args.save_frames else None,
        "metric": metric,
        "visual_effects": visual_metadata(args),
        "command": " ".join([Path(sys.executable).name] + sys.argv),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    write_render_info(out_dir, metadata)

    print(f"[{sequence}] done: {video_path if args.make_video else out_dir}")
    return out_dir


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.model_root = args.model_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sequences = [resolve_sequence(sequence, args.dataset_root) for sequence in args.sequences]
    print("Resolved sequences:", ", ".join(sequences))
    for sequence in sequences:
        run_dir = select_run(sequence, args)
        print(f"[dry] {sequence}: {run_dir}")
    if args.dry_run:
        return

    imageio, torch, GaussianModel, render, Scene, safe_state, getWorld2View2 = import_seqavatar_modules()
    safe_state(args.quiet)

    for sequence in sequences:
        render_sequence(sequence, args, imageio, torch, GaussianModel, render, Scene, getWorld2View2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
