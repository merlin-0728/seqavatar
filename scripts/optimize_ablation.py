#!/usr/bin/env python

import os

# ------------------------- GPU 设置 -------------------------
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import argparse
import json
import pickle
import sys
import time
import subprocess
import gc
from argparse import Namespace
from pathlib import Path

SEQAVATAR_ROOT = Path(__file__).resolve().parents[1]
if str(SEQAVATAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SEQAVATAR_ROOT))

import cv2
import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import build_rotation, inverse_sigmoid, safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim

DATA_ROOT = SEQAVATAR_ROOT / "DNA-Rendering"
MODEL_ROOT = SEQAVATAR_ROOT / "output" / "DNA-Rendering"
DEFAULT_SEQUENCES = ["0007_04", "0019_10", "0044_11", "0051_09", "0206_04", "0813_05"]

if not torch.cuda.is_available():
    raise RuntimeError("optimize.py requires CUDA")

device = torch.device("cuda:0")
torch.cuda.set_device(device)
print(f"[INFO] Using physical GPU: {GPU_ID} as logical device cuda:0")


# ------------------------- 基础工具函数 -------------------------
def load_cfg_args(model_dir: Path, data_dir: Path, device_str: str) -> Namespace:
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg_args: {cfg_path}")
    cfg = eval(cfg_path.read_text(), {"Namespace": Namespace})
    cfg.model_path = str(model_dir)
    cfg.source_path = str(data_dir)
    cfg.data_device = device_str
    return cfg


def make_pipeline_args(args) -> Namespace:
    return Namespace(
        convert_SHs_python=args.convert_SHs_python,
        compute_cov3D_python=args.compute_cov3D_python,
        debug=args.debug,
    )


def make_gaussian_optimizer(gaussians: GaussianModel, args, device: torch.device):
    groups = [
        {"params": [gaussians._xyz], "lr": args.lr_xyz, "name": "xyz"},
        {"params": [gaussians._features_dc], "lr": args.lr_features, "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": args.lr_features / 20.0, "name": "f_rest"},
        {"params": [gaussians._opacity], "lr": args.lr_opacity, "name": "opacity"},
        {"params": [gaussians._scaling], "lr": args.lr_scaling, "name": "scaling"},
        {"params": [gaussians._rotation], "lr": args.lr_rotation, "name": "rotation"},
    ]
    gaussians.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device=device)
    gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device=device)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device=device)
    return gaussians.optimizer


def load_smpl_rot(model_dir: Path, iteration: int):
    smpl_rot_file = model_dir / "smpl_rot" / f"iteration_{iteration}" / "smpl_rot.pickle"
    if not smpl_rot_file.exists():
        return None
    with smpl_rot_file.open("rb") as f:
        return pickle.load(f)


def get_smpl_kwargs(smpl_rot, split_name: str, view):
    if smpl_rot is None:
        return {}
    if split_name not in smpl_rot:
        return {}
    if view.pose_id not in smpl_rot[split_name]:
        return {}
    item = smpl_rot[split_name][view.pose_id]
    return {
        "d_nonrigid": item.get("d_nonrigid"),
        "transforms": item.get("transforms"),
        "translation": item.get("translation"),
    }


def resolve_views(camera_container, preferred_split=None):
    """
    兼容 list / dict 两种返回形式
    """
    if isinstance(camera_container, dict):
        if preferred_split is not None and preferred_split in camera_container:
            return camera_container[preferred_split]
        # 常见候选
        for key in ["train", "training", "novelview", "test"]:
            if key in camera_container:
                return camera_container[key]
        # 实在不行取第一个
        return next(iter(camera_container.values()))
    return camera_container


def as_hw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor.squeeze()


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None or not torch.any(mask):
        return torch.mean(torch.abs(pred - target))
    if pred.ndim == 3:
        return torch.mean(torch.abs(pred[:, mask] - target[:, mask]))
    return torch.mean(torch.abs(pred[mask] - target[mask]))


def masked_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None or not torch.any(mask):
        return torch.mean((pred - target) ** 2)
    return torch.mean((pred[mask] - target[mask]) ** 2)


def colorize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        vals = depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        out[valid] = (np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
    color = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def colorize_error(error: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(error.shape, dtype=np.uint8)
    if np.any(valid):
        vals = error[valid]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        out[valid] = (np.clip((error[valid] - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
    color = cv2.applyColorMap(out, cv2.COLORMAP_HOT)
    color[~valid] = 0
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def save_train_visuals(
    iter_dir: Path,
    image_name: str,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    fs_depth: torch.Tensor,
    valid: torch.Tensor,
):
    rgb_dir = iter_dir / "rgb"
    depth_dir = iter_dir / "depth"
    heatmap_dir = iter_dir / "error_heatmap"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    rgb_np = rgb.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    fs_np = fs_depth.detach().cpu().numpy().astype(np.float32)
    valid_np = valid.detach().cpu().numpy().astype(bool)

    error_np = np.zeros_like(depth_np, dtype=np.float32)
    error_np[valid_np] = np.abs(depth_np[valid_np] - fs_np[valid_np])

    imageio.imwrite(rgb_dir / f"{image_name}_rgb.png", (rgb_np * 255).astype(np.uint8))
    imageio.imwrite(depth_dir / f"{image_name}_depth.png", colorize_depth(depth_np, valid_np))
    imageio.imwrite(heatmap_dir / f"{image_name}_error_heatmap.png", colorize_error(error_np, valid_np))


def save_test_visuals(
    test_dir: Path,
    image_name: str,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
):
    rgb_dir = test_dir / "rgb"
    depth_dir = test_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_np = rgb.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    valid_np = valid.detach().cpu().numpy().astype(bool)

    imageio.imwrite(rgb_dir / f"{image_name}_rgb.png", (rgb_np * 255).astype(np.uint8))
    imageio.imwrite(depth_dir / f"{image_name}_depth.png", colorize_depth(depth_np, valid_np))


def load_mask(mask_dir: Path, image_name: str, data_dir: Path, epsilon_split: float):
    """
    训练阶段用的高误差 mask。
    若外部 mask 不存在，则用训练集 3DGS 深度和 FS 深度差值现算。
    """
    candidates = [
        mask_dir / f"{image_name}_mask_high_error.npy",
        mask_dir / f"{image_name}.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path).astype(bool)

    depth_3dgs_file = data_dir / "depth" / "train" / "npy" / f"{image_name}.npy"
    depth_fs_file = data_dir / "render_depth" / "train" / "npy" / f"{image_name}.npy"
    if depth_3dgs_file.exists() and depth_fs_file.exists():
        depth_3dgs = np.load(depth_3dgs_file).astype(np.float32)
        depth_fs = np.load(depth_fs_file).astype(np.float32)
        valid = np.isfinite(depth_3dgs) & np.isfinite(depth_fs) & (depth_3dgs > 0) & (depth_fs > 0)
        return valid & (np.abs(depth_3dgs - depth_fs) > epsilon_split)

    return None


def prepare_metric_images(
    rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    metric_mask: torch.Tensor,
    bg: torch.Tensor,
):
    pred = rgb.detach().clamp(0.0, 1.0).float()
    gt = gt_rgb.detach().clamp(0.0, 1.0).float()

    if metric_mask is not None and torch.any(metric_mask):
        mask3 = metric_mask.unsqueeze(0).expand_as(pred)
        bg_img = bg.view(3, 1, 1).expand_as(pred)
        pred = torch.where(mask3, pred, bg_img)
        gt = torch.where(mask3, gt, bg_img)

    return pred.unsqueeze(0), gt.unsqueeze(0)


@torch.no_grad()
def compute_rgb_metrics(
    rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    metric_mask: torch.Tensor,
    bg: torch.Tensor,
    lpips_model,
):
    pred, gt = prepare_metric_images(rgb, gt_rgb, metric_mask, bg)
    return {
        "psnr": float(psnr(pred, gt).mean().item()),
        "ssim": float(ssim(pred, gt).mean().item()),
        "lpips": float(lpips_model(pred, gt).mean().item()),
    }


@torch.no_grad()
def select_projected_gaussians(view, render_pkg, split_mask: torch.Tensor, error_map: torch.Tensor, max_points: int):
    points = render_pkg["deformed_means3D"].detach()
    if points.ndim != 2 or points.shape[0] == 0 or not torch.any(split_mask):
        return torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device), 0

    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)
    clip = points_h @ view.full_proj_transform
    w = clip[:, 3]
    valid_w = torch.abs(w) > 1e-8
    denom = torch.where(valid_w, w, torch.ones_like(w) * 1e-8)
    ndc = clip[:, :3] / denom[:, None]

    width = int(view.image_width)
    height = int(view.image_height)
    px = ((ndc[:, 0] + 1.0) * 0.5 * width).long()
    py = ((ndc[:, 1] + 1.0) * 0.5 * height).long()

    inside = valid_w & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if "visibility_filter" in render_pkg:
        inside = inside & render_pkg["visibility_filter"].detach()

    selected = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    idx = torch.where(inside)[0]
    if idx.numel() == 0:
        return selected, 0

    pix_x = px[idx]
    pix_y = py[idx]
    hit = split_mask[pix_y, pix_x]
    idx = idx[hit]
    if idx.numel() == 0:
        return selected, 0

    if max_points > 0 and idx.numel() > max_points:
        scores = error_map[py[idx], px[idx]]
        _, order = torch.topk(scores, k=max_points, largest=True)
        idx = idx[order]

    selected[idx] = True
    return selected, int(idx.numel())


@torch.no_grad()
def split_selected_gaussians(gaussians: GaussianModel, selected: torch.Tensor, split_count: int, split_scale: float):
    idx = torch.where(selected)[0]
    if idx.numel() == 0 or split_count <= 0:
        return 0

    parent_xyz = gaussians._xyz.detach()[idx]
    parent_scale = gaussians.get_scaling.detach()[idx]
    parent_rot = gaussians._rotation.detach()[idx]
    rots = build_rotation(parent_rot)

    largest_axis = torch.argmax(parent_scale, dim=1)
    axis = torch.zeros_like(parent_xyz)
    axis[torch.arange(idx.numel(), device=idx.device), largest_axis] = 1.0
    world_axis = torch.bmm(rots, axis[..., None]).squeeze(-1)
    world_axis = world_axis / torch.clamp(torch.linalg.norm(world_axis, dim=1, keepdim=True), min=1e-8)

    offsets = torch.linspace(-1.0, 1.0, steps=split_count + 2, device=idx.device, dtype=parent_xyz.dtype)[1:-1]
    step = parent_scale.max(dim=1).values * split_scale
    new_xyz = parent_xyz.repeat_interleave(split_count, dim=0)
    new_xyz = new_xyz + (
        world_axis.repeat_interleave(split_count, dim=0)
        * (step[:, None] * offsets[None, :]).reshape(-1, 1)
    )

    shrink = 0.8 * (split_count + 1)
    new_scaling = gaussians.scaling_inverse_activation(
        parent_scale.repeat_interleave(split_count, dim=0) / shrink
    )
    new_rotation = gaussians._rotation.detach()[idx].repeat_interleave(split_count, dim=0)
    new_features_dc = gaussians._features_dc.detach()[idx].repeat_interleave(split_count, dim=0)
    new_features_rest = gaussians._features_rest.detach()[idx].repeat_interleave(split_count, dim=0)

    parent_opacity = gaussians.get_opacity.detach()[idx].repeat_interleave(split_count, dim=0)
    new_opacity_value = torch.clamp(parent_opacity / (split_count + 1), min=1e-4, max=1.0 - 1e-4)
    new_opacity = inverse_sigmoid(new_opacity_value)

    gaussians.densification_postfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
    )
    return int(new_xyz.shape[0])


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_ablation_name(args) -> str:
    """
    根据消融开关自动生成输出子目录名称。
    所有结果仍然保存在每个序列的 opt 文件夹下面：
    <sequence>/opt/<ablation_name>/
    """
    if getattr(args, "ablation_name", None) not in [None, "", "auto"]:
        return str(args.ablation_name)

    depth_on = bool(getattr(args, "enable_depth_loss", True))
    split_on = bool(getattr(args, "enable_gaussian_split", True))

    if depth_on and split_on:
        return "depth_loss_and_split"
    if depth_on and not split_on:
        return "depth_loss_only"
    if (not depth_on) and split_on:
        return "split_only"
    return "no_depth_no_split"


# ------------------------- 测试集评价函数 -------------------------
@torch.no_grad()
def evaluate_on_test_set(
    seq: str,
    args,
    scene,
    gaussians,
    pipeline,
    device: torch.device,
    lpips_model,
    bg: torch.Tensor,
    smpl_rot,
    seq_opt_dir: Path,
):
    """
    只做测试集渲染和评价，不做任何优化，不使用测试集深度。
    最终返回的 PSNR / SSIM / LPIPS 就是测试集指标。
    """
    test_views_all = resolve_views(scene.getTestCameras(), args.test_split)
    if args.max_test_frames is not None:
        test_views = test_views_all[: args.max_test_frames]
    else:
        test_views = test_views_all

    test_dir = seq_opt_dir / "test_eval"
    test_dir.mkdir(parents=True, exist_ok=True)

    metrics_records = []
    total_render_sec = 0.0
    total_frame_sec = 0.0

    pbar = tqdm(test_views, desc=f"{seq} test-eval", leave=False)
    for frame_idx, view in enumerate(pbar):
        frame_start = time.perf_counter()

        smpl_kwargs = get_smpl_kwargs(smpl_rot, args.test_split, view)

        maybe_sync(device)
        render_start = time.perf_counter()
        render_pkg = render(view, gaussians, pipeline, bg, **smpl_kwargs)
        maybe_sync(device)
        render_elapsed = time.perf_counter() - render_start
        total_render_sec += render_elapsed

        rgb = torch.clamp(render_pkg["render"], 0.0, 1.0)
        depth = as_hw(render_pkg["depth"])
        gt_rgb = view.original_image[:3].to(device=device, dtype=torch.float32)

        if view.bound_mask is not None:
            metric_mask = view.bound_mask[0].to(device=device, dtype=torch.bool)
        else:
            metric_mask = torch.ones_like(as_hw(render_pkg["depth"]), dtype=torch.bool, device=device)

        valid = metric_mask

        metrics = compute_rgb_metrics(rgb, gt_rgb, metric_mask, bg, lpips_model)

        frame_elapsed = time.perf_counter() - frame_start
        total_frame_sec += frame_elapsed

        if args.fps_mode == "frame":
            fps_val = 1.0 / max(frame_elapsed, 1e-8)
        else:
            fps_val = 1.0 / max(render_elapsed, 1e-8)

        metrics_records.append({
            "image_name": view.image_name,
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
            "fps": float(fps_val),
            "render_time_sec": float(render_elapsed),
            "frame_time_sec": float(frame_elapsed),
        })

        if args.save_test_visuals and (frame_idx % args.save_every == 0):
            save_test_visuals(test_dir, view.image_name, rgb, depth, valid)

        pbar.set_postfix({
            "psnr": f"{metrics['psnr']:.2f}",
            "ssim": f"{metrics['ssim']:.4f}",
        })

    if not metrics_records:
        return {
            "split": args.test_split,
            "num_views": 0,
            "mean_psnr": None,
            "mean_ssim": None,
            "mean_lpips": None,
            "mean_fps": None,
            "throughput_fps_render": None,
            "throughput_fps_frame": None,
            "total_render_sec": None,
            "total_frame_sec": None,
            "test_eval_dir": str(test_dir),
        }

    summary = {
        "split": args.test_split,
        "num_views": len(metrics_records),
        "mean_psnr": float(np.mean([x["psnr"] for x in metrics_records])),
        "mean_ssim": float(np.mean([x["ssim"] for x in metrics_records])),
        "mean_lpips": float(np.mean([x["lpips"] for x in metrics_records])),
        "mean_fps": float(np.mean([x["fps"] for x in metrics_records])),
        "throughput_fps_render": float(len(metrics_records) / total_render_sec) if total_render_sec > 0 else None,
        "throughput_fps_frame": float(len(metrics_records) / total_frame_sec) if total_frame_sec > 0 else None,
        "total_render_sec": float(total_render_sec),
        "total_frame_sec": float(total_frame_sec),
        "test_eval_dir": str(test_dir),
    }

    with (test_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ------------------------- 单序列主流程 -------------------------
def process_sequence(seq: str, args, pipeline, device: torch.device, lpips_model):
    data_dir = args.data_root / seq
    model_dir = args.model_root / seq

    # 每个序列输出统一保存在 <sequence>/opt/<ablation_name>
    ablation_name = get_ablation_name(args)
    seq_opt_dir = args.output_root / seq / "opt" / ablation_name
    train_vis_dir = seq_opt_dir / "train_opt"
    mask_dir = args.output_root / seq / "depth_map" / "mask_high_error"

    # 训练阶段仅使用训练集深度
    fs_depth_dir = data_dir / "render_depth" / "train" / "filtered_npy"

    if not data_dir.exists():
        raise FileNotFoundError(f"Missing sequence data dir: {data_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing sequence model dir: {model_dir}")
    if not fs_depth_dir.exists():
        raise FileNotFoundError(f"Missing FoundationStereo depth dir: {fs_depth_dir}")

    seq_opt_dir.mkdir(parents=True, exist_ok=True)
    train_vis_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg_args(model_dir, data_dir, str(device))
    gaussians = GaussianModel(cfg.sh_degree, cfg.smpl_type, cfg.motion_offset_flag, cfg.actor_gender, cfg)
    scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    optimizer = make_gaussian_optimizer(gaussians, args, device)
    smpl_rot = load_smpl_rot(model_dir, scene.loaded_iter) if args.use_cached_smpl else None
    bg = torch.tensor(
        [1, 1, 1] if cfg.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )

    train_views_all = resolve_views(scene.getTrainCameras(), args.train_split)
    if args.max_frames is not None:
        train_views = train_views_all[: args.max_frames]
    else:
        train_views = train_views_all

    seq_summary = {
        "sequence": seq,
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "output_dir": str(seq_opt_dir),
        "iteration_loaded": int(scene.loaded_iter),
        "num_iterations": args.num_iterations,
        "train_num_views": len(train_views),
        "test_split": args.test_split,
        "lambda_depth": args.lambda_depth,
        "epsilon_split": args.epsilon_split,
        "split_count": args.split_count,
        "fps_mode": args.fps_mode,
        "ablation_name": ablation_name,
        "enable_depth_loss": bool(args.enable_depth_loss),
        "enable_gaussian_split": bool(args.enable_gaussian_split),
        "train_iterations": {},
    }

    global_step = 0

    # ------------------ 训练/优化阶段（只用训练集） ------------------
    for opt_iter in range(1, args.num_iterations + 1):
        iter_dir = train_vis_dir / f"iter_{opt_iter:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        iter_records = []
        total_render_sec = 0.0
        total_frame_sec = 0.0

        pbar = tqdm(train_views, desc=f"{seq} train-opt iter {opt_iter}/{args.num_iterations}", leave=False)

        for frame_idx, view in enumerate(pbar):
            frame_start = time.perf_counter()
            global_step += 1

            fs_depth_file = fs_depth_dir / f"{view.image_name}.npy"
            if not fs_depth_file.exists():
                continue

            depth_fs_np = np.load(fs_depth_file).astype(np.float32)

            high_mask_np = load_mask(mask_dir, view.image_name, data_dir, args.epsilon_split)
            if high_mask_np is None:
                continue
            if high_mask_np.shape != depth_fs_np.shape:
                continue

            smpl_kwargs = get_smpl_kwargs(smpl_rot, args.train_split, view)

            maybe_sync(device)
            render_start = time.perf_counter()
            render_pkg = render(view, gaussians, pipeline, bg, **smpl_kwargs)
            maybe_sync(device)
            render_elapsed = time.perf_counter() - render_start
            total_render_sec += render_elapsed

            rgb = torch.clamp(render_pkg["render"], 0.0, 1.0)
            depth = as_hw(render_pkg["depth"])
            alpha = as_hw(render_pkg["render_alpha"])

            fs_depth = torch.from_numpy(depth_fs_np).to(device=device, dtype=torch.float32)
            high_mask = torch.from_numpy(high_mask_np).to(device=device, dtype=torch.bool)

            bound_mask = view.bound_mask[0].to(device=device, dtype=torch.bool) if view.bound_mask is not None else None
            alpha_target = view.bkgd_mask[0].to(device=device, dtype=torch.float32) if view.bkgd_mask is not None else torch.ones_like(alpha)
            gt_rgb = view.original_image[:3].to(device=device, dtype=torch.float32)

            valid = torch.isfinite(fs_depth) & torch.isfinite(depth) & (fs_depth > 0) & (depth > 0)
            if bound_mask is not None:
                valid = valid & bound_mask
                color_mask = bound_mask
            else:
                color_mask = valid

            current_error = torch.zeros_like(depth)
            current_error[valid] = torch.abs(depth[valid] - fs_depth[valid])

            split_mask = valid & high_mask & (current_error > args.epsilon_split)
            depth_mask = split_mask if args.depth_on_high_error_only else valid
            if not torch.any(depth_mask):
                depth_mask = valid

            loss_color = masked_l1(rgb, gt_rgb, color_mask)
            loss_alpha = masked_l2(alpha, alpha_target, color_mask)

            # 消融开关 1：深度 loss
            # enable_depth_loss=True  时：loss 中加入 lambda_depth * L_depth
            # enable_depth_loss=False 时：仍可计算深度误差用于高斯球分裂，但不参与反向传播损失
            if torch.any(depth_mask):
                loss_depth_raw = torch.mean(torch.abs(depth[depth_mask] - fs_depth[depth_mask]))
            else:
                loss_depth_raw = torch.zeros((), device=device)

            if args.enable_depth_loss:
                loss_depth = loss_depth_raw
                loss = loss_color + args.lambda_alpha * loss_alpha + args.lambda_depth * loss_depth
            else:
                loss_depth = torch.zeros((), device=device)
                loss = loss_color + args.lambda_alpha * loss_alpha

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 消融开关 2：深度误差引导的高斯球分裂
            # enable_gaussian_split=True  时：根据 split_mask 选择并分裂高斯球
            # enable_gaussian_split=False 时：完全不新增高斯球
            selected_count = 0
            new_points = 0
            if args.enable_gaussian_split:
                selected, selected_count = select_projected_gaussians(
                    view,
                    render_pkg,
                    split_mask.detach(),
                    current_error.detach(),
                    args.max_split_points_per_frame,
                )

                should_split = (
                    args.split_count > 0
                    and selected_count > 0
                    and (global_step % args.split_every == 0)
                    and gaussians.get_xyz.shape[0] < args.max_points
                )
                if should_split:
                    new_points = split_selected_gaussians(
                        gaussians,
                        selected,
                        args.split_count,
                        args.split_scale,
                    )

            metric_mask = color_mask if color_mask is not None and torch.any(color_mask) else valid
            metrics = compute_rgb_metrics(rgb, gt_rgb, metric_mask, bg, lpips_model)

            if args.save_visuals and (frame_idx % args.save_every == 0):
                save_train_visuals(iter_dir, view.image_name, rgb, depth, fs_depth, valid)

            frame_elapsed = time.perf_counter() - frame_start
            total_frame_sec += frame_elapsed
            fps_val = 1.0 / max(frame_elapsed if args.fps_mode == "frame" else render_elapsed, 1e-8)

            record = {
                "iteration": opt_iter,
                "image_name": view.image_name,
                "loss_total": float(loss.detach().cpu()),
                "loss_color": float(loss_color.detach().cpu()),
                "loss_alpha": float(loss_alpha.detach().cpu()),
                "loss_depth": float(loss_depth.detach().cpu()),
                "loss_depth_raw": float(loss_depth_raw.detach().cpu()),
                "enable_depth_loss": bool(args.enable_depth_loss),
                "enable_gaussian_split": bool(args.enable_gaussian_split),
                "valid_pixels": int(valid.sum().detach().cpu()),
                "high_error_pixels": int(split_mask.sum().detach().cpu()),
                "selected_gaussians": selected_count,
                "new_gaussians": new_points,
                "num_gaussians": int(gaussians.get_xyz.shape[0]),
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "fps": float(fps_val),
                "render_time_sec": float(render_elapsed),
                "frame_time_sec": float(frame_elapsed),
            }
            iter_records.append(record)

            pbar.set_postfix({
                "loss": f"{record['loss_total']:.4f}",
                "depth": f"{record['loss_depth']:.4f}",
                "psnr": f"{record['psnr']:.2f}",
                "new": new_points,
                "pts": record["num_gaussians"],
            })

            if args.empty_cache_every > 0 and global_step % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

        ply_path = train_vis_dir / f"gaussians_iter{opt_iter:03d}.ply"
        gaussians.save_ply(str(ply_path))

        if iter_records:
            seq_summary["train_iterations"][f"iter_{opt_iter:03d}"] = {
                "mean_loss_total": float(np.mean([x["loss_total"] for x in iter_records])),
                "mean_loss_depth": float(np.mean([x["loss_depth"] for x in iter_records])),
                "mean_loss_depth_raw": float(np.mean([x["loss_depth_raw"] for x in iter_records])),
                "enable_depth_loss": bool(args.enable_depth_loss),
                "enable_gaussian_split": bool(args.enable_gaussian_split),
                "mean_psnr": float(np.mean([x["psnr"] for x in iter_records])),
                "mean_ssim": float(np.mean([x["ssim"] for x in iter_records])),
                "mean_lpips": float(np.mean([x["lpips"] for x in iter_records])),
                "mean_fps": float(np.mean([x["fps"] for x in iter_records])),
                "throughput_fps_render": float(len(iter_records) / total_render_sec) if total_render_sec > 0 else None,
                "throughput_fps_frame": float(len(iter_records) / total_frame_sec) if total_frame_sec > 0 else None,
                "total_render_sec": float(total_render_sec),
                "total_frame_sec": float(total_frame_sec),
                "total_new_gaussians": int(sum(x["new_gaussians"] for x in iter_records)),
                "final_num_gaussians": int(gaussians.get_xyz.shape[0]),
                "ply": str(ply_path),
            }

        with (seq_opt_dir / "optimization_summary.json").open("w", encoding="utf-8") as f:
            json.dump(seq_summary, f, indent=2)

    # ------------------ 测试集评价阶段（只渲染测试集，不参与优化） ------------------
    test_metrics = evaluate_on_test_set(
        seq=seq,
        args=args,
        scene=scene,
        gaussians=gaussians,
        pipeline=pipeline,
        device=device,
        lpips_model=lpips_model,
        bg=bg,
        smpl_rot=smpl_rot,
        seq_opt_dir=seq_opt_dir,
    )

    seq_summary["test_metrics"] = test_metrics

    # 最终高斯数量
    seq_summary["final_num_gaussians"] = int(gaussians.get_xyz.shape[0])

    with (seq_opt_dir / "optimization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(seq_summary, f, indent=2)

    torch.cuda.empty_cache()
    return seq_summary


# ------------------------- 汇总函数 -------------------------
def collect_final_sequence_metrics(summary):
    """
    最终 merged summary 只取测试集指标
    """
    test_metrics = summary.get("test_metrics", {})
    train_iterations = summary.get("train_iterations", {})

    final_train_key = None
    final_train_iter = {}
    num_iterations = summary.get("num_iterations", 0)

    for i in range(num_iterations, 0, -1):
        key = f"iter_{i:03d}"
        if key in train_iterations:
            final_train_key = key
            final_train_iter = train_iterations[key]
            break

    return {
        "sequence": summary.get("sequence"),
        "output_dir": summary.get("output_dir"),
        "ablation_name": summary.get("ablation_name"),
        "enable_depth_loss": summary.get("enable_depth_loss"),
        "enable_gaussian_split": summary.get("enable_gaussian_split"),
        "num_iterations": summary.get("num_iterations"),
        "train_num_views": summary.get("train_num_views"),
        "test_num_views": test_metrics.get("num_views"),
        "fps_mode": summary.get("fps_mode"),
        "final_train_iteration": final_train_key,
        "final_num_gaussians": summary.get("final_num_gaussians", final_train_iter.get("final_num_gaussians")),
        "train_ply": final_train_iter.get("ply"),
        "mean_psnr": test_metrics.get("mean_psnr"),
        "mean_ssim": test_metrics.get("mean_ssim"),
        "mean_lpips": test_metrics.get("mean_lpips"),
        "mean_fps": test_metrics.get("mean_fps"),
        "throughput_fps_render": test_metrics.get("throughput_fps_render"),
        "throughput_fps_frame": test_metrics.get("throughput_fps_frame"),
        "total_render_sec": test_metrics.get("total_render_sec"),
        "total_frame_sec": test_metrics.get("total_frame_sec"),
        "test_eval_dir": test_metrics.get("test_eval_dir"),
    }


# ------------------------- 参数 -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth-guided SeqAvatar Gaussian optimization (train-depth optimize + test-set evaluation)"
    )
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model_root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output_root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--iteration", type=int, default=25000)

    # 训练/优化只用训练集
    parser.add_argument("--train_split", type=str, default="train")
    # 测试指标只用测试集
    parser.add_argument("--test_split", type=str, default="novelview")

    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--lambda_depth", type=float, default=0.1)
    parser.add_argument("--lambda_alpha", type=float, default=0.1)

    # ---------------- 消融实验开关 ----------------
    # 两个都开：--enable_depth_loss --enable_gaussian_split
    # 只开深度 loss：--enable_depth_loss --no-enable_gaussian_split
    # 只开高斯球分裂：--no-enable_depth_loss --enable_gaussian_split
    # 两个都关：--no-enable_depth_loss --no-enable_gaussian_split
    parser.add_argument("--enable_depth_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_gaussian_split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ablation_name", type=str, default="auto")

    parser.add_argument("--epsilon_split", type=float, default=0.02)
    parser.add_argument("--split_count", type=int, default=3)
    parser.add_argument("--split_scale", type=float, default=0.5)
    parser.add_argument("--split_every", type=int, default=1)
    parser.add_argument("--max_split_points_per_frame", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=65000)
    parser.add_argument("--empty_cache_every", type=int, default=5)
    parser.add_argument("--depth_on_high_error_only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lr_xyz", type=float, default=1e-4)
    parser.add_argument("--lr_features", type=float, default=1e-3)
    parser.add_argument("--lr_opacity", type=float, default=1e-2)
    parser.add_argument("--lr_scaling", type=float, default=1e-3)
    parser.add_argument("--lr_rotation", type=float, default=5e-4)

    # 训练时最多取多少训练帧
    parser.add_argument("--max_frames", type=int, default=60)
    # 测试时最多取多少测试帧；None 表示全部测试帧
    parser.add_argument("--max_test_frames", type=int, default=None)

    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--save_visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_test_visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cached_smpl", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--fps_mode", choices=["render", "frame"], default="render")
    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument("--compute_cov3D_python", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--run_single_sequence", action="store_true", help="Internal flag: run exactly one sequence in this process.")

    return parser.parse_args()


# ------------------------- 主函数 -------------------------
def main():
    args = parse_args()

    # 多序列：逐序列单独起进程，防止显存残留
    if (not args.run_single_sequence) and len(args.sequences) > 1:
        print("\n[INFO] Multiple sequences detected, running one sequence per process.")
        for seq in args.sequences:
            print(f"\n[LAUNCH] Start sequence: {seq}")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--sequences", seq,
                "--data_root", str(args.data_root),
                "--model_root", str(args.model_root),
                "--output_root", str(args.output_root),
                "--iteration", str(args.iteration),
                "--train_split", str(args.train_split),
                "--test_split", str(args.test_split),
                "--num_iterations", str(args.num_iterations),
                "--lambda_depth", str(args.lambda_depth),
                "--lambda_alpha", str(args.lambda_alpha),
                "--ablation_name", str(args.ablation_name),
                "--epsilon_split", str(args.epsilon_split),
                "--split_count", str(args.split_count),
                "--split_scale", str(args.split_scale),
                "--split_every", str(args.split_every),
                "--max_split_points_per_frame", str(args.max_split_points_per_frame),
                "--max_points", str(args.max_points),
                "--empty_cache_every", str(args.empty_cache_every),
                "--save_every", str(args.save_every),
                "--fps_mode", str(args.fps_mode),
                "--run_single_sequence",
            ]

            if args.max_frames is not None:
                cmd.extend(["--max_frames", str(args.max_frames)])
            if args.max_test_frames is not None:
                cmd.extend(["--max_test_frames", str(args.max_test_frames)])

            if args.enable_depth_loss:
                cmd.append("--enable_depth_loss")
            else:
                cmd.append("--no-enable_depth_loss")

            if args.enable_gaussian_split:
                cmd.append("--enable_gaussian_split")
            else:
                cmd.append("--no-enable_gaussian_split")

            if args.depth_on_high_error_only:
                cmd.append("--depth_on_high_error_only")
            else:
                cmd.append("--no-depth_on_high_error_only")

            if args.save_visuals:
                cmd.append("--save_visuals")
            else:
                cmd.append("--no-save_visuals")

            if args.save_test_visuals:
                cmd.append("--save_test_visuals")
            else:
                cmd.append("--no-save_test_visuals")

            if args.use_cached_smpl:
                cmd.append("--use_cached_smpl")
            else:
                cmd.append("--no-use_cached_smpl")

            if args.compute_cov3D_python:
                cmd.append("--compute_cov3D_python")
            else:
                cmd.append("--no-compute_cov3D_python")

            if args.convert_SHs_python:
                cmd.append("--convert_SHs_python")

            if args.debug:
                cmd.append("--debug")
            if args.quiet:
                cmd.append("--quiet")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[ERROR] Sequence {seq} failed with return code {result.returncode}")
                break
            print(f"[DONE] Sequence {seq} finished.")

        # 合并六个序列的最终测试集指标
        merged = {"sequences": [], "fps_mode": args.fps_mode}
        for seq in args.sequences:
            seq_summary_path = args.output_root / seq / "opt" / get_ablation_name(args) / "optimization_summary.json"
            if not seq_summary_path.exists():
                print(f"[WARNING] Missing sequence summary: {seq_summary_path}")
                continue
            with seq_summary_path.open("r", encoding="utf-8") as f:
                seq_summary = json.load(f)
            merged["sequences"].append(collect_final_sequence_metrics(seq_summary))

        merged_path = args.output_root / f"depth_guided_test_summary_all_{get_ablation_name(args)}.json"
        with merged_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f"[DONE] Merged summary saved to {merged_path}")
        return

    # 单序列模式
    safe_state(args.quiet)
    pipeline = make_pipeline_args(args)
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()

    start = time.time()
    summaries = []

    for seq in args.sequences:
        print(f"\n[SEQ] Optimizing {seq}")
        summary = process_sequence(seq, args, pipeline, device, lpips_model)
        summaries.append(summary)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    elapsed = time.time() - start

    for summary in summaries:
        summary["elapsed_sec"] = elapsed
        seq_summary_path = args.output_root / summary["sequence"] / "opt" / get_ablation_name(args) / "optimization_summary.json"
        seq_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with seq_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[DONE] Per-sequence summary saved to {seq_summary_path}")


if __name__ == "__main__":
    main()