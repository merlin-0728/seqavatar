# depth_render.py
# Copyright (C) 2023, Inria
# GRAPHDECO research group
# Render RGB + Depth for SeqAvatar novelview set.

import torch
import os
import time
from tqdm import tqdm
from os import makedirs
import torchvision
import imageio
import numpy as np
import lpips
from PIL import Image, ImageDraw

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from scene import Scene
from gaussian_renderer import render
from gaussian_renderer import GaussianModel

# LPIPS perceptual loss
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))


def _colorize_depth(depth_vis):
    """Map a uint8 depth image to a simple blue-cyan-yellow-red colormap."""
    x = depth_vis.astype(np.float32) / 255.0
    r = np.clip(1.5 * x - 0.25, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * x - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.25 - 1.5 * x, 0.0, 1.0)
    color = np.stack([r, g, b], axis=-1)
    color[depth_vis == 0] = 0.0
    return (color * 255).astype("uint8")


def _make_preview(preview_items, preview_path):
    if not preview_items:
        return

    cell_w, cell_h = 320, 220
    label_h = 24
    cols = 3
    rows = len(preview_items)
    canvas = Image.new("RGB", (cell_w * cols, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    labels = ("render", "depth", "gt")
    for row, (image_name, render_np, depth_np, gt_np) in enumerate(preview_items):
        y0 = row * (cell_h + label_h)
        for col, (label, arr) in enumerate(zip(labels, (render_np, depth_np, gt_np))):
            img = Image.fromarray(arr).convert("RGB")
            img.thumbnail((cell_w, cell_h), Image.Resampling.BILINEAR)
            x = col * cell_w + (cell_w - img.width) // 2
            y = y0 + label_h + (cell_h - img.height) // 2
            canvas.paste(img, (x, y))
            draw.text((col * cell_w + 8, y0 + 5), f"{image_name} {label}", fill=(0, 0, 0))

    makedirs(os.path.dirname(preview_path), exist_ok=True)
    canvas.save(preview_path)


# ------------------- 渲染单个集合 -------------------
def render_set(model_path, dataset_path, name, iteration, views, gaussians, pipeline, background):
    """Render the novelview split and save RGB + depth."""
    output_path = os.path.join(dataset_path, "depth", name)
    rgb_path = os.path.join(output_path, "rgb")
    depth_raw_path = os.path.join(output_path, "npy")
    depth_vis_path = os.path.join(output_path, "vis")
    gt_path = os.path.join(output_path, "gt")

    makedirs(rgb_path, exist_ok=True)
    makedirs(depth_raw_path, exist_ok=True)
    makedirs(depth_vis_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)

    elapsed_time = 0
    preview_items = []
    psnrs, ssims, lpipss = 0.0, 0.0, 0.0
    rendered_count = 0

    for _, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        gt = view.original_image[0:3, :, :].cuda()
        bound_mask = view.bound_mask

        start_time = time.time()
        # Do not pass cached smpl_rot. The renderer computes the pose deformation
        # directly for novelview cameras.
        render_output = render(view, gaussians, pipeline, background)
        end_time = time.time()
        elapsed_time += end_time - start_time

        rendering = render_output["render"]
        depth_map = render_output["depth"].squeeze()
        alpha = render_output["render_alpha"].squeeze()

        # Apply bound mask
        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        # Mask invalid depth
        valid_depth = torch.isfinite(depth_map) & (alpha > 1e-4) & (bound_mask[0].to(depth_map.device) > 0)
        depth_raw = torch.where(valid_depth, depth_map, torch.zeros_like(depth_map))

        # ------------------- Save RGB -------------------
        torchvision.utils.save_image(rendering, os.path.join(rgb_path, f"{view.image_name}.png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, f"{view.image_name}.png"))

        # ------------------- Save Depth -------------------
        # Raw depth
        depth_np = depth_raw.detach().cpu().numpy().astype(np.float32)
        np.save(os.path.join(depth_raw_path, f"{view.image_name}.npy"), depth_np)

        # Normalized depth for visualization
        depth_vis = torch.zeros_like(depth_map, dtype=torch.float32)
        if valid_depth.any():
            valid_values = depth_map[valid_depth]
            depth_min = valid_values.min()
            depth_max = valid_values.max()
            denom = torch.clamp(depth_max - depth_min, min=1e-8)
            depth_vis[valid_depth] = (valid_values - depth_min) / denom
        depth_vis = (depth_vis * 255).detach().cpu().numpy().astype("uint8")
        depth_vis_color = _colorize_depth(depth_vis)
        imageio.imwrite(os.path.join(depth_vis_path, f"{view.image_name}.png"), depth_vis_color)

        if len(preview_items) < 6:
            render_np = (torch.clamp(rendering, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
            gt_np = (torch.clamp(gt, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
            preview_items.append((view.image_name, render_np, depth_vis_color, gt_np))

        # ------------------- Metrics -------------------
        rendering_for_metric = torch.clamp(rendering, 0.0, 1.0)
        gt_for_metric = torch.clamp(gt, 0.0, 1.0)
        psnrs += psnr(rendering_for_metric, gt_for_metric).mean().double()
        ssims += ssim(rendering_for_metric, gt_for_metric).mean().double()
        lpipss += loss_fn_vgg(rendering_for_metric, gt_for_metric).mean().double()
        rendered_count += 1

    fps = len(views) / elapsed_time if elapsed_time > 0 else 0.0
    print(f"[{name}] Elapsed time: {elapsed_time:.2f}s, FPS: {fps:.2f}")
    _make_preview(preview_items, os.path.join(output_path, "preview.png"))
    print(f"Depth saved to: {output_path}")

    if rendered_count > 0:
        psnrs /= rendered_count
        ssims /= rendered_count
        lpipss /= rendered_count
        print(f"[ITER {iteration}] Evaluating {name} #{rendered_count}: PSNR {psnrs:.4f} SSIM {ssims:.4f} LPIPS {lpipss:.4f}")


# ------------------- 渲染测试集合 -------------------
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type,
                                  dataset.motion_offset_flag, dataset.actor_gender, dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # ------------------- novelview / test 集 -------------------
        if not skip_test:
            for key, views in scene.getTestCameras().items():
                render_set(dataset.model_path, dataset.source_path, key, scene.loaded_iter, views, gaussians, pipeline, background)


# ------------------- main -------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="Depth rendering for SeqAvatar novelview only")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering dataset at:", args.model_path)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
