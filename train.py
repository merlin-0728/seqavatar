#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, l1_loss_masked, l2_loss_masked, ssim, full_aiap_loss, weighted_aiap_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from part_label.common import build_part_neighbor_weight_matrix
import json
import numpy as np
import pickle
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import shutil
from torchvision.ops import masks_to_boxes
import time
torch.backends.cudnn.enabled = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))


@torch.no_grad()
def select_point_guided_anchors(
    viewpoint_camera,
    deformed_means3D,
    visibility_filter,
    image,
    gt_image,
    bound_mask,
    patch_size,
    topk_patches,
    anchor_radius,
    max_anchors,
    min_patch_coverage,
):
    """Select visible canonical anchors near high-error image patches."""
    num_points = int(deformed_means3D.shape[0])
    guided_mask = torch.zeros((num_points,), dtype=torch.bool, device=deformed_means3D.device)
    empty_stats = {
        "patches": 0,
        "anchors": 0,
        "error_mean": 0.0,
        "coverage_mean": 0.0,
    }

    if num_points == 0:
        return guided_mask, empty_stats

    fg_mask = bound_mask
    if fg_mask.ndim == 3:
        fg_mask = fg_mask[0]
    fg_mask = fg_mask.to(device=image.device, dtype=torch.float32)
    error_map = torch.abs(image.detach() - gt_image.detach()).mean(dim=0)
    patch_size = max(1, int(patch_size))
    pooled_error = F.avg_pool2d(
        (error_map * fg_mask)[None, None],
        kernel_size=patch_size,
        stride=patch_size,
        ceil_mode=True,
    )[0, 0]
    pooled_coverage = F.avg_pool2d(
        fg_mask[None, None],
        kernel_size=patch_size,
        stride=patch_size,
        ceil_mode=True,
    )[0, 0]
    valid_patch = pooled_coverage >= float(min_patch_coverage)
    if not bool(valid_patch.any()):
        return guided_mask, empty_stats

    patch_scores = pooled_error / pooled_coverage.clamp_min(1e-6)
    patch_scores = torch.where(valid_patch, patch_scores, torch.full_like(patch_scores, -1.0))
    valid_count = int(valid_patch.sum().item())
    patch_count = min(max(1, int(topk_patches)), valid_count)
    top_scores, top_indices = torch.topk(patch_scores.flatten(), k=patch_count)
    del top_scores
    patch_cols = int(patch_scores.shape[1])
    patch_rows = int(patch_scores.shape[0])
    patch_y = torch.div(top_indices, patch_cols, rounding_mode="floor")
    patch_x = top_indices.remainder(patch_cols)
    height, width = int(image.shape[-2]), int(image.shape[-1])
    centers = torch.stack(
        [
            ((patch_x.float() + 0.5) * patch_size).clamp(0, width - 1),
            ((patch_y.float() + 0.5) * patch_size).clamp(0, height - 1),
        ],
        dim=-1,
    )

    ones = torch.ones(
        (num_points, 1),
        dtype=deformed_means3D.dtype,
        device=deformed_means3D.device,
    )
    means_h = torch.cat([deformed_means3D, ones], dim=-1)
    clip = means_h @ viewpoint_camera.full_proj_transform.to(deformed_means3D.device)
    w = clip[:, 3]
    safe_w = torch.where(w.abs() > 1e-8, w, torch.full_like(w, 1e-8))
    ndc = clip[:, :3] / safe_w[:, None]
    projected_xy = torch.stack(
        [
            (ndc[:, 0] + 1.0) * 0.5 * max(width - 1, 1),
            (ndc[:, 1] + 1.0) * 0.5 * max(height - 1, 1),
        ],
        dim=-1,
    )
    visible = visibility_filter.to(device=deformed_means3D.device, dtype=torch.bool)
    visible = visible & (w > 1e-6)
    visible = visible & (ndc[:, 0].abs() <= 1.0) & (ndc[:, 1].abs() <= 1.0)
    if not bool(visible.any()):
        return guided_mask, {
            "patches": patch_count,
            "anchors": 0,
            "error_mean": float(patch_scores.flatten()[top_indices].mean().item()),
            "coverage_mean": float(pooled_coverage.flatten()[top_indices].mean().item()),
        }

    point_xy = projected_xy[visible]
    visible_indices = torch.where(visible)[0]
    distances = torch.sum(
        (centers[:, None, :] - point_xy[None, :, :]) ** 2,
        dim=-1,
    )
    radius_sq = float(anchor_radius) ** 2
    near = distances <= radius_sq
    if bool(near.any()):
        nearest_distance = torch.where(
            near,
            distances,
            torch.full_like(distances, float("inf")),
        ).min(dim=0).values
        candidate = torch.isfinite(nearest_distance)
        candidate_indices = torch.where(candidate)[0]
        if candidate_indices.numel() > int(max_anchors):
            keep = torch.topk(
                nearest_distance[candidate_indices].neg(),
                k=int(max_anchors),
            ).indices
            candidate_indices = candidate_indices[keep]
    else:
        nearest_distance = distances.min(dim=0).values
        keep_count = min(int(max_anchors), int(nearest_distance.numel()))
        candidate_indices = torch.topk(nearest_distance.neg(), k=keep_count).indices

    guided_mask[visible_indices[candidate_indices]] = True
    return guided_mask, {
        "patches": patch_count,
        "anchors": int(guided_mask.sum().item()),
        "error_mean": float(patch_scores.flatten()[top_indices].mean().item()),
        "coverage_mean": float(pooled_coverage.flatten()[top_indices].mean().item()),
    }


@torch.no_grad()
def get_boundary_band(mask, kernel_size=5):
    mask = mask.float()[None, None]
    pad = 1
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1).squeeze(0).squeeze(0)


@torch.no_grad()
def select_point_anchor_tb_anchors(
    viewpoint_camera,
    deformed_means3D,
    visibility_filter,
    image,
    gt_image,
    bound_mask,
    patch_size,
    topk_patches,
    anchor_radius,
    max_anchors,
    min_patch_coverage,
    history_score=None,
    temporal_alpha=0.5,
    history_momentum=0.8,
    boundary_beta=0.5,
    boundary_kernel=5,
    edge_beta=0.0,
    edge_kernel=3,
    nonrigid_beta=0.0,
    nonrigid_norm=None,
    min_persistence=0.0,
    ):
    fg_mask = bound_mask
    if fg_mask.ndim == 3:
        fg_mask = fg_mask[0]
    fg_mask = fg_mask.to(device=image.device, dtype=torch.float32)
    error_map = torch.abs(image.detach() - gt_image.detach()).mean(dim=0) * fg_mask
    boundary_map = get_boundary_band(fg_mask, boundary_kernel)
    edge_map = image_edge_prior(gt_image, fg_mask, edge_kernel) if float(edge_beta) > 0.0 else None

    patch_size = max(1, int(patch_size))
    err_patch = F.avg_pool2d(error_map[None, None], kernel_size=patch_size, stride=patch_size, ceil_mode=True)[0, 0]
    cov_patch = F.avg_pool2d(fg_mask[None, None], kernel_size=patch_size, stride=patch_size, ceil_mode=True)[0, 0]
    boundary_patch = F.avg_pool2d(boundary_map[None, None], kernel_size=patch_size, stride=patch_size, ceil_mode=True)[0, 0]
    edge_patch = (
        F.avg_pool2d(edge_map[None, None], kernel_size=patch_size, stride=patch_size, ceil_mode=True)[0, 0]
        if edge_map is not None
        else torch.zeros_like(err_patch)
    )

    if history_score is None or history_score.shape != err_patch.shape:
        history_score = err_patch.detach()
    else:
        history_score = history_score.to(device=err_patch.device, dtype=err_patch.dtype)

    history_score = history_score.clamp(0.0, 1.0)
    score = err_patch.clamp(min=0.0) * (1.0 + float(temporal_alpha) * history_score)
    score = score * (1.0 + float(boundary_beta) * boundary_patch)
    if float(edge_beta) > 0.0:
        score = score * (1.0 + float(edge_beta) * edge_patch)
    valid = cov_patch >= float(min_patch_coverage)
    score = score * valid.to(dtype=score.dtype)
    score = score.masked_fill(~valid, -1.0)
    k = min(max(1, int(topk_patches)), int(valid.sum().item()))
    if k <= 0:
        new_history = history_score * float(history_momentum) + err_patch * (1.0 - float(history_momentum))
        return None, new_history, {
            "patches": 0,
            "anchors": 0,
            "error_mean": 0.0,
            "boundary_mean": 0.0,
            "edge_mean": 0.0,
            "nonrigid_mean": 0.0,
            "coverage_mean": 0.0,
            "history_mean": float(new_history.mean().item()),
        }

    vals, inds = torch.topk(score.flatten(), k=k)
    keep = vals > float(min_persistence)
    inds = inds[keep]
    if inds.numel() == 0:
        new_history = history_score * float(history_momentum) + err_patch * (1.0 - float(history_momentum))
        return None, new_history, {
            "patches": 0,
            "anchors": 0,
            "error_mean": 0.0,
            "boundary_mean": 0.0,
            "edge_mean": 0.0,
            "nonrigid_mean": 0.0,
            "coverage_mean": 0.0,
            "history_mean": float(new_history.mean().item()),
        }

    patch_w = score.shape[1]
    ys = inds // patch_w
    xs = inds % patch_w
    centers = torch.stack(
        [
            (xs.float() + 0.5) * patch_size,
            (ys.float() + 0.5) * patch_size,
        ],
        dim=-1,
    )

    ones = torch.ones((deformed_means3D.shape[0], 1), dtype=deformed_means3D.dtype, device=deformed_means3D.device)
    means_h = torch.cat([deformed_means3D, ones], dim=-1)
    clip = means_h @ viewpoint_camera.full_proj_transform.to(deformed_means3D.device)
    w = clip[:, 3]
    safe_w = torch.where(w.abs() > 1e-8, w, torch.full_like(w, 1e-8))
    ndc = clip[:, :3] / safe_w[:, None]
    height, width = int(image.shape[-2]), int(image.shape[-1])
    projected_xy = torch.stack(
        [
            (ndc[:, 0] + 1.0) * 0.5 * max(width - 1, 1),
            (ndc[:, 1] + 1.0) * 0.5 * max(height - 1, 1),
        ],
        dim=-1,
    )
    visible = visibility_filter.to(device=deformed_means3D.device, dtype=torch.bool)
    visible = visible & (w > 1e-6)
    visible = visible & (ndc[:, 0].abs() <= 1.0) & (ndc[:, 1].abs() <= 1.0)
    if not bool(visible.any()):
        new_history = history_score * float(history_momentum) + err_patch * (1.0 - float(history_momentum))
        return None, new_history, {
            "patches": int(inds.numel()),
            "anchors": 0,
            "error_mean": float(err_patch.flatten()[inds].mean().item()),
            "boundary_mean": float(boundary_patch.flatten()[inds].mean().item()),
            "edge_mean": float(edge_patch.flatten()[inds].mean().item()),
            "coverage_mean": float(cov_patch.flatten()[inds].mean().item()),
            "history_mean": float(new_history.mean().item()),
        }

    point_xy = projected_xy[visible]
    visible_indices = torch.where(visible)[0]
    distances = torch.sum((centers[:, None, :] - point_xy[None, :, :]) ** 2, dim=-1)
    radius_sq = float(anchor_radius) ** 2
    selected_patch_scores = score.flatten()[inds]
    nearest_patch = distances.min(dim=0).indices
    nearest_patch_score = selected_patch_scores[nearest_patch].clamp(min=0.0)
    nonrigid_point = torch.zeros_like(nearest_patch_score)
    if nonrigid_norm is not None and float(nonrigid_beta) > 0.0:
        nonrigid_norm = nonrigid_norm.to(device=deformed_means3D.device, dtype=nearest_patch_score.dtype).flatten()
        if nonrigid_norm.numel() >= deformed_means3D.shape[0] and visible_indices.numel() > 0:
            nonrigid_point = nonrigid_norm[visible_indices].clamp(min=0.0)
            if nonrigid_point.numel() > 0:
                nonrigid_point = nonrigid_point / nonrigid_point.max().clamp_min(1e-6)
    anchor_score = nearest_patch_score
    if float(nonrigid_beta) > 0.0:
        anchor_score = anchor_score * (1.0 + float(nonrigid_beta) * nonrigid_point)
    near = distances <= radius_sq
    if bool(near.any()):
        nearest_distance = torch.where(near, distances, torch.full_like(distances, float("inf"))).min(dim=0).values
        candidate = torch.isfinite(nearest_distance)
        candidate_indices = torch.where(candidate)[0]
        if candidate_indices.numel() > int(max_anchors):
            keep_ids = torch.topk(anchor_score[candidate_indices], k=int(max_anchors)).indices
            candidate_indices = candidate_indices[keep_ids]
    else:
        keep_count = min(int(max_anchors), int(anchor_score.numel()))
        candidate_indices = torch.topk(anchor_score, k=keep_count).indices

    guided_mask = torch.zeros((deformed_means3D.shape[0],), dtype=torch.bool, device=deformed_means3D.device)
    guided_mask[visible_indices[candidate_indices]] = True
    new_history = history_score * float(history_momentum) + err_patch * (1.0 - float(history_momentum))
    stats = {
        "patches": int(inds.numel()),
        "anchors": int(guided_mask.sum().item()),
        "error_mean": float(err_patch.flatten()[inds].mean().item()),
        "boundary_mean": float(boundary_patch.flatten()[inds].mean().item()),
        "edge_mean": float(edge_patch.flatten()[inds].mean().item()),
        "nonrigid_mean": float(nonrigid_point[candidate_indices].mean().item()) if float(nonrigid_beta) > 0.0 and candidate_indices.numel() > 0 else 0.0,
        "coverage_mean": float(cov_patch.flatten()[inds].mean().item()),
        "history_mean": float(new_history.mean().item()),
    }
    return guided_mask, new_history, stats


def image_gradient_prior(gt_image, fg_mask):
    gray = gt_image.detach().mean(dim=0, keepdim=True).unsqueeze(0)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=gt_image.dtype,
        device=gt_image.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)[0, 0] * fg_mask
    return grad / grad.max().clamp_min(1e-6)


def image_edge_prior(gt_image, fg_mask, kernel_size=3):
    image = gt_image[:3].detach().unsqueeze(0).to(torch.float32)
    fg_mask = fg_mask.to(device=image.device, dtype=torch.float32)
    rgb_255 = torch.round(image * 255.0)
    grayscale_uint8 = torch.round(
        (
            299.0 * rgb_255[:, 0:1]
            + 587.0 * rgb_255[:, 1:2]
            + 114.0 * rgb_255[:, 2:3]
        ) / 1000.0
    )
    edge_kernel = torch.tensor(
        [[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]],
        dtype=torch.float32,
        device=image.device,
    ).unsqueeze(0)
    pad = max(0, int(kernel_size) // 2)
    edge_response = F.conv2d(grayscale_uint8, edge_kernel, padding=pad)
    edge_map = torch.clamp(edge_response, min=0.0, max=255.0) / 255.0
    edge_map = edge_map[0, 0] * fg_mask
    return edge_map / edge_map.max().clamp_min(1e-6)


@torch.no_grad()
def select_point_cloth_budget_anchors(
    viewpoint_camera,
    deformed_means3D,
    visibility_filter,
    image,
    gt_image,
    bound_mask,
    patch_size,
    topk_patches,
    anchor_radius,
    max_anchors,
    min_patch_coverage,
    history_score=None,
    temporal_alpha=0.5,
    history_momentum=0.8,
    boundary_beta=1.0,
    hf_beta=0.0,
    nonrigid_beta=0.0,
    boundary_kernel=5,
    protect_thresh=0.6,
    min_persistence=0.0,
    nonrigid_norm=None,
):
    fg_mask = bound_mask
    if fg_mask.ndim == 3:
        fg_mask = fg_mask[0]
    fg_mask = fg_mask.to(device=image.device, dtype=torch.float32)
    error_map = torch.abs(image.detach() - gt_image.detach()).mean(dim=0) * fg_mask
    boundary_map = get_boundary_band(fg_mask, boundary_kernel)
    highfreq_map = image_gradient_prior(gt_image, fg_mask)

    patch_size = max(1, int(patch_size))
    err_patch = F.avg_pool2d(error_map[None, None], kernel_size=patch_size, stride=patch_size, ceil_mode=True)[0, 0]
    cov_patch = F.avg_pool2d(fg_mask[None, None], kernel_size=patch_size, stride=patch_size, ceil_mode=True)[0, 0]
    boundary_patch = F.avg_pool2d(boundary_map[None, None], kernel_size=patch_size, stride=patch_size, ceil_mode=True)[0, 0]
    highfreq_patch = F.avg_pool2d(highfreq_map[None, None], kernel_size=patch_size, stride=patch_size, ceil_mode=True)[0, 0]

    if history_score is None or history_score.shape != err_patch.shape:
        history_score = err_patch.detach()
    else:
        history_score = history_score.to(device=err_patch.device, dtype=err_patch.dtype)
    history_score = history_score.clamp(0.0, 1.0)

    score = err_patch.clamp(min=0.0)
    score = score * (1.0 + float(temporal_alpha) * history_score)
    score = score * (1.0 + float(boundary_beta) * boundary_patch)
    if float(hf_beta) > 0.0:
        score = score * (1.0 + float(hf_beta) * highfreq_patch)

    valid = cov_patch >= float(min_patch_coverage)
    score = score * valid.to(dtype=score.dtype)
    score = score.masked_fill(~valid, -1.0)
    k = min(max(1, int(topk_patches)), int(valid.sum().item()))
    new_history = history_score * float(history_momentum) + err_patch * (1.0 - float(history_momentum))

    empty_stats = {
        "patches": 0,
        "anchors": 0,
        "error_mean": 0.0,
        "boundary_mean": 0.0,
        "highfreq_mean": 0.0,
        "nonrigid_mean": 0.0,
        "coverage_mean": 0.0,
        "history_mean": float(new_history.mean().item()),
        "protect_ratio": 0.0,
        "anchor_score_mean": 0.0,
    }
    if k <= 0:
        return None, None, new_history, empty_stats

    vals, inds = torch.topk(score.flatten(), k=k)
    keep = vals > float(min_persistence)
    inds = inds[keep]
    if inds.numel() == 0:
        return None, None, new_history, empty_stats

    patch_w = score.shape[1]
    ys = inds // patch_w
    xs = inds % patch_w
    centers = torch.stack(
        [
            (xs.float() + 0.5) * patch_size,
            (ys.float() + 0.5) * patch_size,
        ],
        dim=-1,
    )

    ones = torch.ones((deformed_means3D.shape[0], 1), dtype=deformed_means3D.dtype, device=deformed_means3D.device)
    means_h = torch.cat([deformed_means3D, ones], dim=-1)
    clip = means_h @ viewpoint_camera.full_proj_transform.to(deformed_means3D.device)
    w = clip[:, 3]
    safe_w = torch.where(w.abs() > 1e-8, w, torch.full_like(w, 1e-8))
    ndc = clip[:, :3] / safe_w[:, None]
    height, width = int(image.shape[-2]), int(image.shape[-1])
    projected_xy = torch.stack(
        [
            (ndc[:, 0] + 1.0) * 0.5 * max(width - 1, 1),
            (ndc[:, 1] + 1.0) * 0.5 * max(height - 1, 1),
        ],
        dim=-1,
    )
    visible = visibility_filter.to(device=deformed_means3D.device, dtype=torch.bool)
    visible = visible & (w > 1e-6)
    visible = visible & (ndc[:, 0].abs() <= 1.0) & (ndc[:, 1].abs() <= 1.0)
    if not bool(visible.any()):
        stats = dict(empty_stats)
        stats.update(
            {
                "patches": int(inds.numel()),
                "error_mean": float(err_patch.flatten()[inds].mean().item()),
                "boundary_mean": float(boundary_patch.flatten()[inds].mean().item()),
                "highfreq_mean": float(highfreq_patch.flatten()[inds].mean().item()),
                "coverage_mean": float(cov_patch.flatten()[inds].mean().item()),
            }
        )
        return None, None, new_history, stats

    point_xy = projected_xy[visible]
    visible_indices = torch.where(visible)[0]
    distances = torch.sum((centers[:, None, :] - point_xy[None, :, :]) ** 2, dim=-1)
    radius_sq = float(anchor_radius) ** 2
    nearest_distance, nearest_patch = distances.min(dim=0)
    near = distances <= radius_sq
    if bool(near.any()):
        candidate = near.any(dim=0)
        candidate_indices = torch.where(candidate)[0]
    else:
        keep_count = min(int(max_anchors), int(nearest_distance.numel()))
        candidate_indices = torch.topk(nearest_distance.neg(), k=keep_count).indices

    selected_patch_scores = score.flatten()[inds]
    nearest_patch_score = selected_patch_scores[nearest_patch].clamp(min=0.0)

    boundary_point = torch.zeros_like(nearest_patch_score)
    highfreq_point = torch.zeros_like(nearest_patch_score)
    px = projected_xy[visible, 0].round().long().clamp(0, max(width - 1, 0))
    py = projected_xy[visible, 1].round().long().clamp(0, max(height - 1, 0))
    if px.numel() > 0:
        boundary_point = boundary_map[py, px].to(dtype=nearest_patch_score.dtype)
        highfreq_point = highfreq_map[py, px].to(dtype=nearest_patch_score.dtype)

    nonrigid_point = torch.zeros_like(nearest_patch_score)
    if nonrigid_norm is not None:
        nonrigid_norm = nonrigid_norm.to(device=deformed_means3D.device, dtype=nearest_patch_score.dtype).flatten()
        if nonrigid_norm.numel() >= deformed_means3D.shape[0]:
            nonrigid_point = nonrigid_norm[visible_indices].clamp(min=0.0)
            nonrigid_point = nonrigid_point / nonrigid_point.max().clamp_min(1e-6)

    anchor_score = nearest_patch_score
    if float(nonrigid_beta) > 0.0:
        anchor_score = anchor_score * (1.0 + float(nonrigid_beta) * nonrigid_point)

    if candidate_indices.numel() > int(max_anchors):
        keep_ids = torch.topk(anchor_score[candidate_indices], k=int(max_anchors)).indices
        candidate_indices = candidate_indices[keep_ids]

    guided_mask = torch.zeros((deformed_means3D.shape[0],), dtype=torch.bool, device=deformed_means3D.device)
    guided_mask[visible_indices[candidate_indices]] = True

    protect_components = [boundary_point]
    if float(hf_beta) > 0.0:
        protect_components.append(highfreq_point)
    if float(nonrigid_beta) > 0.0:
        protect_components.append(nonrigid_point)
    protect_score = torch.stack(protect_components, dim=0).max(dim=0).values
    protect_visible = protect_score >= float(protect_thresh)
    protect_mask = torch.zeros_like(guided_mask)
    protect_mask[visible_indices[protect_visible]] = True
    protect_mask = protect_mask | guided_mask

    stats = {
        "patches": int(inds.numel()),
        "anchors": int(guided_mask.sum().item()),
        "error_mean": float(err_patch.flatten()[inds].mean().item()),
        "boundary_mean": float(boundary_patch.flatten()[inds].mean().item()),
        "highfreq_mean": float(highfreq_patch.flatten()[inds].mean().item()),
        "nonrigid_mean": float(nonrigid_point[candidate_indices].mean().item()) if candidate_indices.numel() > 0 else 0.0,
        "coverage_mean": float(cov_patch.flatten()[inds].mean().item()),
        "history_mean": float(new_history.mean().item()),
        "protect_ratio": float(protect_mask.float().mean().item()),
        "anchor_score_mean": float(anchor_score[candidate_indices].mean().item()) if candidate_indices.numel() > 0 else 0.0,
    }
    return guided_mask, protect_mask, new_history, stats


def get_point_update_target_points(dataset):
    target_points = int(getattr(dataset, "point_tb_target_points", 0))
    if target_points > 0:
        return target_points
    sequence = os.path.basename(os.path.normpath(getattr(dataset, "source_path", "")))
    original_targets = {
        "0044_11": 61950,
        "0051_09": 50783,
        "0206_04": 42664,
        "0813_05": 39757,
        "0007_04": 27168,
        "0019_10": 34698,
    }
    return int(original_targets.get(sequence, 0))


@torch.no_grad()
def select_point_depth_anchors(
    viewpoint_camera,
    canonical_xyz,
    canonical_vertices,
    deformed_means3D,
    visibility_filter,
    image,
    gt_image,
    bound_mask,
    render_depth,
    render_alpha,
    patch_size,
    topk_patches,
    anchor_radius,
    max_anchors,
    min_patch_coverage,
    depth_threshold,
    depth_relative,
    surface_threshold,
):
    """Select high-error anchors consistent with rendered depth and SMPL surface."""
    guided_mask, stats = select_point_guided_anchors(
        viewpoint_camera=viewpoint_camera,
        deformed_means3D=deformed_means3D,
        visibility_filter=visibility_filter,
        image=image,
        gt_image=gt_image,
        bound_mask=bound_mask,
        patch_size=patch_size,
        topk_patches=topk_patches,
        anchor_radius=anchor_radius,
        max_anchors=max_anchors,
        min_patch_coverage=min_patch_coverage,
    )
    stats.update(
        {
            "depth_matches": 0,
            "surface_matches": 0,
            "depth_error_mean": 0.0,
            "surface_distance_mean": 0.0,
        }
    )
    anchor_ids = torch.where(guided_mask)[0]
    if anchor_ids.numel() == 0:
        return guided_mask, None, stats

    height, width = int(image.shape[-2]), int(image.shape[-1])
    points = deformed_means3D[anchor_ids]
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=-1)
    clip = points_h @ viewpoint_camera.full_proj_transform.to(points.device)
    safe_w = torch.where(clip[:, 3].abs() > 1e-8, clip[:, 3], torch.full_like(clip[:, 3], 1e-8))
    ndc = clip[:, :3] / safe_w[:, None]
    grid = torch.stack([ndc[:, 0], ndc[:, 1]], dim=-1).view(1, 1, -1, 2)

    depth_map = render_depth.detach()
    alpha_map = render_alpha.detach()
    if depth_map.ndim == 2:
        depth_map = depth_map.unsqueeze(0)
    if alpha_map.ndim == 2:
        alpha_map = alpha_map.unsqueeze(0)
    sampled_depth = F.grid_sample(
        depth_map.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)
    sampled_alpha = F.grid_sample(
        alpha_map.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)

    # The rasterizer stores alpha-weighted view-space z. Normalize it before
    # comparing against each Gaussian's own view-space z.
    surface_depth = sampled_depth / sampled_alpha.clamp_min(1e-6)
    view = points_h @ viewpoint_camera.world_view_transform.to(points.device)
    point_depth = view[:, 2]
    depth_error = torch.abs(point_depth - surface_depth)
    depth_limit = float(depth_threshold) + float(depth_relative) * surface_depth.abs()
    depth_valid = (
        (sampled_alpha > 0.5)
        & torch.isfinite(surface_depth)
        & torch.isfinite(point_depth)
        & (surface_depth > 0)
        & (point_depth > 0)
        & (depth_error <= depth_limit)
    )
    stats["depth_matches"] = int(depth_valid.sum().item())
    if bool(depth_valid.any()):
        stats["depth_error_mean"] = float(depth_error[depth_valid].mean().item())

    depth_anchor_ids = anchor_ids[depth_valid]
    if depth_anchor_ids.numel() == 0:
        return torch.zeros_like(guided_mask), None, stats

    surface_query = canonical_xyz[depth_anchor_ids].unsqueeze(0)
    surface_distances = torch.cdist(surface_query, canonical_vertices).squeeze(0)
    nearest_dist, nearest_ids = surface_distances.min(dim=-1)
    surface_valid = nearest_dist <= float(surface_threshold)
    stats["surface_matches"] = int(surface_valid.sum().item())
    if bool(surface_valid.any()):
        stats["surface_distance_mean"] = float(nearest_dist[surface_valid].mean().item())

    selected_ids = depth_anchor_ids[surface_valid]
    selected_mask = torch.zeros_like(guided_mask)
    selected_mask[selected_ids] = True
    if selected_ids.numel() == 0:
        return selected_mask, None, stats

    surface_xyz = canonical_vertices[0, nearest_ids[surface_valid]].detach()
    stats["anchors"] = int(selected_ids.numel())
    return selected_mask, surface_xyz, stats


def compute_part_moe_alpha(iteration, dataset):
    if not getattr(dataset, "use_part_moe", False):
        return 0.0
    start_iter = int(getattr(dataset, "part_moe_start_iter", 15000))
    if iteration <= start_iter:
        return 0.0
    global_keep = float(getattr(dataset, "part_moe_global_keep", 0.1))
    max_part_weight = max(0.0, min(1.0, 1.0 - global_keep))
    warmup = int(getattr(dataset, "part_moe_warmup", 1000))
    if warmup <= 0:
        return max_part_weight
    t = float(iteration - start_iter) / float(warmup)
    t = max(0.0, min(1.0, t))
    return t * max_part_weight


def compute_tri_gate_alpha_scale(iteration, dataset):
    if not getattr(dataset, "use_tri_gate", False):
        return 1.0
    start_iter = int(getattr(dataset, "tri_gate_start_iter", 3000))
    if iteration <= start_iter:
        return 0.0
    warmup = int(getattr(dataset, "tri_gate_warmup", 3000))
    if warmup <= 0:
        return 1.0
    t = float(iteration - start_iter) / float(warmup)
    return max(0.0, min(1.0, t))


def compute_part_budget_alpha_scale(iteration, dataset):
    if not getattr(dataset, "use_part_budget", False):
        return 0.0
    start_iter = int(
        getattr(
            dataset,
            "part_budget_start_iter",
            int(getattr(dataset, "part_moe_start_iter", 15000)) + 1000,
        )
    )
    if iteration <= start_iter:
        return 0.0
    warmup = int(getattr(dataset, "part_budget_warmup", 1000))
    if warmup <= 0:
        return 1.0
    t = float(iteration - start_iter) / float(warmup)
    return max(0.0, min(1.0, t))


def compute_tri_token_alpha_scale(iteration, dataset):
    if not getattr(dataset, "use_tri_token", False):
        return 0.0
    start_iter = int(getattr(dataset, "token_tri_start_iter", getattr(dataset, "part_moe_start_iter", 10000)))
    if iteration <= start_iter:
        return 0.0
    warmup = int(getattr(dataset, "token_tri_warmup", 1000))
    if warmup <= 0:
        return 1.0
    t = float(iteration - start_iter) / float(warmup)
    return max(0.0, min(1.0, t))


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
    scene = Scene(dataset, gaussians)
    gaussians.configure_mapo_training()
    gaussians.training_setup(opt)
    part_controller = None
    dynomo_controller = None
    vggt_garment_controller = None
    dynomo_part_weight_matrix = None
    part_label_start_iter = int(getattr(dataset, "part_moe_start_iter", 15000))
    if getattr(dataset, "use_part_moe", False) and opt.densify_until_iter > part_label_start_iter:
        raise ValueError(
            "Part labels are tied to Gaussian order/count. "
            "Set --densify_until_iter <= the part-label start iteration for this ablation."
        )
    if getattr(dataset, "use_part_moe", False):
        from ablations.part_moe_controller import build_part_moe_controller
        part_controller = build_part_moe_controller(dataset)
    if getattr(dataset, "use_dynomo_c", False):
        from ablations.dynomo_controller import build_dynomo_controller
        dynomo_controller = build_dynomo_controller(dataset)
        dynomo_part_weight_matrix = torch.tensor(
            build_part_neighbor_weight_matrix(
                schema=str(getattr(dataset, "part_label_schema", "anatomy5")),
                same_weight=float(getattr(opt, "dynomo_c_part_same_w", 1.0)),
                adjacent_weight=float(getattr(opt, "dynomo_c_part_adj_w", 0.1)),
                other_weight=float(getattr(opt, "dynomo_c_part_other_w", 0.0)),
            ),
            device=gaussians.get_xyz.device,
            dtype=gaussians.get_xyz.dtype,
        )
    if getattr(dataset, "use_vggt_garment", False):
        from ablations.vggt_garment_controller import build_vggt_garment_controller
        vggt_garment_controller = build_vggt_garment_controller(dataset)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log, Ll1_loss_for_log, mask_loss_for_log, ssim_loss_for_log, lpips_loss_for_log = 0.0, 0.0, 0.0, 0.0, 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training")
    first_iter += 1
    point_guided_mask = None
    point_stats = None
    point_tb_history = None
    point_cloth_history = None

    elapsed_time = 0
    for iteration in range(first_iter, opt.iterations + 1):  
        gaussians.update_mapo_partition(iteration)
        if iteration == opt.iterations and vggt_garment_controller is not None:
            vggt_garment_controller.finalize_budget(scene, gaussians)
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, iteration=iteration)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.part_moe_alpha = compute_part_moe_alpha(iteration, dataset)
        gaussians.tri_gate_alpha_scale = compute_tri_gate_alpha_scale(iteration, dataset)
        gaussians.part_budget_alpha_scale = compute_part_budget_alpha_scale(iteration, dataset)
        gaussians.tri_token_alpha_scale = compute_tri_token_alpha_scale(iteration, dataset)
        if (
            getattr(dataset, "use_tri_gate", False)
            and iteration == int(getattr(dataset, "tri_gate_start_iter", 3000)) + 1
        ):
            print(
                f"[TRI_GATE Status] active=True "
                f"alpha_scale={gaussians.tri_gate_alpha_scale:.6f} "
                f"alpha_max={getattr(dataset, 'tri_gate_alpha', 0.2)} "
                f"warmup={getattr(dataset, 'tri_gate_warmup', 3000)}"
            )
        if (
            getattr(dataset, "use_part_moe", False)
            and gaussians.part_label_enabled
            and (
                gaussians.non_rigid_deformer.part_moe_active
                or gaussians.non_rigid_deformer.temporal_conditioned_part_active
            )
            and iteration == int(getattr(dataset, "part_moe_start_iter", 15000)) + 1
        ):
            print(
                f"[PartMoE Status] active=True part_enabled=True "
                f"alpha={gaussians.part_moe_alpha:.6f} "
                f"global_keep={gaussians.part_moe_global_keep}"
            )
        if dynomo_controller is not None:
            dynomo_controller.refresh(scene, gaussians, iteration)
        if (
            getattr(dataset, "use_part_budget", False)
            and gaussians.part_label_enabled
            and (
                gaussians.non_rigid_deformer.part_moe_active
                or gaussians.non_rigid_deformer.temporal_conditioned_part_active
            )
            and iteration == int(getattr(dataset, "part_budget_start_iter", 16000)) + 1
        ):
            print(
                f"[PartBudget Status] active=True part_enabled=True "
                f"alpha_scale={gaussians.part_budget_alpha_scale:.6f} "
                f"alpha={getattr(dataset, 'part_budget_alpha', 1.0)} "
                f"warmup={getattr(dataset, 'part_budget_warmup', 1000)}"
            )
        if (
            getattr(dataset, "use_tri_token", False)
            and iteration == int(getattr(dataset, "token_tri_start_iter", 10000)) + 1
        ):
            tri_token_part_enabled = (
                getattr(dataset, "use_part_moe", False)
                and gaussians.part_label_enabled
                and (
                    gaussians.non_rigid_deformer.part_moe_active
                    or gaussians.non_rigid_deformer.temporal_conditioned_part_active
                )
            )
            print(
                f"[TRI_TOKEN Status] active=True part_enabled={tri_token_part_enabled} "
                f"alpha_scale={gaussians.tri_token_alpha_scale:.6f} "
                f"alpha={getattr(dataset, 'token_tri_alpha', 1.0)} "
                f"warmup={getattr(dataset, 'token_tri_warmup', 1000)}"
            )
        if (
            getattr(dataset, "use_tri_token", False)
            and getattr(dataset, "token_tri_fusion_mode", "") == "route_hard"
            and gaussians.part_label_enabled
            and (
                gaussians.non_rigid_deformer.part_moe_active
                or gaussians.non_rigid_deformer.temporal_conditioned_part_active
            )
            and iteration == int(getattr(dataset, "token_tri_start_iter", 10000)) + 1
        ):
            print(
                f"[TRI_TOKEN_HARD Status] active=True part_enabled=True "
                f"alpha_scale={gaussians.tri_token_alpha_scale:.6f} "
                f"alpha={getattr(dataset, 'token_tri_alpha', 1.0)} "
                f"warmup={getattr(dataset, 'token_tri_warmup', 1000)} "
                f"boundary_mix={getattr(dataset, 'token_tri_route_hard_boundary_mix', 0.65)} "
                f"motion_mix={getattr(dataset, 'token_tri_route_hard_motion_mix', 0.35)} "
                f"supervision_w={getattr(dataset, 'token_tri_route_hard_w', 0.0)}"
            )

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Start timer
        start_time = time.time()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, iteration=iteration)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()
        bound_mask_full = viewpoint_cam.bound_mask.cuda()
        bound_mask = bound_mask_full
        # crop the object region
        x1, y1, x2, y2 = masks_to_boxes(bound_mask).int().squeeze(0)
        img_pred_rect = image[:, y1:y2+1, x1:x2+1].unsqueeze(0)
        img_gt_rect = gt_image[:, y1:y2+1, x1:x2+1].unsqueeze(0)
        bound_mask = bound_mask[0] == 1
        point_anchor_parent_attrs = None
        point_anchor_stats = None
        point_tb_parent_attrs = None
        point_tb_stats = None
        point_tb_mask = None
        point_cloth_parent_attrs = None
        point_cloth_stats = None
        point_cloth_mask = None
        point_cloth_protect_mask = None
        point_depth_parent_attrs = None
        point_depth_stats = None
        if (
            getattr(dataset, "use_point", False)
            and iteration >= int(getattr(dataset, "point_start_iter", 800))
            and iteration % max(1, int(getattr(dataset, "point_interval", 100))) == 0
            and iteration < opt.densify_until_iter
        ):
            point_guided_mask, point_stats = select_point_guided_anchors(
                viewpoint_camera=viewpoint_cam,
                deformed_means3D=render_pkg["deformed_means3D"].detach(),
                visibility_filter=visibility_filter,
                image=image,
                gt_image=gt_image,
                bound_mask=bound_mask_full,
                patch_size=getattr(dataset, "point_patch_size", 32),
                topk_patches=getattr(dataset, "point_topk_patches", 16),
                anchor_radius=getattr(dataset, "point_anchor_radius", 20.0),
                max_anchors=getattr(dataset, "point_max_anchors", 512),
                min_patch_coverage=getattr(dataset, "point_min_patch_coverage", 0.20),
            )
        else:
            point_guided_mask = None
            point_stats = None
        Ll1 = l1_loss_masked(image, gt_image, bound_mask)
        alpha_loss = l2_loss_masked(alpha, bkgd_mask, bound_mask)
        # ssim loss
        ssim_loss = ssim(img_pred_rect, img_gt_rect)
        # lipis loss
        lpips_loss = loss_fn_vgg(img_pred_rect, img_gt_rect).squeeze()

        loss = opt.l1_loss_w * Ll1 + 0.1 * alpha_loss + opt.ssim_loss_w * (1.0 - ssim_loss) + opt.lpips_loss_w * lpips_loss

        # iospos ioscov loss
        if getattr(dataset, "use_dynomo_c", False):
            dynomo_part_label = scene.gaussians.get_part_label if scene.gaussians.part_label_enabled else None
            dynomo_part_conf = scene.gaussians.get_part_conf if scene.gaussians.part_label_enabled else None
            dynomo_affinity = getattr(scene.gaussians.non_rigid_deformer, "last_dynomo_affinity", None)
            d_xyz, d_rotation, d_scaling = render_pkg.get("d_nonrigid", (None, None, None))
            motion_obs = d_xyz.squeeze(0) if d_xyz is not None else None
            rotation_obs = d_rotation.squeeze(0) if d_rotation is not None else None
            loss_aiap_xyz, loss_aiap_cov, loss_aiap_motion, loss_aiap_rotation = weighted_aiap_loss(
                scene.gaussians.get_xyz,
                render_pkg["deformed_means3D"],
                scene.gaussians.get_covariance(),
                render_pkg["deformed_cov3D"],
                affinity_feat=dynomo_affinity,
                part_label=dynomo_part_label,
                part_weight_matrix=dynomo_part_weight_matrix,
                part_conf=dynomo_part_conf,
                motion_obs=motion_obs,
                rotation_obs=rotation_obs,
                n_neighbors=int(getattr(opt, "dynomo_c_knn", 5)),
            )
            loss = loss + opt.iospos_w * loss_aiap_xyz + opt.ioscov_w * loss_aiap_cov
            loss = loss + float(getattr(opt, "dynomo_c_motion_w", 0.01)) * loss_aiap_motion
            loss = loss + float(getattr(opt, "dynomo_c_rotation_w", 0.01)) * loss_aiap_rotation
        else:
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(
                scene.gaussians.get_xyz,
                render_pkg["deformed_means3D"],
                scene.gaussians.get_covariance(),
                render_pkg["deformed_cov3D"],
            )
            loss = loss + opt.iospos_w * loss_aiap_xyz + opt.ioscov_w * loss_aiap_cov
        tri_part_reg_loss = None
        if getattr(dataset, "use_tri_part", False):
            tri_part_reg_loss = getattr(scene.gaussians.non_rigid_deformer, "last_tri_part_reg_loss", None)
            tri_part_reg_w = float(getattr(dataset, "tri_part_reg_w", 0.0))
            if tri_part_reg_loss is not None and tri_part_reg_w > 0.0:
                loss = loss + tri_part_reg_w * tri_part_reg_loss
        part_budget_sup_loss = None
        if getattr(dataset, "use_part_budget", False):
            part_budget_sup_loss = getattr(scene.gaussians.non_rigid_deformer, "last_part_budget_loss", None)
            if part_budget_sup_loss is not None:
                loss = loss + part_budget_sup_loss
        tri_token_route_loss = None
        if getattr(dataset, "use_tri_token", False):
            tri_token_route_loss = getattr(scene.gaussians.non_rigid_deformer, "last_tri_token_loss", None)
            if tri_token_route_loss is not None:
                loss = loss + tri_token_route_loss

        loss.backward()

        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        if (iteration in testing_iterations):
            print("[Elapsed time]: ", elapsed_time) 

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            Ll1_loss_for_log = 0.4 * Ll1.item() + 0.6 * Ll1_loss_for_log
            mask_loss_for_log = 0.4 * alpha_loss.item() + 0.6 * mask_loss_for_log
            ssim_loss_for_log = 0.4 * ssim_loss.item() + 0.6 * ssim_loss_for_log
            lpips_loss_for_log = 0.4 * lpips_loss.item() + 0.6 * lpips_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}", "mask Loss": f"{mask_loss_for_log:.{2}f}",
                                          "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}"})
                progress_bar.update(10)
            if getattr(dataset, "use_tri_part", False) and iteration % 1000 == 0:
                stats = getattr(scene.gaussians.non_rigid_deformer, "last_tri_part_stats", None)
                if stats is not None:
                    print(
                        "[TRI_PART Stats] "
                        f"iter={iteration} "
                        f"gate_mean={stats['gate_mean'].item():.6f} "
                        f"gate_std={stats['gate_std'].item():.6f} "
                        f"motion_mean={stats['motion_mean'].item():.6f} "
                        f"boundary_mean={stats['boundary_mean'].item():.6f} "
                        f"residual_norm={stats['residual_norm'].item():.6f} "
                        f"reg={stats['reg_loss'].detach().item():.8f}"
                    )
            if getattr(dataset, "use_part_budget", False) and iteration % 1000 == 0:
                stats = getattr(scene.gaussians.non_rigid_deformer, "last_part_budget_stats", None)
                if stats is not None:
                    budget_mean = stats["budget_mean"].detach().cpu().tolist()
                    budget_std = stats["budget_std"].detach().cpu().tolist()
                    budget_min = stats["budget_min"].detach().cpu().tolist()
                    budget_max = stats["budget_max"].detach().cpu().tolist()
                    print(
                        "[PartBudget Stats] "
                        f"iter={iteration} "
                        f"mean={budget_mean} "
                        f"std={budget_std} "
                        f"min={budget_min} "
                        f"max={budget_max} "
                        f"feature_delta_norm={stats['feature_delta_norm'].item():.6f} "
                        f"adapter_norm={stats['adapter_norm'].item():.6f} "
                        f"entropy={stats['entropy'].item():.6f} "
                        f"logits_mean={stats['logits_mean'].item():.6f} "
                        f"motion_mean={stats['motion_mean'].item():.6f} "
                        f"boundary_mean={stats['boundary_mean'].item():.6f} "
                        f"query_norm={stats['query_norm'].item():.6f} "
                        f"budget_kl_uniform={stats['budget_kl_uniform'].item():.6f} "
                        f"routed_budget_kl_uniform={stats['routed_budget_kl_uniform'].item():.6f} "
                        f"budget_target_loss={stats['budget_target_loss'].item():.6f}"
                    )
                    if part_budget_sup_loss is not None:
                        print(
                            "[PartBudget Sup] "
                            f"iter={iteration} "
                            f"loss={part_budget_sup_loss.detach().item():.8f}"
                        )
                    per_part_mean = stats["per_part_budget_mean"].detach().mean(dim=0).cpu().tolist()
                    per_part_std = stats["per_part_budget_std"].detach().mean(dim=0).cpu().tolist()
                    for pid, (mean_vals, std_vals) in enumerate(zip(per_part_mean, per_part_std)):
                        print(
                            f"  [PartBudget p{pid}] "
                            f"mean={mean_vals} std={std_vals}"
                        )
            if getattr(dataset, "use_part_score_route", False) and iteration % 1000 == 0:
                stats = getattr(scene.gaussians.non_rigid_deformer, "last_part_score_route_stats", None)
                if stats is not None:
                    msg = (
                        "[PART_SCORE_ROUTE Stats] "
                        f"iter={iteration} "
                        f"part_weight_mean={stats['score_route_part_weight_mean'].item():.6f} "
                        f"part_weight_std={stats['score_route_part_weight_std'].item():.6f} "
                        f"base_part_weight_mean={stats['score_route_base_part_weight_mean'].item():.6f} "
                        f"gate_mean={stats['score_route_gate_mean'].item():.6f} "
                        f"gate_std={stats['score_route_gate_std'].item():.6f} "
                        f"focus_mean={stats['score_route_focus_mean'].item():.6f} "
                        f"motion_mean={stats['score_route_motion_mean'].item():.6f} "
                        f"boundary_mean={stats['score_route_boundary_mean'].item():.6f} "
                        f"unknown_mean={stats['score_route_unknown_mean'].item():.6f} "
                        f"query_norm={stats['score_route_query_norm'].item():.6f}"
                    )
                    if "score_route_use_route_gate" in stats:
                        msg += f" use_route_gate={stats['score_route_use_route_gate'].item():.0f}"
                    if "score_route_use_unknown_mix" in stats:
                        msg += f" use_unknown_mix={stats['score_route_use_unknown_mix'].item():.0f}"
                    if "score_route_signal_mode_unknown_only" in stats:
                        msg += f" signal_mode_unknown_only={stats['score_route_signal_mode_unknown_only'].item():.0f}"
                    if "score_route_unknown_weight_mean" in stats:
                        msg += f" unknown_weight_mean={stats['score_route_unknown_weight_mean'].item():.6f}"
                    if "score_route_unknown_weight_std" in stats:
                        msg += f" unknown_weight_std={stats['score_route_unknown_weight_std'].item():.6f}"
                    if "score_route_unknown_route_entropy" in stats:
                        msg += f" unknown_route_entropy={stats['score_route_unknown_route_entropy'].item():.6f}"
                    if "score_route_unknown_count" in stats:
                        msg += f" unknown_count={stats['score_route_unknown_count'].item():.0f}"
                    print(msg)
            if getattr(dataset, "use_time", False) and iteration % 1000 == 0:
                stats = getattr(scene.gaussians.non_rigid_deformer, "last_time_stats", None)
                if stats is not None:
                    scale_mean = stats["scale_mean"].detach().cpu().tolist()
                    print(
                        "[TIME Stats] "
                        f"iter={iteration} "
                        f"alpha_mean={stats['alpha_mean'].item():.6f} "
                        f"alpha_std={stats['alpha_std'].item():.6f} "
                        f"alpha_min={stats['alpha_min'].item():.6f} "
                        f"alpha_max={stats['alpha_max'].item():.6f} "
                        f"entropy={stats['alpha_entropy'].item():.6f} "
                        f"scale_mean={scale_mean}"
                    )
            if getattr(dataset, "use_tri_token", False) and iteration % 1000 == 0:
                stats = getattr(scene.gaussians.non_rigid_deformer, "last_tri_token_stats", None)
                if stats is not None:
                    msg = (
                        "[TRI_TOKEN Stats] "
                        f"iter={iteration} "
                        f"alpha_scale={stats['alpha_scale'].item():.6f} "
                        f"plane_mean={stats['plane_mean'].item():.6f} "
                        f"plane_std={stats['plane_std'].item():.6f} "
                        f"plane_delta_norm={stats['plane_delta_norm'].item():.6f} "
                        f"raw_feature_norm={stats['raw_feature_norm'].item():.6f} "
                        f"projected_norm={stats['projected_norm'].item():.6f} "
                        f"token_std={stats['token_std'].item():.6f} "
                        f"part_stats_mean={stats['part_stats_mean'].item():.6f}"
                    )
                    if "route_gate_mean" in stats:
                        msg += (
                            f" route_gate_mean={stats['route_gate_mean'].item():.6f} "
                            f"route_gate_std={stats['route_gate_std'].item():.6f} "
                            f"route_delta_norm={stats['route_delta_norm'].item():.6f} "
                            f"routed_token_norm={stats['routed_token_norm'].item():.6f} "
                            f"motion_mean={stats['motion_mean'].item():.6f} "
                            f"boundary_mean={stats['boundary_mean'].item():.6f} "
                        f"boundary_focus_mean={stats['boundary_focus_mean'].item():.6f} "
                        f"query_norm={stats['query_norm'].item():.6f}"
                    )
                    if "route_boundary_target_mean" in stats:
                        msg += f" route_boundary_target_mean={stats['route_boundary_target_mean'].item():.6f}"
                    if "route_boundary_loss" in stats:
                        msg += f" route_boundary_loss={stats['route_boundary_loss'].item():.8f}"
                    if "route_output_gate_mean" in stats:
                        msg += (
                            f" route_output_gate_mean={stats['route_output_gate_mean'].item():.6f} "
                            f"route_output_gate_std={stats['route_output_gate_std'].item():.6f} "
                            f"route_output_delta_norm={stats['route_output_delta_norm'].item():.6f} "
                            f"route_output_token_norm={stats['route_output_token_norm'].item():.6f} "
                            f"route_output_mean={stats['route_output_mean'].item():.6f} "
                            f"route_output_std={stats['route_output_std'].item():.6f} "
                            f"route_output_boundary_mean={stats['route_output_boundary_mean'].item():.6f} "
                            f"route_output_focus_mean={stats['route_output_focus_mean'].item():.6f}"
                        )
                    if "route_hard_gate_mean" in stats:
                        msg += (
                            f" route_hard_gate_mean={stats['route_hard_gate_mean'].item():.6f} "
                            f"route_hard_gate_std={stats['route_hard_gate_std'].item():.6f} "
                            f"route_hard_expert_gate_mean={stats['route_hard_expert_gate_mean'].item():.6f} "
                            f"route_hard_expert_gate_std={stats['route_hard_expert_gate_std'].item():.6f} "
                            f"route_hard_delta_norm={stats['route_hard_delta_norm'].item():.6f} "
                            f"route_hard_part_alpha_mean={stats['route_hard_part_alpha_mean'].item():.6f} "
                            f"route_hard_motion_mean={stats['route_hard_motion_mean'].item():.6f} "
                            f"route_hard_motion_focus_mean={stats['route_hard_motion_focus_mean'].item():.6f} "
                            f"route_hard_boundary_mean={stats['route_hard_boundary_mean'].item():.6f} "
                            f"route_hard_focus_mean={stats['route_hard_focus_mean'].item():.6f} "
                            f"route_hard_query_norm={stats['route_hard_query_norm'].item():.6f}"
                        )
                    if "route_hard_loss" in stats:
                        msg += f" route_hard_loss={stats['route_hard_loss'].item():.8f}"
                    if "fusion_part_weight_mean" in stats:
                        msg += (
                            f" fusion_part_weight_mean={stats['fusion_part_weight_mean'].item():.6f} "
                            f"fusion_part_weight_std={stats['fusion_part_weight_std'].item():.6f} "
                            f"fusion_base_part_weight_mean={stats['fusion_base_part_weight_mean'].item():.6f} "
                            f"fusion_delta_mean={stats['fusion_delta_mean'].item():.6f} "
                            f"fusion_delta_abs_mean={stats['fusion_delta_abs_mean'].item():.6f} "
                            f"fusion_delta_unit_std={stats['fusion_delta_unit_std'].item():.6f} "
                            f"fusion_global_weight_mean={stats['fusion_global_weight_mean'].item():.6f} "
                            f"fusion_boundary_mean={stats['fusion_boundary_mean'].item():.6f} "
                            f"fusion_motion_mean={stats['fusion_motion_mean'].item():.6f} "
                            f"fusion_query_norm={stats['fusion_query_norm'].item():.6f}"
                        )
                    if "spatial_part_weight_mean" in stats:
                        msg += (
                            f" spatial_part_weight_mean={stats['spatial_part_weight_mean'].item():.6f} "
                            f"spatial_part_weight_std={stats['spatial_part_weight_std'].item():.6f} "
                            f"spatial_base_part_weight_mean={stats['spatial_base_part_weight_mean'].item():.6f} "
                            f"spatial_delta_mean={stats['spatial_delta_mean'].item():.6f} "
                            f"spatial_delta_abs_mean={stats['spatial_delta_abs_mean'].item():.6f} "
                            f"spatial_delta_unit_std={stats['spatial_delta_unit_std'].item():.6f} "
                            f"spatial_route_gate_mean={stats['spatial_route_gate_mean'].item():.6f} "
                            f"spatial_route_gate_std={stats['spatial_route_gate_std'].item():.6f} "
                            f"spatial_spatial_gate_mean={stats['spatial_spatial_gate_mean'].item():.6f} "
                            f"spatial_spatial_gate_std={stats['spatial_spatial_gate_std'].item():.6f} "
                            f"spatial_part_gate_mean={stats['spatial_part_gate_mean'].item():.6f} "
                            f"spatial_part_gate_std={stats['spatial_part_gate_std'].item():.6f} "
                            f"spatial_global_weight_mean={stats['spatial_global_weight_mean'].item():.6f} "
                            f"spatial_boundary_mean={stats['spatial_boundary_mean'].item():.6f} "
                            f"spatial_motion_mean={stats['spatial_motion_mean'].item():.6f} "
                            f"spatial_focus_mean={stats['spatial_focus_mean'].item():.6f} "
                            f"spatial_query_norm={stats['spatial_query_norm'].item():.6f}"
                        )
                    if "spatial_loss" in stats:
                        msg += f" spatial_loss={stats['spatial_loss'].item():.8f}"
                    print(msg)
            if getattr(dataset, "use_mapo_all_dynamic", False) and getattr(
                dataset, "mapo_dynamic_score_enabled", False
            ) and iteration % 1000 == 0:
                stats = getattr(scene.gaussians.non_rigid_deformer, "last_mapo_dynamic_stats", None)
                if stats is not None:
                    print(
                        "[MAPO_DYNAMIC Stats] "
                        f"iter={iteration} mean={stats['mean']:.8f} "
                        f"p50={stats['p50']:.8f} p95={stats['p95']:.8f} max={stats['max']:.8f}"
                    )
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), saving_iterations)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Start timer
            start_time = time.time()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if (
                    getattr(dataset, "use_point_update", False)
                    or getattr(dataset, "use_point_cloth_budget", False)
                    or getattr(dataset, "use_vggt_garment", False)
                ):
                    gaussians.update_point_value_ema(
                        getattr(dataset, "point_update_ema_momentum", 0.95)
                    )

                # VGGT strict modes consume the current render before regular
                # densification changes the point index space.
                if (
                    vggt_garment_controller is not None
                    and bool(getattr(dataset, "vggt_strict_budget", False))
                    and iteration < opt.densify_until_iter
                ):
                    vggt_garment_controller.after_iteration(
                        iteration,
                        scene,
                        gaussians,
                        context={
                            "viewpoint_camera": viewpoint_cam,
                            "deformed_means3D": render_pkg["deformed_means3D"].detach(),
                            "visibility_filter": visibility_filter,
                            "image": image,
                            "gt_image": gt_image,
                            "bound_mask": bound_mask_full,
                        },
                    )

                if (
                    getattr(dataset, "use_point_anchor", False)
                    and iteration >= int(getattr(dataset, "point_anchor_start_iter", 800))
                    and iteration <= int(getattr(dataset, "point_anchor_end_iter", 1800))
                    and iteration % max(1, int(getattr(dataset, "point_anchor_interval", 100))) == 0
                ):
                    point_anchor_mask, point_anchor_stats = select_point_guided_anchors(
                        viewpoint_camera=viewpoint_cam,
                        deformed_means3D=render_pkg["deformed_means3D"].detach(),
                        visibility_filter=visibility_filter,
                        image=image,
                        gt_image=gt_image,
                        bound_mask=bound_mask_full,
                        patch_size=getattr(dataset, "point_anchor_patch_size", 32),
                        topk_patches=getattr(dataset, "point_anchor_topk", 16),
                        anchor_radius=getattr(dataset, "point_anchor_radius", 20.0),
                        max_anchors=getattr(dataset, "point_anchor_max_anchors", 512),
                        min_patch_coverage=getattr(dataset, "point_anchor_min_coverage", 0.20),
                    )
                    point_anchor_parent_attrs = gaussians.cache_point_anchor_parents(
                        torch.where(point_anchor_mask)[0]
                    )
                if (
                    (getattr(dataset, "use_point_anchor_tb", False) or getattr(dataset, "use_point_update", False))
                    and iteration >= int(getattr(dataset, "point_tb_start_iter", 800))
                    and iteration <= int(getattr(dataset, "point_tb_end_iter", 1800))
                    and iteration % max(1, int(getattr(dataset, "point_tb_interval", 100))) == 0
                ):
                    point_update_nonrigid_beta = float(getattr(dataset, "point_update_nonrigid_beta", 0.0))
                    point_update_nonrigid_norm = None
                    if point_update_nonrigid_beta > 0.0:
                        d_nonrigid = render_pkg.get("d_nonrigid", None)
                        if d_nonrigid is not None and d_nonrigid[0] is not None:
                            d_xyz = d_nonrigid[0].detach().reshape(-1, 3)
                            point_update_nonrigid_norm = d_xyz.norm(dim=-1)
                            point_update_nonrigid_norm = point_update_nonrigid_norm / point_update_nonrigid_norm.max().clamp_min(1e-6)
                    point_update_edge_beta = float(getattr(dataset, "point_update_edge_beta", 0.0))
                    point_update_edge_kernel = int(getattr(dataset, "point_update_edge_kernel", 3))
                    point_tb_mask, point_tb_history, point_tb_stats = select_point_anchor_tb_anchors(
                        viewpoint_camera=viewpoint_cam,
                        deformed_means3D=render_pkg["deformed_means3D"].detach(),
                        visibility_filter=visibility_filter,
                        image=image,
                        gt_image=gt_image,
                        bound_mask=bound_mask_full,
                        patch_size=getattr(dataset, "point_tb_patch_size", 32),
                        topk_patches=getattr(dataset, "point_tb_topk", 16),
                        anchor_radius=getattr(dataset, "point_tb_anchor_radius", 20.0),
                        max_anchors=getattr(dataset, "point_tb_max_anchors", 512),
                        min_patch_coverage=getattr(dataset, "point_tb_min_coverage", 0.20),
                        history_score=point_tb_history,
                        temporal_alpha=getattr(dataset, "point_tb_temporal_alpha", 0.5),
                        history_momentum=getattr(dataset, "point_tb_history_momentum", 0.8),
                        boundary_beta=getattr(dataset, "point_tb_boundary_beta", 0.5),
                        boundary_kernel=getattr(dataset, "point_tb_boundary_kernel", 5),
                        edge_beta=point_update_edge_beta,
                        edge_kernel=point_update_edge_kernel,
                        nonrigid_beta=point_update_nonrigid_beta,
                        nonrigid_norm=point_update_nonrigid_norm,
                        min_persistence=getattr(dataset, "point_tb_min_persistence", 0.0),
                    )
                    if point_tb_mask is not None:
                        point_tb_parent_attrs = gaussians.cache_point_anchor_parents(
                            torch.where(point_tb_mask)[0]
                        )
                if (
                    getattr(dataset, "use_point_cloth_budget", False)
                    and iteration >= int(getattr(dataset, "point_cloth_start_iter", 800))
                    and iteration <= int(getattr(dataset, "point_cloth_end_iter", 1800))
                    and iteration % max(1, int(getattr(dataset, "point_cloth_interval", 100))) == 0
                ):
                    d_xyz_norm = None
                    d_nonrigid = render_pkg.get("d_nonrigid", None)
                    if d_nonrigid is not None and d_nonrigid[0] is not None:
                        d_xyz = d_nonrigid[0].detach().reshape(-1, 3)
                        d_xyz_norm = d_xyz.norm(dim=-1)
                        d_xyz_norm = d_xyz_norm / d_xyz_norm.max().clamp_min(1e-6)
                    (
                        point_cloth_mask,
                        point_cloth_protect_mask,
                        point_cloth_history,
                        point_cloth_stats,
                    ) = select_point_cloth_budget_anchors(
                        viewpoint_camera=viewpoint_cam,
                        deformed_means3D=render_pkg["deformed_means3D"].detach(),
                        visibility_filter=visibility_filter,
                        image=image,
                        gt_image=gt_image,
                        bound_mask=bound_mask_full,
                        patch_size=getattr(dataset, "point_cloth_patch_size", 32),
                        topk_patches=getattr(dataset, "point_cloth_topk", 16),
                        anchor_radius=getattr(dataset, "point_cloth_anchor_radius", 20.0),
                        max_anchors=getattr(dataset, "point_cloth_max_anchors", 512),
                        min_patch_coverage=getattr(dataset, "point_cloth_min_coverage", 0.20),
                        history_score=point_cloth_history,
                        temporal_alpha=getattr(dataset, "point_cloth_temporal_alpha", 0.5),
                        history_momentum=getattr(dataset, "point_cloth_history_momentum", 0.8),
                        boundary_beta=getattr(dataset, "point_cloth_boundary_beta", 1.0),
                        hf_beta=getattr(dataset, "point_cloth_hf_beta", 0.0),
                        nonrigid_beta=getattr(dataset, "point_cloth_nonrigid_beta", 0.0),
                        boundary_kernel=getattr(dataset, "point_cloth_boundary_kernel", 5),
                        protect_thresh=getattr(dataset, "point_cloth_protect_thresh", 0.6),
                        min_persistence=getattr(dataset, "point_cloth_min_persistence", 0.0),
                        nonrigid_norm=d_xyz_norm,
                    )
                    if point_cloth_mask is not None:
                        point_cloth_parent_attrs = gaussians.cache_point_anchor_parents(
                            torch.where(point_cloth_mask)[0]
                        )
                if (
                    getattr(dataset, "use_point_depth", False)
                    and iteration >= int(getattr(dataset, "point_depth_start_iter", 800))
                    and iteration <= int(getattr(dataset, "point_depth_end_iter", 1800))
                    and iteration % max(1, int(getattr(dataset, "point_depth_interval", 100))) == 0
                ):
                    point_depth_mask, point_depth_surface_xyz, point_depth_stats = select_point_depth_anchors(
                        viewpoint_camera=viewpoint_cam,
                        canonical_xyz=gaussians.get_xyz.detach(),
                        canonical_vertices=gaussians.canon_vertices.detach(),
                        deformed_means3D=render_pkg["deformed_means3D"].detach(),
                        visibility_filter=visibility_filter,
                        image=image,
                        gt_image=gt_image,
                        bound_mask=bound_mask_full,
                        render_depth=render_pkg["depth"],
                        render_alpha=alpha,
                        patch_size=getattr(dataset, "point_depth_patch_size", 32),
                        topk_patches=getattr(dataset, "point_depth_topk", 16),
                        anchor_radius=getattr(dataset, "point_depth_anchor_radius", 20.0),
                        max_anchors=getattr(dataset, "point_depth_max_anchors", 512),
                        min_patch_coverage=getattr(dataset, "point_depth_min_coverage", 0.20),
                        depth_threshold=getattr(dataset, "point_depth_depth_threshold", 0.08),
                        depth_relative=getattr(dataset, "point_depth_depth_relative", 0.04),
                        surface_threshold=getattr(dataset, "point_depth_surface_threshold", 0.12),
                    )
                    point_depth_parent_attrs = gaussians.cache_point_depth_parents(
                        torch.where(point_depth_mask)[0],
                        point_depth_surface_xyz,
                    )
                if point_guided_mask is not None:
                    boosted = gaussians.boost_guided_densification(
                        point_guided_mask,
                        opt.densify_grad_threshold * float(getattr(dataset, "point_grad_boost", 2.0)),
                    )
                    print(
                        "[POINT Stats] "
                        f"iter={iteration} "
                        f"patches={point_stats['patches']} "
                        f"anchors={boosted} "
                        f"error_mean={point_stats['error_mean']:.6f} "
                        f"coverage_mean={point_stats['coverage_mean']:.6f} "
                        f"point_count_before={len(gaussians.get_xyz)}"
                    )

                point_depth_fixed_budget_window = (
                    getattr(dataset, "use_point_depth", False)
                    and bool(getattr(dataset, "point_depth_fixed_budget", True))
                    and iteration >= int(getattr(dataset, "point_depth_start_iter", 800))
                    and iteration <= int(getattr(dataset, "point_depth_end_iter", 1800))
                )
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and len(gaussians.get_xyz) < 120000
                    and not point_depth_fixed_budget_window
                ):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

                if point_anchor_parent_attrs is not None:
                    spawned = gaussians.spawn_from_cached_anchors(
                        point_anchor_parent_attrs,
                        children_per_anchor=getattr(dataset, "point_anchor_children_per_anchor", 1),
                        offset_scale=getattr(dataset, "point_anchor_offset_scale", 0.35),
                        scale_ratio=getattr(dataset, "point_anchor_scale_ratio", 0.7),
                        opacity_ratio=getattr(dataset, "point_anchor_opacity_ratio", 0.8),
                        opacity_min=getattr(dataset, "point_anchor_opacity_min", 0.01),
                        max_points=getattr(dataset, "point_anchor_max_points", 120000),
                    )
                    print(
                        "[POINT_ANCHOR] "
                        f"iter={iteration} "
                        f"patches={point_anchor_stats['patches']} "
                        f"anchors={point_anchor_stats['anchors']} "
                        f"spawned={spawned} "
                        f"total_points={len(gaussians.get_xyz)} "
                        f"patch_error_mean={point_anchor_stats['error_mean']:.6f} "
                        f"coverage_mean={point_anchor_stats['coverage_mean']:.6f}"
                    )

                use_point_update = bool(getattr(dataset, "use_point_update", False))
                use_point_update_nonrigid = float(getattr(dataset, "point_update_nonrigid_beta", 0.0)) > 0.0
                use_point_update_edge = float(getattr(dataset, "point_update_edge_beta", 0.0)) > 0.0
                if point_tb_stats is not None:
                    if point_tb_parent_attrs is not None:
                        spawned = gaussians.spawn_from_cached_anchors(
                            point_tb_parent_attrs,
                            children_per_anchor=getattr(dataset, "point_tb_children_per_anchor", 1),
                            offset_scale=getattr(dataset, "point_tb_offset_scale", 0.35),
                            scale_ratio=getattr(dataset, "point_tb_scale_ratio", 0.7),
                            opacity_ratio=getattr(dataset, "point_tb_opacity_ratio", 0.8),
                            opacity_min=getattr(dataset, "point_tb_opacity_min", 0.01),
                            max_points=getattr(dataset, "point_tb_max_points", 120000),
                        )
                    else:
                        spawned = 0
                    replaced = 0
                    target_points = (
                        get_point_update_target_points(dataset)
                        if (use_point_update or use_point_update_nonrigid or use_point_update_edge)
                        else int(getattr(dataset, "point_tb_target_points", 0))
                    )
                    replace_after_iter = int(
                        getattr(
                            dataset,
                            "point_tb_replace_after_iter",
                            int(getattr(dataset, "point_tb_start_iter", 800))
                            if (use_point_update or use_point_update_nonrigid or use_point_update_edge)
                            else 1500,
                        )
                    )
                    replacement_ratio = float(getattr(dataset, "point_tb_replacement_ratio", 1.0))
                    if (
                        target_points > 0
                        and iteration >= replace_after_iter
                        and len(gaussians.get_xyz) > target_points
                    ):
                        overflow = len(gaussians.get_xyz) - target_points
                        replace_count = int(max(0, overflow) * max(0.0, replacement_ratio))
                        replaced = gaussians.replace_low_value_points(
                            replace_count,
                            exclude_last=spawned,
                            opacity_w=getattr(dataset, "point_tb_replace_opacity_w", 1.0),
                            gradient_w=getattr(dataset, "point_tb_replace_gradient_w", 0.25),
                            visibility_w=getattr(dataset, "point_tb_replace_visibility_w", 0.10),
                            use_long_term=(use_point_update or use_point_update_nonrigid or use_point_update_edge),
                            protect_mask=(
                                point_tb_mask
                                if (use_point_update or use_point_update_nonrigid or use_point_update_edge)
                                and bool(getattr(dataset, "point_update_protect_anchors", True))
                                else None
                            ),
                            min_keep=getattr(dataset, "point_update_min_keep", 1024),
                        )
                    log_name = (
                        "[PART_POINT]"
                        if bool(getattr(dataset, "use_part_point", False))
                        else (
                            "[POINT_UPDATE_NONRIGID]"
                            if use_point_update_nonrigid
                            else ("[POINT_UPDATE_EDGE]" if use_point_update_edge else ("[POINT_UPDATE]" if use_point_update else "[POINT_TB]"))
                        )
                    )
                    log_msg = (
                        f"{log_name} "
                        f"iter={iteration} "
                        f"patches={point_tb_stats['patches']} "
                        f"anchors={point_tb_stats['anchors']} "
                        f"spawned={spawned} "
                        f"replaced={replaced} "
                        f"target_points={target_points} "
                        f"total_points={len(gaussians.get_xyz)} "
                        f"error_mean={point_tb_stats['error_mean']:.6f} "
                        f"boundary_mean={point_tb_stats['boundary_mean']:.6f} "
                        f"coverage_mean={point_tb_stats['coverage_mean']:.6f} "
                        f"history_mean={point_tb_stats['history_mean']:.6f}"
                    )
                    if use_point_update_nonrigid:
                        log_msg += f" nonrigid_mean={point_tb_stats['nonrigid_mean']:.6f}"
                    if use_point_update_edge:
                        log_msg += f" edge_mean={point_tb_stats['edge_mean']:.6f}"
                    print(
                        log_msg,
                        flush=True,
                    )

            if point_cloth_stats is not None:
                if point_cloth_parent_attrs is not None:
                    spawned = gaussians.spawn_from_cached_anchors(
                        point_cloth_parent_attrs,
                        children_per_anchor=getattr(dataset, "point_cloth_children_per_anchor", 1),
                        offset_scale=getattr(dataset, "point_cloth_offset_scale", 0.35),
                        scale_ratio=getattr(dataset, "point_cloth_scale_ratio", 0.7),
                        opacity_ratio=getattr(dataset, "point_cloth_opacity_ratio", 0.8),
                        opacity_min=getattr(dataset, "point_cloth_opacity_min", 0.01),
                        max_points=getattr(dataset, "point_cloth_max_points", 120000),
                    )
                else:
                    spawned = 0
                replaced = 0
                target_points = get_point_update_target_points(dataset)
                replace_after_iter = int(getattr(dataset, "point_cloth_replace_after_iter", 800))
                replacement_ratio = float(getattr(dataset, "point_cloth_replacement_ratio", 1.0))
                if (
                    target_points > 0
                    and iteration >= replace_after_iter
                    and len(gaussians.get_xyz) > target_points
                ):
                    overflow = len(gaussians.get_xyz) - target_points
                    replace_count = int(max(0, overflow) * max(0.0, replacement_ratio))
                    replaced = gaussians.replace_low_value_points(
                        replace_count,
                        exclude_last=spawned,
                        opacity_w=getattr(dataset, "point_tb_replace_opacity_w", 1.0),
                        gradient_w=getattr(dataset, "point_tb_replace_gradient_w", 0.25),
                        visibility_w=getattr(dataset, "point_tb_replace_visibility_w", 0.10),
                        use_long_term=True,
                        protect_mask=point_cloth_protect_mask,
                        min_keep=getattr(dataset, "point_update_min_keep", 1024),
                    )
                print(
                    "[POINT_CLOTH] "
                    f"iter={iteration} "
                    f"patches={point_cloth_stats['patches']} "
                    f"anchors={point_cloth_stats['anchors']} "
                    f"spawned={spawned} "
                    f"replaced={replaced} "
                    f"target_points={target_points} "
                    f"total_points={len(gaussians.get_xyz)} "
                    f"error_mean={point_cloth_stats['error_mean']:.6f} "
                    f"boundary_mean={point_cloth_stats['boundary_mean']:.6f} "
                    f"highfreq_mean={point_cloth_stats['highfreq_mean']:.6f} "
                    f"nonrigid_mean={point_cloth_stats['nonrigid_mean']:.6f} "
                    f"coverage_mean={point_cloth_stats['coverage_mean']:.6f} "
                    f"history_mean={point_cloth_stats['history_mean']:.6f} "
                    f"protect_ratio={point_cloth_stats['protect_ratio']:.6f} "
                    f"anchor_score_mean={point_cloth_stats['anchor_score_mean']:.6f}",
                    flush=True,
                )

            if getattr(dataset, "use_point_update", False) or getattr(dataset, "use_point_cloth_budget", False):
                final_clamp_iter = int(getattr(dataset, "point_update_final_clamp_iter", 0))
                clamp_interval = max(1, int(getattr(dataset, "point_update_clamp_interval", 100)))
                if final_clamp_iter > 0 and iteration >= final_clamp_iter and iteration % clamp_interval == 0:
                    target_points = get_point_update_target_points(dataset)
                    overflow = len(gaussians.get_xyz) - target_points
                    if target_points > 0 and overflow > 0:
                        clamp_ratio = max(0.0, float(getattr(dataset, "point_update_clamp_ratio", 1.0)))
                        replace_count = int(max(1, overflow * clamp_ratio))
                        clamped = gaussians.replace_low_value_points(
                            replace_count,
                            exclude_last=0,
                            opacity_w=getattr(dataset, "point_tb_replace_opacity_w", 1.0),
                            gradient_w=getattr(dataset, "point_tb_replace_gradient_w", 0.25),
                            visibility_w=getattr(dataset, "point_tb_replace_visibility_w", 0.10),
                            use_long_term=True,
                            protect_mask=None,
                            min_keep=getattr(dataset, "point_update_min_keep", 1024),
                        )
                        print(
                            "[POINT_UPDATE_CLAMP] "
                            f"iter={iteration} "
                            f"replaced={clamped} "
                            f"target_points={target_points} "
                            f"total_points={len(gaussians.get_xyz)}",
                            flush=True,
                        )

            if point_depth_parent_attrs is not None:
                parent_count = int(point_depth_parent_attrs["xyz"].shape[0])
                children_per_anchor = max(
                    1, int(getattr(dataset, "point_depth_children_per_anchor", 1))
                )
                requested_spawn = parent_count * children_per_anchor
                replaced = 0
                if bool(getattr(dataset, "point_depth_fixed_budget", True)):
                    current_points = len(gaussians.get_xyz)
                    max_points = int(getattr(dataset, "point_depth_max_points", 120000))
                    replace_count = max(
                        requested_spawn,
                        current_points + requested_spawn - max_points,
                    )
                    replaced = gaussians.replace_low_value_points(
                        replace_count,
                        opacity_w=getattr(dataset, "point_depth_replace_opacity_w", 1.0),
                        gradient_w=getattr(dataset, "point_depth_replace_gradient_w", 0.25),
                        visibility_w=getattr(dataset, "point_depth_replace_visibility_w", 0.10),
                    )
                spawned = gaussians.spawn_from_cached_depth_anchors(
                    point_depth_parent_attrs,
                    children_per_anchor=children_per_anchor,
                    offset_scale=getattr(dataset, "point_depth_offset_scale", 0.25),
                    scale_ratio=getattr(dataset, "point_depth_scale_ratio", 0.7),
                    opacity_ratio=getattr(dataset, "point_depth_opacity_ratio", 0.8),
                    opacity_min=getattr(dataset, "point_depth_opacity_min", 0.01),
                    max_points=getattr(dataset, "point_depth_max_points", 120000),
                    surface_max_distance=getattr(dataset, "point_depth_surface_threshold", 0.12),
                )
                print(
                    "[POINT_DEPTH] "
                    f"iter={iteration} "
                    f"patches={point_depth_stats['patches']} "
                    f"anchors={point_depth_stats['anchors']} "
                    f"depth_matches={point_depth_stats['depth_matches']} "
                    f"surface_matches={point_depth_stats['surface_matches']} "
                    f"replaced={replaced} "
                    f"spawned={spawned} "
                    f"total_points={len(gaussians.get_xyz)} "
                    f"patch_error_mean={point_depth_stats['error_mean']:.6f} "
                    f"depth_error_mean={point_depth_stats['depth_error_mean']:.6f} "
                    f"surface_distance_mean={point_depth_stats['surface_distance_mean']:.6f}"
                )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                gaussians.mlp_optimizer.step()
                gaussians.mlp_optimizer.zero_grad()
                gaussians.mlp_scheduler.step()

                if part_controller is not None:
                    part_controller.after_iteration(iteration, scene, gaussians)
                    if (
                        getattr(dataset, "use_part_moe", False)
                        and iteration == int(getattr(dataset, "part_moe_start_iter", 15000))
                    ):
                        if not gaussians.part_label_enabled:
                            raise RuntimeError("[PartMoE] Part labels must be loaded before initializing experts.")
                        if getattr(dataset, "use_temporal_conditioned_part_moe", False):
                            gaussians.init_temporal_conditioned_part_moe()
                        else:
                            gaussians.init_part_moe_from_shared()
                if (
                    vggt_garment_controller is not None
                    and not bool(getattr(dataset, "vggt_strict_budget", False))
                ):
                    vggt_garment_controller.after_iteration(
                        iteration,
                        scene,
                        gaussians,
                        context={
                            "viewpoint_camera": viewpoint_cam,
                            "deformed_means3D": render_pkg["deformed_means3D"].detach(),
                            "visibility_filter": visibility_filter,
                            "image": image,
                            "gt_image": gt_image,
                            "bound_mask": bound_mask_full,
                        },
                    )

            # end time
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time += (end_time - start_time)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
def prepare_output_and_logger(args, opt=None):
    if not args.model_path:
        args.model_path = os.path.join("./output/", args.exp_name)

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    if opt is not None:
        train_cfg = {**vars(args), **vars(opt)}
        with open(os.path.join(args.model_path, "train_cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**train_cfg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, saving_iterations):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        smpl_rot = {}
        validation_configs = [{'name': 'train', 'cameras' : scene.getTrainCameras()}]
        smpl_rot['train'] = {}
        for key in scene.getTestCameras().keys():
            validation_configs.append({'name': key, 'cameras' : scene.getTestCameras()[key]})
            smpl_rot[key] = {}        
        for config in validation_configs:
            if config['name'] != 'train' and config['cameras'] and len(config['cameras']) > 0: 
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                ssims, psnrs, lpipss, img_names, full_dict, per_view_dict = [], [], [], [], {}, {}
                    
                for idx, viewpoint in enumerate(config['cameras']):
                    smpl_rot[config['name']][viewpoint.pose_id] = {}
                    render_output = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        *renderArgs,
                        return_smpl_rot=True,
                        iteration=iteration,
                    )
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    cur_psnr = psnr(image, gt_image).mean().double()
                    psnr_test += cur_psnr
                    cur_ssim = ssim(image, gt_image).mean().double()
                    ssim_test += cur_ssim
                    cur_lpips = loss_fn_vgg(image, gt_image).mean().double()
                    lpips_test += cur_lpips

                    img_names.append(viewpoint.image_name)
                    psnrs.append(cur_psnr)
                    ssims.append(cur_ssim)
                    lpipss.append(cur_lpips)

                    smpl_rot[config['name']][viewpoint.pose_id]['d_nonrigid'] = render_output['d_nonrigid']
                    smpl_rot[config['name']][viewpoint.pose_id]['transforms'] = render_output['transforms']
                    smpl_rot[config['name']][viewpoint.pose_id]['translation'] = render_output['translation']

                full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                                  "PSNR": torch.tensor(psnrs).mean().item(),
                                  "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), img_names)},
                                      "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), img_names)},
                                      "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), img_names)}})
                os.makedirs(scene.model_path + '/metrics', exist_ok=True)
                with open(scene.model_path + '/metrics/results_' + config['name'] + '_' + str(iteration) + '.json', 'w') as fp:
                    json.dump(full_dict, fp, indent=True)
                with open(scene.model_path + '/metrics/per_view' + config['name'] + '_' + str(iteration) + '.json', 'w') as fp:
                    json.dump(per_view_dict, fp, indent=True)

                l1_test /= len(config['cameras']) 
                psnr_test /= len(config['cameras'])   
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])      
                print("\n[ITER {}] Evaluating {} #{}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], len(config['cameras']), l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # Store data (serialize)
        if iteration in saving_iterations:
            save_path = os.path.join(scene.model_path, 'smpl_rot', f'iteration_{iteration}')
            os.makedirs(save_path, exist_ok=True)
            with open(save_path+"/smpl_rot.pickle", 'wb') as handle:
                pickle.dump(smpl_rot, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 15_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 15_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--mono_test", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if getattr(args, "use_part_moe", False):
        from part_label.common import enable_part_stdout_logging
        enable_part_stdout_logging(args, "train")
    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
