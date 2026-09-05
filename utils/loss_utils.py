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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from pytorch3d.ops.knn import knn_points

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_masked(network_output, gt, mask):
    # img: C,H,W
    masked_network_output = network_output[:, mask]
    masked_gt = gt[:, mask]
    return F.l1_loss(masked_network_output, masked_gt)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def l2_loss_masked(network_output, gt, mask):
    # img: C,H,W
    masked_network_output = network_output[:, mask]
    masked_gt = gt[:, mask]
    return F.mse_loss(masked_network_output, masked_gt)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def full_aiap_loss(xyz_can, xyz_obs, cov_can, cov_obs, n_neighbors=5):
    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0),
                             xyz_can.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    loss_xyz = aiap_loss(xyz_can, xyz_obs, nn_ix=nn_ix)
    loss_cov = aiap_loss(cov_can, cov_obs, nn_ix=nn_ix)

    return loss_xyz, loss_cov


def _flatten_points(x):
    if x is None:
        return None
    if x.dim() >= 2 and x.shape[0] == 1:
        return x.squeeze(0)
    return x


def _pairwise_weighted_mean(values, weight):
    denom = weight.sum().clamp_min(1e-6)
    return (values * weight).sum() / denom


def _pairwise_consistency(center, neighbors):
    diff = torch.abs(center.unsqueeze(1) - neighbors)
    if diff.dim() > 2:
        diff = diff.mean(dim=-1)
    return diff


def weighted_aiap_loss(
    xyz_can,
    xyz_obs,
    cov_can,
    cov_obs,
    affinity_feat=None,
    part_label=None,
    part_weight_matrix=None,
    part_conf=None,
    motion_obs=None,
    rotation_obs=None,
    n_neighbors=5,
    affinity_floor=0.1,
):
    xyz_can = _flatten_points(xyz_can)
    xyz_obs = _flatten_points(xyz_obs)
    cov_can = _flatten_points(cov_can)
    cov_obs = _flatten_points(cov_obs)
    affinity_feat = _flatten_points(affinity_feat)
    part_label = _flatten_points(part_label)
    part_conf = _flatten_points(part_conf)
    motion_obs = _flatten_points(motion_obs)
    rotation_obs = _flatten_points(rotation_obs)

    _, nn_ix, _ = knn_points(
        xyz_can.unsqueeze(0),
        xyz_can.unsqueeze(0),
        K=n_neighbors,
        return_sorted=True,
    )
    nn_ix = nn_ix.squeeze(0)

    xyz_can_nn = xyz_can[nn_ix]
    xyz_obs_nn = xyz_obs[nn_ix]

    dists_canonical = torch.cdist(xyz_can.unsqueeze(1), xyz_can_nn)[:, 0, 1:]
    dists_deformed = torch.cdist(xyz_obs.unsqueeze(1), xyz_obs_nn)[:, 0, 1:]
    pair_loss_xyz = torch.abs(dists_canonical - dists_deformed)

    weight = torch.ones_like(pair_loss_xyz)
    if affinity_feat is not None:
        affinity_feat = F.normalize(affinity_feat, dim=-1)
        affinity_nn = affinity_feat[nn_ix[:, 1:]]
        center_affinity = affinity_feat.unsqueeze(1).expand_as(affinity_nn)
        cosine_similarity = (center_affinity * affinity_nn).sum(dim=-1)
        # Keep every spatial edge weakly active while learned affinity
        # emphasizes motion-compatible neighbors.
        feature_weight = float(affinity_floor) + (1.0 - float(affinity_floor)) * torch.relu(cosine_similarity)
        weight = weight * feature_weight
    if part_label is not None and part_weight_matrix is not None:
        labels = part_label.long()
        if labels.dim() > 1:
            labels = labels.reshape(-1)
        neighbor_labels = labels[nn_ix[:, 1:]]
        center_labels = labels.unsqueeze(-1).expand_as(neighbor_labels)
        part_weight_matrix = part_weight_matrix.to(device=labels.device, dtype=pair_loss_xyz.dtype)
        weight = weight * part_weight_matrix[center_labels, neighbor_labels]

        if part_conf is not None:
            conf = part_conf.to(device=pair_loss_xyz.device, dtype=pair_loss_xyz.dtype).clamp(0.0, 1.0)
            conf_nn = conf[nn_ix[:, 1:]]
            pair_conf = torch.minimum(conf.unsqueeze(-1), conf_nn)
            weight = weight * (0.25 + 0.75 * pair_conf)

        # A hard gate must not leave a center with a completely inactive
        # neighborhood; fall back to its spatial/feature weights in that case.
        isolated = weight.sum(dim=-1, keepdim=True) <= 1e-8
        if bool(isolated.any()):
            fallback = feature_weight if affinity_feat is not None else torch.ones_like(weight)
            weight = torch.where(isolated, fallback, weight)

    loss_xyz = _pairwise_weighted_mean(pair_loss_xyz, weight)

    loss_cov = aiap_loss(cov_can, cov_obs, nn_ix=nn_ix)

    loss_motion = torch.zeros((), device=xyz_can.device, dtype=xyz_can.dtype)
    if motion_obs is not None:
        motion_obs_nn = motion_obs[nn_ix[:, 1:]]
        pair_motion = _pairwise_consistency(motion_obs, motion_obs_nn)
        loss_motion = _pairwise_weighted_mean(pair_motion, weight)

    loss_rotation = torch.zeros((), device=xyz_can.device, dtype=xyz_can.dtype)
    if rotation_obs is not None:
        rotation_obs_nn = rotation_obs[nn_ix[:, 1:]]
        pair_rotation = _pairwise_consistency(rotation_obs, rotation_obs_nn)
        loss_rotation = _pairwise_weighted_mean(pair_rotation, weight)

    return loss_xyz, loss_cov, loss_motion, loss_rotation

def aiap_loss(x_canonical, x_deformed, n_neighbors=5, nn_ix=None):
    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    if nn_ix is None:
        _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                                 x_canonical.unsqueeze(0),
                                 K=n_neighbors + 1,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0)

    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[nn_ix])[:,0,1:]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[nn_ix])[:,0,1:]

    loss = F.l1_loss(dists_canonical, dists_deformed)

    return loss
