# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, override_color = None, return_smpl_rot=False,
           transforms=None, translation=None, d_nonrigid=None,
           detach_geometry=False):
    """
    Render the scene with depth map output.
    """

    # Create tensor for screen-space gradients
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                         requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz[None]
    pose_id = viewpoint_camera.pose_id
    smpl_params = pc.smpl_params_dict[pose_id]

    if not pc.motion_offset_flag:
        means3D, transforms, _ = pc.coarse_deform_c2source(means3D, smpl_params)
    else:
        dst_posevec = smpl_params['poses'][:, 3:]
        pose_out = pc.pose_decoder(dst_posevec)
        correct_Rs = pose_out['Rs']

        pos_embd = pc.pos_embed_fn(means3D)
        lbs_weights = pc.lweight_offset_decoder(pos_embd.detach())
        lbs_weights = lbs_weights.permute(0,2,1)

        if pc.non_rigid_flag:
            if d_nonrigid is None:
                pose_conds = pc.cond_dict[pose_id]['pose_conds']
                seq_pose_conds = pc.cond_dict[pose_id]['seq_pose_conds']
                seq_xyz_conds = pc.cond_dict[pose_id]['seq_xyz_conds']

                _, vert_ids = pc.custom_knn_near(pc.canon_vertices, means3D)
                query_pts_delta_conds = seq_xyz_conds[vert_ids, :,:,:].permute(0,1,3,2,4,5).contiguous()
                query_pts_acc_conds = None
                if getattr(pc, "use_acc_cond", False) or getattr(pc, "motion_token_use_acc", False):
                    seq_acc_conds = pc.cond_dict[pose_id].get('seq_acc_conds', None)
                    if seq_acc_conds is None:
                        raise RuntimeError("acc condition is enabled but seq_acc_conds is missing from cond_dict.")
                    query_pts_acc_conds = seq_acc_conds[vert_ids, :,:,:].permute(0,1,3,2,4,5).contiguous()
                query_pts_flow_conds = None
                if getattr(pc, "use_flow_cond", False):
                    seq_flow_conds = pc.cond_dict[pose_id].get('seq_flow_conds', None)
                    if seq_flow_conds is None:
                        raise RuntimeError("flow condition is enabled but seq_flow_conds is missing from cond_dict.")
                    knn_flow = seq_flow_conds[vert_ids, :]
                    if getattr(pc, "flow_view_token", False):
                        query_pts_flow_conds = knn_flow.mean(dim=2).contiguous()
                        train_centers = pc.cond_dict[pose_id].get('flow_train_camera_centers', None)
                        if train_centers is None:
                            raise RuntimeError("flow_view_token is enabled but flow_train_camera_centers is missing.")
                        point_xyz = means3D.detach()
                        query_pts_flow_current_dirs = torch.nn.functional.normalize(
                            viewpoint_camera.camera_center.view(1, 1, 3) - point_xyz,
                            dim=-1,
                            eps=1e-6,
                        )
                        query_pts_flow_train_dirs = torch.nn.functional.normalize(
                            train_centers.view(1, 1, train_centers.shape[0], 3) - point_xyz.unsqueeze(2),
                            dim=-1,
                            eps=1e-6,
                        )
                    else:
                        query_pts_flow_current_dirs = None
                        query_pts_flow_train_dirs = None
                        flow_knn_agg = str(getattr(pc, "flow_knn_agg", "mean")).lower()
                        if flow_knn_agg == "mean":
                            query_pts_flow_conds = knn_flow.mean(dim=2).contiguous()
                        elif flow_knn_agg == "mean_max":
                            query_pts_flow_conds = torch.cat(
                                [knn_flow.mean(dim=2), knn_flow.max(dim=2).values],
                                dim=-1,
                            ).contiguous()
                        else:
                            raise RuntimeError(f"Unsupported flow_knn_agg: {flow_knn_agg}")
                else:
                    query_pts_flow_current_dirs = None
                    query_pts_flow_train_dirs = None

                part_label = None
                part_enabled = False
                if getattr(pc, "use_part_moe", False):
                    part_enabled = pc.part_label_enabled
                    part_label = pc.get_part_label if part_enabled else None
                elif getattr(pc, "motion_token_use_part", False):
                    flat_vert_ids = vert_ids.long().reshape(-1)
                    part_probs = pc.SMPL_NEUTRAL['weights'].index_select(0, flat_vert_ids)
                    part_probs = part_probs.reshape(*vert_ids.shape, -1).mean(dim=2)
                    part_label = part_probs.argmax(dim=-1)
                    part_enabled = True

                d_xyz, d_rotation, d_scaling = pc.non_rigid_deformer(
                    pos_embd,
                    pose_conds,
                    seq_pose_conds,
                    query_pts_delta_conds,
                    seq_acc_conds=query_pts_acc_conds,
                    seq_flow_conds=query_pts_flow_conds,
                    seq_flow_current_dirs=query_pts_flow_current_dirs,
                    seq_flow_train_dirs=query_pts_flow_train_dirs,
                    part_label=part_label,
                    part_enabled=part_enabled,
                    part_moe_alpha=getattr(pc, "part_moe_alpha", 0.0),
                    part_moe_global_keep=getattr(pc, "part_moe_global_keep", 0.1),
                )
                d_nonrigid = (d_xyz, d_rotation, d_scaling)
            else:
                d_xyz, d_rotation, d_scaling = d_nonrigid
            
            means3D = means3D + d_xyz

        if transforms is None:
            means3D, transforms, translation = pc.coarse_deform_c2source(
                means3D, smpl_params, lbs_weights=lbs_weights, correct_Rs=correct_Rs, return_transl=return_smpl_rot)
        else:
            means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation

    means3D = means3D.squeeze()
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None

    if pc.non_rigid_flag:
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier, transforms.squeeze(),
                                              d_rotation=d_rotation[0], d_scaling=d_scaling[0])
        else:
            scales = pc.get_scaling + d_scaling[0]
            rotations = pc.get_rotation + d_rotation[0]
    else:
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier, transforms.squeeze())
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1,2).view(-1,3,(pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0],1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb+0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if detach_geometry:
        means3D = means3D.detach()
        means2D = means2D.detach()
        opacity = opacity.detach()
        if scales is not None:
            scales = scales.detach()
        if rotations is not None:
            rotations = rotations.detach()
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp.detach()

    # ------------------ rasterize RGB + depth ------------------
    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    return {
        "render": rendered_image,
        "depth": depth,
        "render_alpha": alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "d_nonrigid": d_nonrigid,
        "transforms": transforms,
        "translation": translation,
        "deformed_means3D": means3D,
        "deformed_cov3D": cov3D_precomp
    }
