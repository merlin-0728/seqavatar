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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, get_embedder
from knn_cuda import KNN

import torch.nn.functional as F
from nets.mlp_delta_body_pose import BodyPoseRefiner
from nets.mlp_delta_weight_lbs import LBSOffsetDecoder
from nets.mlp_delta_non_rigid import NonrigidDeformer
from utils.smpl_utils import read_pickle, SMPL_to_tensor, get_transform_params_torch, batch_rodrigues, quaternion_multiply

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, transform):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if transform is not None:
                actual_covariance = transform @ actual_covariance
                actual_covariance = actual_covariance @ transform.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, smpl_type : str, motion_offset_flag : bool, actor_gender: str, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normal = torch.empty(0) # 3DHGS: Added normal parameter
        self._point_labels = torch.empty(0, dtype=torch.long)
        self._point_vgfeat = torch.empty(0)
        self._vggt_target = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.mlp_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.device=torch.device('cuda', torch.cuda.current_device())
        
        if smpl_type == 'smpl':
            neutral_smpl_path = os.path.join('smpl_model/models', f'SMPL_{actor_gender.upper()}.pkl')
        elif smpl_type == 'smplx':
            neutral_smpl_path = os.path.join('smpl_model/models', f'SMPLX_{actor_gender.upper()}.pkl')
        self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=self.device)
        self.canon_params = None
        self.canon_vertices = None
        self.A2T_pose_tranform = None
        self.canon_pose_offsets = None
        self.smpl_params_dict = None
        self.cond_dict = None

        self.knn = KNN(k=1, transpose_mode=True)
        self.seq_xyz_knn = args.seq_xyz_knn
        self.custom_knn_near = KNN(k=self.seq_xyz_knn, transpose_mode=True)

        self.motion_offset_flag = motion_offset_flag
        self.non_rigid_flag = args.non_rigid_flag
        self.nonrigid_poseconds_flag = args.nonrigid_poseconds_flag
        self.nonrigid_deltaposeconds_flag = args.nonrigid_deltaposeconds_flag
        self.nonrigid_deltaxyzconds_flag = args.nonrigid_deltaxyzconds_flag
        self.source_path = getattr(args, "source_path", "")
        self.vggt_init_path = getattr(args, "vggt_init_path", "")
        self.vggt_init_filenames = getattr(
            args,
            "vggt_init_filenames",
            "vggt_canonical_init.pt,vggt_canonical_init_0402.pt",
        )
        self.vggt_max_points = getattr(args, "vggt_max_points", 60000)
        self.vggt_jitter_std = getattr(args, "vggt_jitter_std", 0.0)
        self.vggt_feat_dim = getattr(args, "vggt_feat_dim", 1)
        self.use_label_embedding = getattr(args, "use_label_embedding", True)
        self.label_embedding_dim = getattr(args, "label_embedding_dim", 8)
        self.use_dual_source_branch = getattr(args, "use_dual_source_branch", False)
        self.has_vggt_points = False
        self.lbs_weights = None

        if self.motion_offset_flag:
            total_bones = self.SMPL_NEUTRAL['weights'].shape[-1]
            self.pose_decoder = BodyPoseRefiner(total_bones=total_bones, embedding_size=3*(total_bones-1), mlp_width=128, mlp_depth=2).to(self.device)

            self.pos_embed_fn, pos_embed_ch = get_embedder(10, 3)
            self.lweight_offset_decoder = LBSOffsetDecoder(total_bones).to(self.device)

            if self.non_rigid_flag:
                self.non_rigid_deformer = NonrigidDeformer(pos_input_dim=pos_embed_ch,
                        use_pose_cond=self.nonrigid_poseconds_flag, use_seq_pose_cond=self.nonrigid_deltaposeconds_flag, use_seq_xyz_cond=self.nonrigid_deltaxyzconds_flag, 
                        seq_len=args.seq_len, seq_xyz_knn=self.seq_xyz_knn, time_step_num=args.time_step_num, smpl_type=smpl_type,
                        use_label_cond=int(self.use_label_embedding), label_emb_dim=self.label_embedding_dim,
                        vg_feat_dim=self.vggt_feat_dim, use_dual_source_branch=int(self.use_dual_source_branch)).to(self.device)
                            
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal, # 3DHGS
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.pose_decoder,
            self.lweight_offset_decoder,
            self._point_labels,
            self._point_vgfeat,
            self._vggt_target,
            self.has_vggt_points,
            self.lbs_weights,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) >= 20:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._normal,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.pose_decoder,
                self.lweight_offset_decoder,
                self._point_labels,
                self._point_vgfeat,
                self._vggt_target,
                self.has_vggt_points,
                self.lbs_weights,
            ) = model_args
        elif len(model_args) == 19:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._normal,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.pose_decoder,
                self.lweight_offset_decoder,
                self._point_labels,
                self._point_vgfeat,
                self._vggt_target,
                self.has_vggt_points,
            ) = model_args
            self.lbs_weights = None
        else:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._normal,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.pose_decoder,
                self.lweight_offset_decoder,
            ) = model_args
            self._point_labels = torch.zeros((self._xyz.shape[0],), dtype=torch.long, device=self._xyz.device)
            self._point_vgfeat = torch.zeros(
                (self._xyz.shape[0], self.vggt_feat_dim), dtype=self._xyz.dtype, device=self._xyz.device
            )
            self._vggt_target = self._xyz.detach().clone()
            self.has_vggt_points = False
            self.lbs_weights = None
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def point_labels(self):
        return self._point_labels

    @property
    def point_vgfeat(self):
        return self._point_vgfeat

    @property
    def vggt_mask(self):
        if self._point_labels.numel() == 0:
            return None
        return self._point_labels == 1
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_normal(self):
        # 3DHGS: Ensures normal is normalized to length 1
        return F.normalize(self._normal, p=2, dim=-1)
    
    def get_covariance(self, scaling_modifier = 1, transform=None, d_rotation=None, d_scaling=None):
        if d_rotation is not None:
            scaling = self.get_scaling + d_scaling
            q1 = d_rotation
            q2 = self._rotation
            rotation = quaternion_multiply(q1, q2)
        else:
            scaling = self.get_scaling
            rotation = self._rotation
        return self.covariance_activation(scaling, scaling_modifier, rotation, transform)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def _resolve_vggt_init_path(self):
        candidates = []
        if self.vggt_init_path:
            candidates.append(self.vggt_init_path)
        if self.source_path:
            base_dir = os.path.join(self.source_path, "OUTPUT_PT")
            for name in self.vggt_init_filenames.split(","):
                name = name.strip()
                if name:
                    candidates.append(os.path.join(base_dir, name))

        seen = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if os.path.exists(path):
                return path
        return None

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        smpl_points = torch.tensor(np.asarray(pcd.points)).float().cuda()
        smpl_colors_rgb = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        if smpl_colors_rgb.numel() == 0:
            smpl_colors_rgb = torch.ones((smpl_points.shape[0], 3), device="cuda") * 0.5
        smpl_colors_sh = RGB2SH(smpl_colors_rgb)

        smpl_vertex_weights = self.SMPL_NEUTRAL["weights"].float()
        if smpl_vertex_weights.shape[0] != smpl_points.shape[0]:
            _, smpl_vert_ids = self.knn(self.canon_vertices, smpl_points.unsqueeze(0))
            smpl_weights = smpl_vertex_weights[smpl_vert_ids].view(smpl_points.shape[0], -1)
        else:
            smpl_weights = smpl_vertex_weights.clone()

        labels_smpl = torch.zeros((smpl_points.shape[0],), dtype=torch.long, device="cuda")
        vg_feat_smpl = torch.zeros((smpl_points.shape[0], self.vggt_feat_dim), dtype=torch.float, device="cuda")

        fused_point_cloud = smpl_points
        fused_weights = smpl_weights
        fused_color = smpl_colors_sh
        fused_labels = labels_smpl
        fused_vgfeat = vg_feat_smpl
        self.has_vggt_points = False

        vggt_init_path = self._resolve_vggt_init_path()
        if vggt_init_path:
            print(f"\n🌟 [SeqAvatar] 加载 VGGT 初始化并与 SMPL 联合初始化: {vggt_init_path}")
            vggt_data = torch.load(vggt_init_path, map_location="cpu")

            if "xyz_vggt" in vggt_data:
                raw_vggt_points = vggt_data["xyz_vggt"].float()
                raw_vggt_weights = vggt_data.get("weights_vggt", None)
            else:
                raw_vggt_points = vggt_data.get("xyz", torch.empty(0, 3)).float()
                raw_vggt_weights = vggt_data.get("weights", None)

            if raw_vggt_points.numel() > 0:
                raw_vggt_points = raw_vggt_points.cuda()
                if raw_vggt_weights is not None:
                    raw_vggt_weights = raw_vggt_weights.float().cuda()

                # Basic outlier removal in canonical space
                distances = torch.norm(raw_vggt_points, dim=-1)
                valid_mask = distances < 1.2
                vggt_points = raw_vggt_points[valid_mask]
                if raw_vggt_weights is not None:
                    vggt_weights = raw_vggt_weights[valid_mask]
                else:
                    _, vggt_vert_ids = self.knn(self.canon_vertices, vggt_points.unsqueeze(0))
                    vggt_weights = smpl_vertex_weights[vggt_vert_ids].view(vggt_points.shape[0], -1)

                if self.vggt_jitter_std > 0:
                    vggt_points = vggt_points + torch.randn_like(vggt_points) * self.vggt_jitter_std

                if "vggt_support" in vggt_data and vggt_data["vggt_support"].shape[0] == raw_vggt_points.shape[0]:
                    support = vggt_data["vggt_support"].float().cuda()[valid_mask]
                    if support.ndim > 1:
                        support = support.squeeze(-1)
                    support = (support - support.min()) / (support.max() - support.min() + 1e-8)
                    vggt_feat = support[:, None]
                    if self.vggt_feat_dim > 1:
                        vggt_feat = vggt_feat.repeat(1, self.vggt_feat_dim)
                elif "vg_feat_vggt" in vggt_data:
                    vggt_feat = vggt_data["vg_feat_vggt"].float().cuda()
                    vggt_feat = vggt_feat[valid_mask]
                elif "vg_feat" in vggt_data and vggt_data["vg_feat"].shape[0] == raw_vggt_points.shape[0]:
                    vggt_feat = vggt_data["vg_feat"].float().cuda()[valid_mask]
                else:
                    vggt_feat = torch.ones((vggt_points.shape[0], self.vggt_feat_dim), dtype=torch.float, device="cuda")

                if vggt_feat.ndim == 1:
                    vggt_feat = vggt_feat[:, None]
                if vggt_feat.shape[-1] != self.vggt_feat_dim:
                    if vggt_feat.shape[-1] > self.vggt_feat_dim:
                        vggt_feat = vggt_feat[:, : self.vggt_feat_dim]
                    else:
                        pad = torch.zeros(
                            (vggt_feat.shape[0], self.vggt_feat_dim - vggt_feat.shape[-1]),
                            device=vggt_feat.device,
                            dtype=vggt_feat.dtype,
                        )
                        vggt_feat = torch.cat([vggt_feat, pad], dim=-1)

                vggt_mean_rgb = smpl_colors_rgb.mean(dim=0, keepdim=True)
                vggt_colors_rgb = vggt_mean_rgb.expand(vggt_points.shape[0], -1).contiguous()
                vggt_colors_sh = RGB2SH(vggt_colors_rgb)
                labels_vggt = torch.ones((vggt_points.shape[0],), dtype=torch.long, device="cuda")

                fused_point_cloud = torch.cat([smpl_points, vggt_points], dim=0)
                fused_weights = torch.cat([smpl_weights, vggt_weights], dim=0)
                fused_color = torch.cat([smpl_colors_sh, vggt_colors_sh], dim=0)
                fused_labels = torch.cat([labels_smpl, labels_vggt], dim=0)
                fused_vgfeat = torch.cat([vg_feat_smpl, vggt_feat], dim=0)

                if fused_point_cloud.shape[0] > self.vggt_max_points:
                    print(
                        f"⚠️ 联合点云过大 ({fused_point_cloud.shape[0]}), 统一降采样到 {self.vggt_max_points}."
                    )
                    smpl_idx = torch.where(fused_labels == 0)[0]
                    vggt_idx = torch.where(fused_labels == 1)[0]
                    if smpl_idx.numel() >= self.vggt_max_points:
                        keep_smpl = smpl_idx[
                            torch.randperm(smpl_idx.numel(), device=fused_point_cloud.device)[: self.vggt_max_points]
                        ]
                        keep = keep_smpl
                    else:
                        remain = self.vggt_max_points - smpl_idx.numel()
                        if vggt_idx.numel() > remain:
                            keep_vggt = vggt_idx[
                                torch.randperm(vggt_idx.numel(), device=fused_point_cloud.device)[:remain]
                            ]
                        else:
                            keep_vggt = vggt_idx
                        keep = torch.cat([smpl_idx, keep_vggt], dim=0)
                        keep = keep[torch.randperm(keep.numel(), device=keep.device)]
                    fused_point_cloud = fused_point_cloud[keep]
                    fused_weights = fused_weights[keep]
                    fused_color = fused_color[keep]
                    fused_labels = fused_labels[keep]
                    fused_vgfeat = fused_vgfeat[keep]

                self.has_vggt_points = bool((fused_labels == 1).any().item())
                print(
                    f"✅ 联合初始化完成: total={fused_point_cloud.shape[0]}, "
                    f"smpl={(fused_labels == 0).sum().item()}, vggt={(fused_labels == 1).sum().item()}"
                )
            else:
                print("⚠️ VGGT 文件存在但无有效点，退回 SMPL 初始化。")
        else:
            print("\n⚠️ 未找到 VGGT 先验，使用 SMPL 初始化。")

        self.lbs_weights = fused_weights
        self._point_labels = fused_labels
        self._point_vgfeat = fused_vgfeat
        self._vggt_target = fused_point_cloud.detach().clone()

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # ========================================================
        # 🛡️ 终极防 NaN / Inf 补丁
        # ========================================================
        dist2 = torch.clamp(distCUDA2(fused_point_cloud), min=1e-7, max=1.0)
        scales = torch.clamp(torch.log(torch.sqrt(dist2)), min=-10.0, max=-4.0)[...,None].repeat(1, 3) # 放宽高斯球体积的限制
        scales = torch.nan_to_num(scales, nan=-5.0, posinf=-4.0, neginf=-10.0)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 3DHGS: Initialize opacity to (N, 2)
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda"))
        
        # 3DHGS: Initialize normal to (N, 3), pointing to z-axis by default
        normals = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
        normals[:, 2] = 1.0 

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._normal = nn.Parameter(normals.requires_grad_(True)) # 3DHGS
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        mlp_l = []
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # 3DHGS: Add normal learning rate (fallback to rotation_lr if not provided in args)
            {'params': [self._normal], 'lr': getattr(training_args, 'normal_lr', training_args.rotation_lr), "name": "normal"} 
        ]
        if self.motion_offset_flag:
            mlp_l = [
                {'params': self.pose_decoder.parameters(), 'lr': training_args.pose_refine_lr, "name": "pose_decoder"},
                {'params': self.lweight_offset_decoder.parameters(), 'lr': training_args.lbs_offset_lr, "name": "lweight_offset_decoder"},
            ]

        if self.non_rigid_flag:
            mlp_l += [{'params': self.non_rigid_deformer.parameters(), 'lr': training_args.non_rigid_deformer_lr,
                 "name": "non_rigid_deformer"},]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.mlp_optimizer = torch.optim.Adam(params=mlp_l, lr=0.001, eps=1e-15)
        gamma = training_args.mlp_lr_ratio ** (1. / training_args.iterations)
        self.mlp_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.mlp_optimizer, gamma=gamma)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        # 3DHGS: Two opacities
        l.append('opacity_0')
        l.append('opacity_1')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # 3DHGS: Save actual normal instead of zeros
        normals = self.get_normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        # 3DHGS: Read two opacities if available, else fallback to duplicate the single opacity
        props = [p.name for p in plydata.elements[0].properties]
        if "opacity_0" in props:
            opacities = np.stack((np.asarray(plydata.elements[0]["opacity_0"]),
                                  np.asarray(plydata.elements[0]["opacity_1"])), axis=1)
        else:
            opacities_1 = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            opacities = np.concatenate([opacities_1, opacities_1], axis=1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 3DHGS: Load Normals
        normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])), axis=1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normals, dtype=torch.float, device="cuda").requires_grad_(True))
        self._point_labels = torch.zeros((self._xyz.shape[0],), dtype=torch.long, device="cuda")
        self._point_vgfeat = torch.zeros((self._xyz.shape[0], self.vggt_feat_dim), dtype=torch.float, device="cuda")
        self._vggt_target = self._xyz.detach().clone()
        self.has_vggt_points = False
        self.lbs_weights = None

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 3DHGS: Include 'normal' in pruning
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'normal']:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"] # 3DHGS

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self._point_labels.numel() == valid_points_mask.shape[0]:
            self._point_labels = self._point_labels[valid_points_mask]
        if self._point_vgfeat.ndim == 2 and self._point_vgfeat.shape[0] == valid_points_mask.shape[0]:
            self._point_vgfeat = self._point_vgfeat[valid_points_mask]
        if self._vggt_target.numel() > 0 and self._vggt_target.shape[0] == valid_points_mask.shape[0]:
            self._vggt_target = self._vggt_target[valid_points_mask]
        if self.lbs_weights is not None and self.lbs_weights.shape[0] == valid_points_mask.shape[0]:
            self.lbs_weights = self.lbs_weights[valid_points_mask]
        if self._point_labels.numel() > 0:
            self.has_vggt_points = bool((self._point_labels == 1).any().item())

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 3DHGS: Include 'normal' in concatenate
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'normal']:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_normal,
        new_point_labels=None,
        new_point_vgfeat=None,
        new_vggt_target=None,
        new_lbs_weights=None,
    ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "normal" : new_normal} # 3DHGS

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"] # 3DHGS

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_point_labels is not None and self._point_labels.numel() > 0:
            self._point_labels = torch.cat((self._point_labels, new_point_labels), dim=0)
        if new_point_vgfeat is not None and self._point_vgfeat.ndim == 2:
            self._point_vgfeat = torch.cat((self._point_vgfeat, new_point_vgfeat), dim=0)
        if new_vggt_target is not None and self._vggt_target.numel() > 0:
            self._vggt_target = torch.cat((self._vggt_target, new_vggt_target), dim=0)
        if new_lbs_weights is not None and self.lbs_weights is not None:
            self.lbs_weights = torch.cat((self.lbs_weights, new_lbs_weights), dim=0)
        if self._point_labels.numel() > 0:
            self.has_vggt_points = bool((self._point_labels == 1).any().item())

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_normal = self._normal[selected_pts_mask].repeat(N,1) # 3DHGS
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_point_labels = self._point_labels[selected_pts_mask].repeat(N) if self._point_labels.numel() > 0 else None
        new_point_vgfeat = self._point_vgfeat[selected_pts_mask].repeat(N, 1) if self._point_vgfeat.ndim == 2 else None
        new_vggt_target = self._vggt_target[selected_pts_mask].repeat(N, 1) if self._vggt_target.numel() > 0 else None
        new_lbs_weights = self.lbs_weights[selected_pts_mask].repeat(N, 1) if self.lbs_weights is not None else None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_normal,
            new_point_labels=new_point_labels,
            new_point_vgfeat=new_point_vgfeat,
            new_vggt_target=new_vggt_target,
            new_lbs_weights=new_lbs_weights,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask] # 3DHGS
        new_point_labels = self._point_labels[selected_pts_mask] if self._point_labels.numel() > 0 else None
        new_point_vgfeat = self._point_vgfeat[selected_pts_mask] if self._point_vgfeat.ndim == 2 else None
        new_vggt_target = self._vggt_target[selected_pts_mask] if self._vggt_target.numel() > 0 else None
        new_lbs_weights = self.lbs_weights[selected_pts_mask] if self.lbs_weights is not None else None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_normal,
            new_point_labels=new_point_labels,
            new_point_vgfeat=new_point_vgfeat,
            new_vggt_target=new_vggt_target,
            new_lbs_weights=new_lbs_weights,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 3DHGS: get_opacity is now (N, 2), taking the maximum opacity across the two channels to determine pruning
        prune_mask = (self.get_opacity.max(dim=1).values < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def get_canon2Tpose_transform(self, cannon_pose_params):
        self.A2T_pose_tranform, _, _, _ = get_transform_params_torch(self.SMPL_NEUTRAL, cannon_pose_params)
        vertices_num = self.canon_vertices.shape[1]
        posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
        pose_ = cannon_pose_params['poses']
        ident = torch.eye(3).cuda().float()
        batch_size = pose_.shape[0]
        rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        
        self.canon_pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)

    def coarse_deform_c2source(self, query_pts, params, lbs_weights=None, correct_Rs=None, return_transl=False):
        bs = query_pts.shape[0]
        joints_num = self.SMPL_NEUTRAL['weights'].shape[-1]
        vertices_num = self.canon_vertices.shape[1]
        
        _, vert_ids = self.knn(self.canon_vertices, query_pts)
        if self.lbs_weights is not None and self.lbs_weights.shape[0] == query_pts.shape[1]:
            base_bweights = self.lbs_weights.unsqueeze(0).expand(bs, -1, -1)
        else:
            base_bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)
        if lbs_weights is None:
            bweights = base_bweights
        else:
            bweights = torch.log(base_bweights + 1e-9) + lbs_weights
            bweights = F.softmax(bweights, dim=-1)

        A2T_pose_RT = torch.matmul(bweights, self.A2T_pose_tranform.reshape(bs, joints_num, -1))
        A2T_pose_RT = torch.reshape(A2T_pose_RT, (bs, -1, 4, 4))
        query_pts = query_pts - A2T_pose_RT[..., :3, 3]
        A2T_pose_R_inv = torch.inverse(A2T_pose_RT[..., :3, :3].float())
        query_pts = torch.matmul(A2T_pose_R_inv, query_pts[..., None]).squeeze(-1)

        transforms = A2T_pose_R_inv
        translation = None

        canon_pose_offsets = torch.gather(self.canon_pose_offsets, 1, vert_ids.expand(-1, -1, 3)) 
        query_pts = query_pts - canon_pose_offsets

        shapedirs = self.SMPL_NEUTRAL['shapedirs'][..., :params['shapes'].shape[-1]]
        shapedirs = shapedirs.unsqueeze(0).expand(bs, *shapedirs.shape)
        shape_offset = torch.matmul(shapedirs, torch.reshape(params['shapes'].cuda(), (bs, 1, -1, 1))).squeeze(-1)
        shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) 
        query_pts = query_pts + shape_offset

        posedirs = self.SMPL_NEUTRAL['posedirs']
        ident = torch.eye(3).cuda().float()
        rot_mats = params['rot_mats']

        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root, correct_Rs)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

        tgt_pose_feature = (rot_mats[:, 1:, :, :] - ident).view([bs, -1])
        tgt_pose_offsets = torch.matmul(tgt_pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(bs, -1, 3)
        tgt_pose_offsets = torch.gather(tgt_pose_offsets, 1, vert_ids.expand(-1, -1, 3)) 
        query_pts = query_pts + tgt_pose_offsets

        cnt2tgt_rigid_RT, global_R, global_Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params, rot_mats=rot_mats)
        cnt2tgt_rigid_RT = torch.matmul(bweights, cnt2tgt_rigid_RT.reshape(bs, joints_num, -1))
        cnt2tgt_rigid_RT = torch.reshape(cnt2tgt_rigid_RT, (bs, -1, 4, 4))
        smpl_tgt_pts = torch.matmul(cnt2tgt_rigid_RT[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_tgt_pts = smpl_tgt_pts + cnt2tgt_rigid_RT[..., :3, 3]
        transforms = torch.matmul(cnt2tgt_rigid_RT[..., :3, :3], transforms)

        global_R_inv = torch.inverse(global_R)
        world_pts = torch.matmul(smpl_tgt_pts, global_R_inv) + global_Th.view(bs, 1, -1)
        transforms = torch.matmul(global_R.view(bs, 1, 3, 3), transforms)

        if return_transl: 
            translation = -A2T_pose_RT[..., :3, 3]
            translation = torch.matmul(A2T_pose_R_inv, translation[..., None]).squeeze(-1)
            translation = translation - canon_pose_offsets + shape_offset + tgt_pose_offsets
            translation = torch.matmul(cnt2tgt_rigid_RT[..., :3, :3], translation[..., None]).squeeze(-1) + cnt2tgt_rigid_RT[..., :3, 3]
            translation = torch.matmul(translation, global_R_inv).squeeze(-1) + global_Th
        
        return world_pts, transforms, translation
