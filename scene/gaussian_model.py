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
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.point_value_opacity_ema = torch.empty(0)
        self.point_value_gradient_ema = torch.empty(0)
        self.point_value_visibility_ema = torch.empty(0)
        self.optimizer = None
        self.mlp_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.device=torch.device('cuda', torch.cuda.current_device())
        # load SMPL model
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

        # load knn module
        self.knn = KNN(k=1, transpose_mode=True)
        self.seq_xyz_knn = args.seq_xyz_knn
        self.custom_knn_near = KNN(k=self.seq_xyz_knn, transpose_mode=True)

        self.motion_offset_flag = motion_offset_flag
        self.non_rigid_flag = args.non_rigid_flag
        self.nonrigid_poseconds_flag = args.nonrigid_poseconds_flag
        self.nonrigid_deltaposeconds_flag = args.nonrigid_deltaposeconds_flag
        self.nonrigid_deltaxyzconds_flag = args.nonrigid_deltaxyzconds_flag
        self.use_part_moe = getattr(args, "use_part_moe", False)
        self.use_dynomo_c = bool(getattr(args, "use_dynomo_c", False))
        self.use_tri = bool(getattr(args, "use_tri", False))
        self.use_tri_part = bool(getattr(args, "use_tri_part", False))
        self.use_tri_gate = bool(getattr(args, "use_tri_gate", False))
        self.use_tri_token = bool(getattr(args, "use_tri_token", False))
        self.use_time = bool(getattr(args, "use_time", False))
        self.use_mapo_all_dynamic = bool(getattr(args, "use_mapo_all_dynamic", False))
        self.mapo_max_partition_level = int(getattr(args, "mapo_max_partition_level", 2))
        self.mapo_partition_level1_iter = int(getattr(args, "mapo_partition_level1_iter", 5000))
        self.mapo_partition_level2_iter = int(getattr(args, "mapo_partition_level2_iter", 10000))
        self.mapo_partition_level3_iter = int(getattr(args, "mapo_partition_level3_iter", 15000))
        self.mapo_num_frames = int(getattr(args, "mapo_num_frames", 100))
        self.mapo_soft_routing = bool(getattr(args, "mapo_soft_routing", False))
        self.mapo_soft_blend_width = float(getattr(args, "mapo_soft_blend_width", 4.0))
        self.mapo_shared_trunk = bool(getattr(args, "mapo_shared_trunk", False))
        self.mapo_partial_sharing = bool(getattr(args, "mapo_partial_sharing", False))
        self.use_temporal_conditioned_part_moe = bool(
            getattr(args, "use_temporal_conditioned_part_moe", False)
        )
        self.temporal_conditioned_part_fusion_mode = str(
            getattr(args, "temporal_conditioned_part_fusion_mode", "replace")
        )
        self.temporal_conditioned_part_conf_threshold = float(
            getattr(args, "temporal_conditioned_part_conf_threshold", 0.5)
        )
        self.temporal_conditioned_part_max_mix = float(
            getattr(args, "temporal_conditioned_part_max_mix", 0.75)
        )
        self.mapo_residual_alpha = float(getattr(args, "mapo_residual_alpha", 1.0))
        self.mapo_dynamic_score_enabled = bool(getattr(args, "mapo_dynamic_score_enabled", False))
        self.mapo_dynamic_score_momentum = float(getattr(args, "mapo_dynamic_score_momentum", 0.95))
        self.mapo_dynamic_score_alpha = float(getattr(args, "mapo_dynamic_score_alpha", 0.5))
        self.use_motion_temporal_temperature = bool(getattr(args, "use_motion_temporal_temperature", False))
        self.motion_temperature_min = float(getattr(args, "motion_temperature_min", 0.50))
        self.motion_temperature_max = float(getattr(args, "motion_temperature_max", 1.50))
        self.motion_velocity_weight = float(getattr(args, "motion_velocity_weight", 0.50))
        self.motion_acceleration_weight = float(getattr(args, "motion_acceleration_weight", 0.50))
        self.smpl_motion_velocity = None
        self.smpl_motion_acceleration = None
        self.smpl_motion_velocity_by_pose = {}
        self.smpl_motion_acceleration_by_pose = {}
        self.use_point = bool(getattr(args, "use_point", False))
        self.use_point_anchor = bool(getattr(args, "use_point_anchor", False))
        self.use_point_anchor_tb = bool(getattr(args, "use_point_anchor_tb", False))
        self.use_part_point = bool(getattr(args, "use_part_point", False))
        self.use_point_update = bool(getattr(args, "use_point_update", False))
        self.use_point_depth = bool(getattr(args, "use_point_depth", False))
        self.point_grad_boost = float(getattr(args, "point_grad_boost", 2.0))
        self.point_anchor_children_per_anchor = int(getattr(args, "point_anchor_children_per_anchor", 1))
        self.point_anchor_offset_scale = float(getattr(args, "point_anchor_offset_scale", 0.35))
        self.point_anchor_scale_ratio = float(getattr(args, "point_anchor_scale_ratio", 0.7))
        self.point_anchor_opacity_ratio = float(getattr(args, "point_anchor_opacity_ratio", 0.8))
        self.point_anchor_opacity_min = float(getattr(args, "point_anchor_opacity_min", 0.01))
        self.point_anchor_max_points = int(getattr(args, "point_anchor_max_points", 120000))
        self.tri_plane_dim = int(getattr(args, "tri_plane_dim", 32))
        self.tri_plane_res = int(getattr(args, "tri_plane_res", 64))
        self.tri_plane_extent = float(getattr(args, "tri_plane_extent", 1.2))
        self.token_tri_dim = int(getattr(args, "token_tri_dim", 32))
        self.token_tri_res = int(getattr(args, "token_tri_res", 16))
        self.token_tri_extent = float(getattr(args, "token_tri_extent", 1.0))
        self.token_tri_heads = int(getattr(args, "token_tri_heads", 4))
        self.token_tri_layers = int(getattr(args, "token_tri_layers", 2))
        self.token_tri_hidden_dim = int(getattr(args, "token_tri_hidden_dim", 128))
        self.token_tri_fusion_mode = str(getattr(args, "token_tri_fusion_mode", "concat")).lower()
        self.token_tri_fusion_hidden_dim = int(getattr(args, "token_tri_fusion_hidden_dim", 128))
        self.token_tri_alpha = float(getattr(args, "token_tri_alpha", 1.0))
        self.token_tri_start_iter = int(getattr(args, "token_tri_start_iter", 10000))
        self.token_tri_warmup = int(getattr(args, "token_tri_warmup", 1000))
        self.token_tri_route_boundary_w = float(getattr(args, "token_tri_route_boundary_w", 0.0))
        self.token_tri_route_boundary_floor = float(getattr(args, "token_tri_route_boundary_floor", 0.15))
        self.token_tri_route_output_alpha = float(getattr(args, "token_tri_route_output_alpha", 0.2))
        self.token_tri_route_hard_w = float(getattr(args, "token_tri_route_hard_w", 0.0))
        self.token_tri_route_hard_boundary_mix = float(getattr(args, "token_tri_route_hard_boundary_mix", 0.65))
        self.token_tri_route_hard_motion_mix = float(getattr(args, "token_tri_route_hard_motion_mix", 0.35))
        self.token_tri_part_fusion_delta_scale = float(getattr(args, "token_tri_part_fusion_delta_scale", 0.35))
        self.token_tri_part_fusion_spatial_delta_scale = float(getattr(args, "token_tri_part_fusion_spatial_delta_scale", 0.45))
        self.token_tri_part_fusion_spatial_w = float(getattr(args, "token_tri_part_fusion_spatial_w", 0.01))
        self.token_tri_part_fusion_spatial_std_floor = float(getattr(args, "token_tri_part_fusion_spatial_std_floor", 0.08))
        self.token_tri_part_fusion_spatial_motion_mix = float(getattr(args, "token_tri_part_fusion_spatial_motion_mix", 0.5))
        self.token_tri_part_fusion_spatial_boundary_mix = float(getattr(args, "token_tri_part_fusion_spatial_boundary_mix", 0.5))
        self.token_tri_part_fusion_spatial_part_mix = float(getattr(args, "token_tri_part_fusion_spatial_part_mix", 0.5))
        self.time_scale_emb_dim = int(getattr(args, "time_scale_emb_dim", 16))
        self.time_scale_temperature = float(getattr(args, "time_scale_temperature", 1.5))
        self.tri_gate_alpha = float(getattr(args, "tri_gate_alpha", 0.2))
        self.tri_gate_init = float(getattr(args, "tri_gate_init", 0.5))
        self.tri_gate_hidden_dim = int(getattr(args, "tri_gate_hidden_dim", 128))
        self.tri_gate_mode = str(getattr(args, "tri_gate_mode", "additive")).lower()
        self.tri_gate_start_iter = int(getattr(args, "tri_gate_start_iter", 3000))
        self.tri_gate_warmup = int(getattr(args, "tri_gate_warmup", 3000))
        self.tri_part_alpha = float(getattr(args, "tri_part_alpha", 1.0))
        self.tri_part_motion_gain = float(getattr(args, "tri_part_motion_gain", 0.5))
        self.tri_part_boundary_gain = float(getattr(args, "tri_part_boundary_gain", 0.5))
        self.tri_part_hidden_dim = int(getattr(args, "tri_part_hidden_dim", 64))
        self.part_label_schema = str(getattr(args, "part_label_schema", "anatomy5"))
        self.use_part_budget = bool(getattr(args, "use_part_budget", False))
        self.part_budget_alpha = float(getattr(args, "part_budget_alpha", 1.0))
        self.part_budget_start_iter = int(getattr(args, "part_budget_start_iter", 16000))
        self.part_budget_warmup = int(getattr(args, "part_budget_warmup", 1000))
        self.part_budget_hidden_dim = int(getattr(args, "part_budget_hidden_dim", 128))
        self.part_budget_token_dim = int(getattr(args, "part_budget_token_dim", 32))
        self.part_budget_mode = str(getattr(args, "part_budget_mode", "base")).lower()
        self.part_budget_sup_w = float(getattr(args, "part_budget_sup_w", 0.02))
        self.part_budget_balance_w = float(getattr(args, "part_budget_balance_w", 0.005))
        self.part_budget_target_mix = float(getattr(args, "part_budget_target_mix", 0.6))
        self.part_budget_target_sharpness = float(getattr(args, "part_budget_target_sharpness", 2.0))
        self.part_budget_router_sharpness = float(getattr(args, "part_budget_router_sharpness", 1.0))
        self.use_part_score_route = bool(getattr(args, "use_part_score_route", False))
        self.part_score_route_hidden_dim = int(getattr(args, "part_score_route_hidden_dim", 128))
        self.part_score_route_alpha = float(getattr(args, "part_score_route_alpha", 1.0))
        self.part_score_route_gate_bias = float(getattr(args, "part_score_route_gate_bias", -2.0))
        self.part_score_route_mode = str(getattr(args, "part_score_route_mode", "boost")).lower()
        if self.tri_gate_mode not in ("additive", "concat", "scale"):
            raise ValueError("[TRI_GATE] --tri_gate_mode must be 'additive', 'concat', or 'scale'.")
        if self.use_tri_part and not self.use_tri:
            raise ValueError("[TRI_PART] --use_tri_part must be used with --use_tri.")
        if self.use_tri_gate and not self.use_tri:
            raise ValueError("[TRI_GATE] --use_tri_gate must be used with --use_tri.")
        if self.use_tri_gate and self.use_tri_part:
            raise ValueError("[TRI_GATE] --use_tri_gate and --use_tri_part are separate ablations.")
        if self.use_tri_token and self.use_tri:
            raise ValueError("[TRI_TOKEN] --use_tri_token is a separate ablation; do not combine it with --use_tri.")
        if self.use_part_budget and not self.use_part_moe:
            raise ValueError("[PART_BUDGET] --use_part_budget must be used with --use_part_moe.")
        if self.use_tri_token and self.use_part_budget:
            raise ValueError("[TRI_TOKEN] --use_tri_token is a separate ablation; do not combine it with part_budget.")
        if self.use_part_budget and (self.part_label_schema != "part_moe_leg" or int(getattr(args, "num_parts", 0)) != 7):
            raise ValueError("[PART_BUDGET] part_budget is defined on top of part_moe_leg: use --part_label_schema part_moe_leg --num_parts 7.")
        if self.use_time and (self.use_part_moe or self.use_tri or self.use_tri_token or self.use_part_budget):
            raise ValueError("[TIME] time is an original-baseline ablation; do not combine it with part_moe/tri/tri_token/part_budget.")
        if self.use_point and (self.use_part_moe or self.use_tri or self.use_tri_token or self.use_time or self.use_part_budget):
            raise ValueError("[POINT] point is an original-baseline ablation; do not combine it with part_moe/tri/tri_token/time/part_budget.")
        if self.use_point_anchor and (self.use_point or self.use_part_moe or self.use_tri or self.use_tri_token or self.use_time or self.use_part_budget):
            raise ValueError("[POINT_ANCHOR] point_anchor is an original-baseline ablation; do not combine it with point/part_moe/tri/tri_token/time/part_budget.")
        if self.use_part_point and not self.use_part_moe:
            raise ValueError("[PART_POINT] --use_part_point must be used with --use_part_moe.")
        if self.use_part_point and not self.use_point_anchor_tb:
            raise ValueError("[PART_POINT] --use_part_point requires --use_point_anchor_tb.")
        if self.use_point_anchor_tb and not self.use_part_point and (
            self.use_point or self.use_point_anchor or self.use_point_depth or self.use_part_moe or
            self.use_tri or self.use_tri_token or self.use_time or self.use_part_budget
        ):
            raise ValueError(
                "[POINT_ANCHOR_TB] point_anchor_tb is an original-baseline ablation; "
                "do not combine it with point/point_anchor/point_depth/part_moe/tri/tri_token/time/part_budget."
            )
        if self.use_point_depth and (
            self.use_point or self.use_point_anchor or self.use_point_anchor_tb or self.use_part_moe or
            self.use_tri or self.use_tri_token or self.use_time or self.use_part_budget
        ):
            raise ValueError(
                "[POINT_DEPTH] point_depth is an original-baseline ablation; "
                "do not combine it with point/point_anchor/part_moe/tri/tri_token/time/part_budget."
            )
        if self.use_dynomo_c and (
            self.use_part_moe or self.use_tri or self.use_tri_token or self.use_time or self.use_part_budget
        ):
            raise ValueError(
                "[DYNOMO_C] dynomo_c is an original-baseline ablation; "
                "do not combine it with part_moe/tri/tri_token/time/part_budget."
            )
        if self.use_tri_token and self.token_tri_fusion_mode in ("route_hard", "part_fusion", "part_fusion_spatial") and not self.use_part_moe:
            raise ValueError("[TRI_TOKEN] route_hard/part_fusion/part_fusion_spatial requires --use_part_moe.")
        if self.use_tri_token and self.use_part_moe and (self.part_label_schema != "part_moe_leg" or int(getattr(args, "num_parts", 0)) != 7):
            raise ValueError("[TRI_TOKEN] tri_token is defined on top of part_moe_leg: use --part_label_schema part_moe_leg --num_parts 7.")
        if self.use_tri_token and self.token_tri_fusion_mode not in ("concat", "residual", "route", "route_output", "route_hard", "part_fusion", "part_fusion_spatial"):
            raise ValueError("[TRI_TOKEN] token_tri_fusion_mode must be 'concat', 'residual', 'route', 'route_output', 'route_hard', 'part_fusion', or 'part_fusion_spatial'.")
        if self.use_tri:
            part_label_schema = self.part_label_schema
            if not self.use_part_moe:
                raise ValueError("[TRI] --use_tri must be used with --use_part_moe.")
            if part_label_schema != "part_moe_leg" or int(getattr(args, "num_parts", 0)) != 7:
                raise ValueError("[TRI] tri is defined on top of part_moe_leg: use --part_label_schema part_moe_leg --num_parts 7.")
            print("[TRI] enabled=True; running on top of part_moe_leg.")
            if self.use_tri_part:
                print(
                    "[TRI_PART] enabled=True; part/motion-aware residual tri feature fusion. "
                    f"alpha={self.tri_part_alpha} motion_gain={self.tri_part_motion_gain} "
                    f"boundary_gain={self.tri_part_boundary_gain} hidden={self.tri_part_hidden_dim}"
                )
            if self.use_tri_gate:
                print(
                    "[TRI_GATE] enabled=True; gated adapter tri feature fusion. "
                    f"alpha={self.tri_gate_alpha} init={self.tri_gate_init} hidden={self.tri_gate_hidden_dim} "
                    f"mode={self.tri_gate_mode} start={self.tri_gate_start_iter} warmup={self.tri_gate_warmup}"
                )
        if self.use_part_budget:
            print(
                "[PART_BUDGET] enabled=True; "
                f"mode={self.part_budget_mode} sup_w={self.part_budget_sup_w} "
                f"balance_w={self.part_budget_balance_w} target_mix={self.part_budget_target_mix} "
                f"target_sharpness={self.part_budget_target_sharpness} "
                f"router_sharpness={self.part_budget_router_sharpness}"
            )
        if self.use_tri_token:
            part_context = "on top of part_moe_leg" if self.use_part_moe else "without part_moe"
            print(
                f"[TRI_TOKEN] enabled=True; token-conditioned tri-plane {part_context}. "
                f"dim={self.token_tri_dim} res={self.token_tri_res} extent={self.token_tri_extent} "
                f"heads={self.token_tri_heads} layers={self.token_tri_layers} "
                f"hidden={self.token_tri_hidden_dim} fusion={self.token_tri_fusion_mode} "
                f"fusion_hidden={self.token_tri_fusion_hidden_dim} alpha={self.token_tri_alpha} "
                f"start={self.token_tri_start_iter} warmup={self.token_tri_warmup}"
            )
            if self.token_tri_fusion_mode == "route_output":
                print(
                    "[TRI_TOKEN] route_output alpha="
                    f"{self.token_tri_route_output_alpha}; direct output modulation is enabled."
                )
            if self.token_tri_fusion_mode == "route_hard":
                print(
                    "[TRI_TOKEN] hard-route fusion active; hard points and part blend are modulated. "
                    f"boundary_mix={self.token_tri_route_hard_boundary_mix} "
                    f"motion_mix={self.token_tri_route_hard_motion_mix} "
                    f"supervision_w={self.token_tri_route_hard_w}"
                )
            if self.token_tri_fusion_mode == "part_fusion_spatial":
                print(
                    "[TRI_TOKEN] part-fusion-spatial active; token-conditioned tri-plane controls "
                    "global/part fusion with explicit spatial, part, and motion bias. "
                    f"delta_scale={self.token_tri_part_fusion_spatial_delta_scale} "
                    f"loss_w={self.token_tri_part_fusion_spatial_w} "
                    f"std_floor={self.token_tri_part_fusion_spatial_std_floor}"
                )
        if self.use_time:
            print(
                "[TIME] enabled=True; scale-attention SeqXYZEncoder on original baseline. "
                f"scale_emb_dim={self.time_scale_emb_dim} temperature={self.time_scale_temperature}"
            )
        if self.use_point:
            print(
                "[POINT] enabled=True; error-patch guided canonical anchor densification. "
                f"grad_boost={self.point_grad_boost}"
            )
        if self.use_point_anchor:
            print(
                "[POINT_ANCHOR] enabled=True; error-guided visible canonical Gaussian spawning. "
                f"children_per_anchor={self.point_anchor_children_per_anchor} "
                f"max_points={self.point_anchor_max_points}"
            )
        if self.use_point_anchor_tb:
            print(
                "[POINT_ANCHOR_TB] enabled=True; temporal-boundary visible canonical spawning "
                "with optional delayed replacement."
            )
        if self.use_point_depth:
            print(
                "[POINT_DEPTH] enabled=True; depth/surface-guided canonical spawning "
                "with fixed-budget replacement."
            )
        self.part_moe_start_iter = getattr(args, "part_moe_start_iter", 15000)
        self.part_moe_warmup = getattr(args, "part_moe_warmup", 1000)
        self.part_moe_global_keep = getattr(args, "part_moe_global_keep", 0.1)
        self.part_moe_conf_threshold = float(getattr(args, "part_moe_conf_threshold", 0.5))
        self.part_confidence_route = bool(getattr(args, "part_confidence_route", False))
        self.part_moe_alpha = 0.0
        self.tri_gate_alpha_scale = 1.0
        self.tri_token_alpha_scale = 0.0
        self.part_budget_alpha_scale = 0.0
        self.num_parts = getattr(args, "num_parts", 5)
        self._part_label = None
        self._part_conf = None
        self.part_label_enabled = False

        if self.motion_offset_flag:
            # load pose correction module
            total_bones = self.SMPL_NEUTRAL['weights'].shape[-1]
            self.pose_decoder = BodyPoseRefiner(total_bones=total_bones, embedding_size=3*(total_bones-1), mlp_width=128, mlp_depth=2).to(self.device)

            # load lbs weight module
            self.pos_embed_fn, pos_embed_ch = get_embedder(10, 3)
            self.lweight_offset_decoder = LBSOffsetDecoder(total_bones).to(self.device)

            # non-rigid deformer
            if self.non_rigid_flag:
                non_rigid_mlp_depth = getattr(args, "non_rigid_mlp_depth", 3)
                non_rigid_mlp_width = getattr(args, "non_rigid_mlp_width", 512)
                self.non_rigid_deformer = NonrigidDeformer(pos_input_dim=pos_embed_ch,
                        D=non_rigid_mlp_depth, W=non_rigid_mlp_width,
                        use_pose_cond=self.nonrigid_poseconds_flag, use_seq_pose_cond=self.nonrigid_deltaposeconds_flag, use_seq_xyz_cond=self.nonrigid_deltaxyzconds_flag, 
                        seq_len=args.seq_len, seq_xyz_knn=self.seq_xyz_knn, time_step_num=args.time_step_num, smpl_type=smpl_type,
                        use_part_moe=self.use_part_moe, num_parts=self.num_parts,
                        part_moe_global_keep=self.part_moe_global_keep,
                        part_moe_conf_threshold=self.part_moe_conf_threshold,
                        part_confidence_route=self.part_confidence_route,
                        use_dynomo_c=self.use_dynomo_c,
                        dynomo_c_affinity_dim=getattr(args, "dynomo_c_affinity_dim", 32),
                        use_tri=self.use_tri,
                        tri_plane_dim=self.tri_plane_dim,
                        tri_plane_res=self.tri_plane_res,
                        tri_plane_extent=self.tri_plane_extent,
                        use_tri_token=self.use_tri_token,
                        token_tri_dim=self.token_tri_dim,
                        token_tri_res=self.token_tri_res,
                        token_tri_extent=self.token_tri_extent,
                        token_tri_heads=self.token_tri_heads,
                        token_tri_layers=self.token_tri_layers,
                        token_tri_hidden_dim=self.token_tri_hidden_dim,
                        token_tri_fusion_mode=self.token_tri_fusion_mode,
                        token_tri_fusion_hidden_dim=self.token_tri_fusion_hidden_dim,
                        token_tri_alpha=self.token_tri_alpha,
                        token_tri_start_iter=self.token_tri_start_iter,
                        token_tri_warmup=self.token_tri_warmup,
                        token_tri_route_boundary_w=self.token_tri_route_boundary_w,
                        token_tri_route_boundary_floor=self.token_tri_route_boundary_floor,
                        token_tri_route_output_alpha=self.token_tri_route_output_alpha,
                        token_tri_route_hard_w=self.token_tri_route_hard_w,
                        token_tri_route_hard_boundary_mix=self.token_tri_route_hard_boundary_mix,
                        token_tri_route_hard_motion_mix=self.token_tri_route_hard_motion_mix,
                        token_tri_part_fusion_delta_scale=self.token_tri_part_fusion_delta_scale,
                        token_tri_part_fusion_spatial_delta_scale=self.token_tri_part_fusion_spatial_delta_scale,
                        token_tri_part_fusion_spatial_w=self.token_tri_part_fusion_spatial_w,
                        token_tri_part_fusion_spatial_std_floor=self.token_tri_part_fusion_spatial_std_floor,
                        token_tri_part_fusion_spatial_motion_mix=self.token_tri_part_fusion_spatial_motion_mix,
                        token_tri_part_fusion_spatial_boundary_mix=self.token_tri_part_fusion_spatial_boundary_mix,
                        token_tri_part_fusion_spatial_part_mix=self.token_tri_part_fusion_spatial_part_mix,
                        use_time=self.use_time,
                        time_scale_emb_dim=self.time_scale_emb_dim,
                        time_scale_temperature=self.time_scale_temperature,
                        use_mapo_all_dynamic=self.use_mapo_all_dynamic,
                        mapo_max_partition_level=self.mapo_max_partition_level,
                        mapo_num_frames=self.mapo_num_frames,
                        mapo_soft_routing=self.mapo_soft_routing,
                        mapo_soft_blend_width=self.mapo_soft_blend_width,
                        mapo_shared_trunk=self.mapo_shared_trunk,
                        mapo_partial_sharing=self.mapo_partial_sharing,
                        use_temporal_conditioned_part_moe=self.use_temporal_conditioned_part_moe,
                        temporal_conditioned_part_fusion_mode=self.temporal_conditioned_part_fusion_mode,
                        temporal_conditioned_part_conf_threshold=self.temporal_conditioned_part_conf_threshold,
                        temporal_conditioned_part_max_mix=self.temporal_conditioned_part_max_mix,
                        mapo_residual_alpha=self.mapo_residual_alpha,
                        mapo_dynamic_score_enabled=self.mapo_dynamic_score_enabled,
                        mapo_dynamic_score_momentum=self.mapo_dynamic_score_momentum,
                        mapo_dynamic_score_alpha=self.mapo_dynamic_score_alpha,
                        use_motion_temporal_temperature=self.use_motion_temporal_temperature,
                        motion_temperature_min=self.motion_temperature_min,
                        motion_temperature_max=self.motion_temperature_max,
                        motion_velocity_weight=self.motion_velocity_weight,
                        motion_acceleration_weight=self.motion_acceleration_weight,
                        use_tri_part=self.use_tri_part,
                        use_tri_gate=self.use_tri_gate,
                        tri_gate_alpha=self.tri_gate_alpha,
                        tri_gate_init=self.tri_gate_init,
                        tri_gate_hidden_dim=self.tri_gate_hidden_dim,
                        tri_gate_mode=self.tri_gate_mode,
                        tri_part_alpha=self.tri_part_alpha,
                        tri_part_motion_gain=self.tri_part_motion_gain,
                        tri_part_boundary_gain=self.tri_part_boundary_gain,
                        tri_part_hidden_dim=self.tri_part_hidden_dim,
                        part_label_schema=self.part_label_schema,
                        use_part_budget=self.use_part_budget,
                        part_budget_alpha=self.part_budget_alpha,
                        part_budget_start_iter=self.part_budget_start_iter,
                        part_budget_warmup=self.part_budget_warmup,
                        part_budget_hidden_dim=self.part_budget_hidden_dim,
                        part_budget_token_dim=self.part_budget_token_dim,
                        part_budget_mode=self.part_budget_mode,
                        part_budget_sup_w=self.part_budget_sup_w,
                        part_budget_balance_w=self.part_budget_balance_w,
                        part_budget_target_mix=self.part_budget_target_mix,
                        part_budget_target_sharpness=self.part_budget_target_sharpness,
                        part_budget_router_sharpness=self.part_budget_router_sharpness,
                        use_part_score_route=self.use_part_score_route,
                        part_score_route_hidden_dim=self.part_score_route_hidden_dim,
                        part_score_route_alpha=self.part_score_route_alpha,
                        part_score_route_gate_bias=self.part_score_route_gate_bias,
                        part_score_route_mode=self.part_score_route_mode,
                        part_score_route_use_route_gate=getattr(args, "part_score_route_use_route_gate", 1),
                        part_score_route_use_unknown_mix=getattr(args, "part_score_route_use_unknown_mix", 1),
                        part_score_route_signal_mode=getattr(args, "part_score_route_signal_mode", "full")).to(self.device)
                self.non_rigid_deformer.part_budget_mode = self.part_budget_mode
                self.non_rigid_deformer.part_budget_sup_w = self.part_budget_sup_w
                self.non_rigid_deformer.part_budget_balance_w = self.part_budget_balance_w
                self.non_rigid_deformer.part_budget_target_mix = self.part_budget_target_mix
                self.non_rigid_deformer.part_budget_target_sharpness = self.part_budget_target_sharpness
                self.non_rigid_deformer.part_budget_router_sharpness = self.part_budget_router_sharpness

    @torch.no_grad()
    def prepare_smpl_motion_stats(self):
        """Precompute sequence-level and pose-local SMPL/LBS motion statistics."""
        if not self.use_motion_temporal_temperature or not self.smpl_params_dict:
            return
        entries = []
        for pose_id, params in self.smpl_params_dict.items():
            obs_xyz = params.get("obs_xyz") if isinstance(params, dict) else None
            if obs_xyz is not None:
                entries.append((int(pose_id), obs_xyz))
        entries.sort(key=lambda item: item[0])
        if len(entries) < 2:
            raise RuntimeError("[MOTION_TEMP] Need at least two SMPL/LBS frames for velocity statistics.")
        trajectory = torch.stack([
            value.to(device=self.device, dtype=torch.float32).reshape(-1, 3)
            for _, value in entries
        ], dim=0)
        if trajectory.shape[1] != self.canon_vertices.shape[1]:
            raise RuntimeError(
                f"[MOTION_TEMP] SMPL vertex count {trajectory.shape[1]} does not match "
                f"canonical vertices {self.canon_vertices.shape[1]}."
            )
        velocity_vec = trajectory[1:] - trajectory[:-1]
        frame_count = trajectory.shape[0]

        # Use local finite differences so routing depends on the current pose.
        frame_velocity = torch.empty((frame_count, trajectory.shape[1]), device=self.device)
        frame_velocity[0] = velocity_vec[0].norm(dim=-1)
        frame_velocity[-1] = velocity_vec[-1].norm(dim=-1)
        if frame_count > 2:
            frame_velocity[1:-1] = (
                0.5 * (trajectory[2:] - trajectory[:-2])
            ).norm(dim=-1)

        frame_acceleration = torch.zeros_like(frame_velocity)
        if frame_count >= 2:
            endpoint_acceleration = (velocity_vec[1:] - velocity_vec[:-1]).norm(dim=-1)
            frame_acceleration[0] = endpoint_acceleration[0]
            frame_acceleration[-1] = endpoint_acceleration[-1]
        if frame_count > 2:
            frame_acceleration[1:-1] = (
                trajectory[2:] - 2.0 * trajectory[1:-1] + trajectory[:-2]
            ).norm(dim=-1)

        velocity = frame_velocity.mean(dim=0)
        acceleration = frame_acceleration.mean(dim=0)

        def robust_normalize(value):
            scale = torch.quantile(value.reshape(-1), 0.95).clamp_min(1e-6)
            return (value / scale).clamp(0.0, 1.0)

        # Keep aggregate fields for compatibility; routing uses the pose-local maps.
        self.smpl_motion_velocity = robust_normalize(velocity).detach()
        self.smpl_motion_acceleration = robust_normalize(acceleration).detach()
        normalized_velocity = robust_normalize(frame_velocity).detach()
        normalized_acceleration = robust_normalize(frame_acceleration).detach()
        self.smpl_motion_velocity_by_pose = {
            pose_id: normalized_velocity[index]
            for index, (pose_id, _) in enumerate(entries)
        }
        self.smpl_motion_acceleration_by_pose = {
            pose_id: normalized_acceleration[index]
            for index, (pose_id, _) in enumerate(entries)
        }
        print(
            "[MOTION_TEMP] SMPL/LBS stats ready: "
            f"frames={len(entries)} "
            f"velocity(mean/p50/p95)={velocity.mean().item():.6f}/"
            f"{torch.quantile(velocity, 0.50).item():.6f}/"
            f"{torch.quantile(velocity, 0.95).item():.6f} "
            f"acceleration(mean/p50/p95)={acceleration.mean().item():.6f}/"
            f"{torch.quantile(acceleration, 0.50).item():.6f}/"
            f"{torch.quantile(acceleration, 0.95).item():.6f} "
            f"local_velocity_p95={torch.quantile(frame_velocity.reshape(-1), 0.95).item():.6f} "
            f"local_acceleration_p95={torch.quantile(frame_acceleration.reshape(-1), 0.95).item():.6f}"
        )

    def configure_mapo_training(self):
        if self.use_mapo_all_dynamic:
            self.non_rigid_deformer.mapo_set_training_level(0)
            print(
                "[MAPO_ALL_DYNAMIC] training configured: "
                f"max_level={self.mapo_max_partition_level} "
                f"branches={1 << self.mapo_max_partition_level} "
                f"num_frames={self.mapo_num_frames}"
            )

    def update_mapo_partition(self, iteration):
        if not self.use_mapo_all_dynamic:
            return
        target_level = 0
        if (
            self.mapo_max_partition_level >= 1
            and iteration >= self.mapo_partition_level1_iter
        ):
            target_level = 1
        if (
            self.mapo_max_partition_level >= 2
            and iteration >= self.mapo_partition_level2_iter
        ):
            target_level = 2
        if (
            self.mapo_max_partition_level >= 3
            and iteration >= self.mapo_partition_level3_iter
        ):
            target_level = 3
        activated_params = self.non_rigid_deformer.mapo_activate_level(target_level)
        if activated_params and self.mlp_optimizer is not None:
            for param in activated_params:
                self.mlp_optimizer.state.pop(param, None)
            print(
                f"[MAPO_ALL_DYNAMIC] reset optimizer state for "
                f"{len(activated_params)} activated branch parameters."
            )
                            
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.pose_decoder,
            self.lweight_offset_decoder,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.pose_decoder,
        self.lweight_offset_decoder) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def load_part_labels(self, part_label_path, part_conf_path=None):
        labels = np.load(part_label_path).astype(np.int64)
        labels = torch.from_numpy(labels).to(self.device)
        num_gaussians = self.get_xyz.shape[0]
        if labels.shape[0] != num_gaussians:
            raise RuntimeError(f"Part label number {labels.shape[0]} != Gaussian number {num_gaussians}")

        self._part_label = labels
        if part_conf_path is not None and os.path.exists(part_conf_path):
            conf = np.load(part_conf_path).astype(np.float32)
            conf = torch.from_numpy(conf).to(self.device)
            if conf.shape[0] != num_gaussians:
                raise RuntimeError(f"Part conf number {conf.shape[0]} != Gaussian number {num_gaussians}")
            self._part_conf = conf
        else:
            self._part_conf = None
        self.part_label_enabled = True

        unique, counts = torch.unique(labels, return_counts=True)
        print("[PartLabel] Loaded part labels:")
        for label, count in zip(unique.tolist(), counts.tolist()):
            print(f"  part {label}: {count}")
        print(f"[PartLabel] Total gaussians: {num_gaussians}")

    @property
    def get_part_label(self):
        return self._part_label

    @property
    def get_part_conf(self):
        return self._part_conf

    def prepare_part_moe_for_loading(self):
        if not (self.non_rigid_flag and self.use_part_moe):
            return
        self.non_rigid_deformer.init_part_moe_from_shared(num_parts=self.num_parts)
        self.non_rigid_deformer.to(self.device)

    def init_part_moe_from_shared(self):
        if not (self.non_rigid_flag and self.use_part_moe):
            return False
        created = self.non_rigid_deformer.init_part_moe_from_shared(num_parts=self.num_parts)
        self.non_rigid_deformer.to(self.device)
        if not created:
            return False

        if self.mlp_optimizer is not None:
            current_lr = None
            for group in self.mlp_optimizer.param_groups:
                if group.get("name") == "non_rigid_deformer":
                    current_lr = group["lr"]
                    break
            if current_lr is None:
                current_lr = self.mlp_optimizer.param_groups[-1]["lr"]

            for pid, expert in enumerate(self.non_rigid_deformer.part_experts):
                self.mlp_optimizer.add_param_group({
                    "params": list(expert.parameters()),
                    "lr": current_lr,
                    "initial_lr": current_lr,
                    "name": f"part_expert_{pid}",
                })
                if hasattr(self.mlp_scheduler, "base_lrs"):
                    self.mlp_scheduler.base_lrs.append(current_lr)

            self.non_rigid_deformer.freeze_shared_after_part_moe()
            print(f"[PartMoE] Added {self.num_parts} expert param groups to mlp_optimizer at lr={current_lr}.")
            print("[PartMoE] Frozen the original shared non-rigid branch; experts train from iteration after activation.")
            print("[PartMoE] mlp_optimizer param groups:")
            for idx, group in enumerate(self.mlp_optimizer.param_groups):
                print(f"  {idx}: {group.get('name', 'noname')} lr={group['lr']}")
        return True

    def init_temporal_conditioned_part_moe(self):
        if not (self.non_rigid_flag and self.use_temporal_conditioned_part_moe):
            return False
        created = self.non_rigid_deformer.init_temporal_conditioned_part_moe()
        self.non_rigid_deformer.to(self.device)
        return created

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
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1, transform=None, d_rotation=None, d_scaling=None):
        if d_rotation is not None:
            scaling = self.get_scaling + d_scaling
            # rotation = self._rotation + d_rotation

            q1 = d_rotation
            # q1[0] = 1. # [1,0,0,0] represents identity rotation
            # d_rotation = d_rotation[1:]
            q2 = self._rotation
            rotation = quaternion_multiply(q1, q2)

        else:
            scaling = self.get_scaling
            rotation = self._rotation
        return self.covariance_activation(scaling, scaling_modifier, rotation, transform)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
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
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.motion_offset_flag:
            mlp_l = [
                {'params': self.pose_decoder.parameters(), 'lr': training_args.pose_refine_lr, "name": "pose_decoder"},
                {'params': self.lweight_offset_decoder.parameters(), 'lr': training_args.lbs_offset_lr, "name": "lweight_offset_decoder"},
            ]

        if self.non_rigid_flag:
            base_lr = training_args.non_rigid_deformer_lr
            mlp_l += [{'params': self.non_rigid_deformer.parameters(), 'lr': base_lr,
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
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
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
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

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
            # if len(group["params"]) == 1:
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
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

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._prune_point_value_buffers(valid_points_mask)

    def _ensure_point_value_buffers(self):
        total = int(self.get_xyz.shape[0])
        device = self.get_xyz.device
        dtype = self.get_xyz.dtype

        def resize(buf):
            if not torch.is_tensor(buf) or buf.numel() == 0:
                return torch.zeros((total,), device=device, dtype=dtype)
            buf = buf.to(device=device, dtype=dtype).flatten()
            if buf.shape[0] == total:
                return buf
            if buf.shape[0] > total:
                return buf[:total]
            return torch.cat([buf, torch.zeros((total - buf.shape[0],), device=device, dtype=dtype)], dim=0)

        self.point_value_opacity_ema = resize(self.point_value_opacity_ema)
        self.point_value_gradient_ema = resize(self.point_value_gradient_ema)
        self.point_value_visibility_ema = resize(self.point_value_visibility_ema)

    def _append_point_value_buffers(self, count):
        count = int(count)
        if count <= 0:
            return
        self._ensure_point_value_buffers()
        device = self.get_xyz.device
        dtype = self.get_xyz.dtype
        self.point_value_opacity_ema = torch.cat(
            [self.point_value_opacity_ema, torch.zeros((count,), device=device, dtype=dtype)], dim=0
        )
        self.point_value_gradient_ema = torch.cat(
            [self.point_value_gradient_ema, torch.zeros((count,), device=device, dtype=dtype)], dim=0
        )
        self.point_value_visibility_ema = torch.cat(
            [self.point_value_visibility_ema, torch.zeros((count,), device=device, dtype=dtype)], dim=0
        )

    def _prune_point_value_buffers(self, valid_points_mask):
        if not torch.is_tensor(valid_points_mask):
            return
        if not torch.is_tensor(self.point_value_opacity_ema) or self.point_value_opacity_ema.numel() == 0:
            return
        if self.point_value_opacity_ema.shape[0] != valid_points_mask.shape[0]:
            self._ensure_point_value_buffers()
        self.point_value_opacity_ema = self.point_value_opacity_ema[valid_points_mask]
        self.point_value_gradient_ema = self.point_value_gradient_ema[valid_points_mask]
        self.point_value_visibility_ema = self.point_value_visibility_ema[valid_points_mask]

    @torch.no_grad()
    def update_point_value_ema(self, momentum=0.95):
        self._ensure_point_value_buffers()
        momentum = float(momentum)
        momentum = min(max(momentum, 0.0), 0.999)
        opacity = self.get_opacity.detach().squeeze(-1)
        gradient = (self.xyz_gradient_accum.detach() / self.denom.detach().clamp_min(1e-6)).squeeze(-1)
        visibility = self.max_radii2D.detach()

        def normalize(value):
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return value / value.max().clamp_min(1e-6)

        self.point_value_opacity_ema.mul_(momentum).add_(normalize(opacity), alpha=1.0 - momentum)
        self.point_value_gradient_ema.mul_(momentum).add_(normalize(gradient), alpha=1.0 - momentum)
        self.point_value_visibility_ema.mul_(momentum).add_(normalize(visibility), alpha=1.0 - momentum)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group["params"]) == 1
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._ensure_point_value_buffers()

    @torch.no_grad()
    def cache_point_anchor_parents(self, anchor_ids):
        if anchor_ids is None:
            return None
        anchor_ids = anchor_ids.to(device=self.get_xyz.device, dtype=torch.long)
        if anchor_ids.numel() == 0:
            return None
        anchor_ids = anchor_ids[(anchor_ids >= 0) & (anchor_ids < self.get_xyz.shape[0])]
        if anchor_ids.numel() == 0:
            return None
        return {
            "xyz": self._xyz[anchor_ids].detach().clone(),
            "features_dc": self._features_dc[anchor_ids].detach().clone(),
            "features_rest": self._features_rest[anchor_ids].detach().clone(),
            "opacity": self.get_opacity[anchor_ids].detach().clone(),
            "scaling": self.get_scaling[anchor_ids].detach().clone(),
            "rotation": self._rotation[anchor_ids].detach().clone(),
        }

    @torch.no_grad()
    def cache_point_depth_parents(self, anchor_ids, surface_xyz=None):
        attrs = self.cache_point_anchor_parents(anchor_ids)
        if attrs is None:
            return None
        if surface_xyz is not None:
            surface_xyz = surface_xyz.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype)
            if surface_xyz.shape[0] == attrs["xyz"].shape[0]:
                attrs["surface_xyz"] = surface_xyz.detach().clone()
        return attrs

    @torch.no_grad()
    def spawn_from_cached_anchors(
        self,
        parent_attrs,
        children_per_anchor=1,
        offset_scale=0.35,
        scale_ratio=0.7,
        opacity_ratio=0.8,
        opacity_min=0.01,
        max_points=120000,
    ):
        if parent_attrs is None:
            return 0
        parent_xyz = parent_attrs["xyz"]
        if parent_xyz.numel() == 0:
            return 0
        children_per_anchor = max(1, int(children_per_anchor))
        max_points = int(max_points)
        available = max_points - int(self.get_xyz.shape[0])
        if available <= 0:
            return 0
        max_parents = max(1, available // children_per_anchor)
        if parent_xyz.shape[0] > max_parents:
            keep = torch.randperm(parent_xyz.shape[0], device=parent_xyz.device)[:max_parents]
            parent_attrs = {name: value[keep] for name, value in parent_attrs.items()}
            parent_xyz = parent_attrs["xyz"]

        repeat = children_per_anchor
        parent_xyz = parent_attrs["xyz"].repeat_interleave(repeat, dim=0)
        parent_scaling = parent_attrs["scaling"].repeat_interleave(repeat, dim=0)
        parent_rotation = parent_attrs["rotation"].repeat_interleave(repeat, dim=0)
        new_features_dc = parent_attrs["features_dc"].repeat_interleave(repeat, dim=0)
        new_features_rest = parent_attrs["features_rest"].repeat_interleave(repeat, dim=0)

        samples = torch.randn_like(parent_scaling) * (parent_scaling * float(offset_scale))
        rots = build_rotation(parent_rotation)
        local_offset = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        new_xyz = parent_xyz + local_offset
        new_scaling = self.scaling_inverse_activation(
            torch.clamp(parent_scaling * float(scale_ratio), min=1e-6)
        )
        new_rotation = parent_rotation
        new_opacity = inverse_sigmoid(
            torch.clamp(
                parent_attrs["opacity"].repeat_interleave(repeat, dim=0) * float(opacity_ratio),
                min=float(opacity_min),
                max=0.99,
            )
        )

        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        spawned = int(new_xyz.shape[0])
        self.xyz_gradient_accum = torch.cat(
            [self.xyz_gradient_accum, torch.zeros((spawned, 1), device=self.get_xyz.device)],
            dim=0,
        )
        self.denom = torch.cat(
            [self.denom, torch.zeros((spawned, 1), device=self.get_xyz.device)],
            dim=0,
        )
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros((spawned,), device=self.get_xyz.device)],
            dim=0,
        )
        self._ensure_point_value_buffers()
        return spawned

    @torch.no_grad()
    def spawn_from_explicit_candidates(
        self,
        candidate_xyz,
        colors=None,
        opacity=0.04,
        scale_ratio=0.65,
        min_distance=0.004,
        max_points=120000,
    ):
        """Add precomputed canonical proposals while preserving learned parents."""
        if candidate_xyz is None or candidate_xyz.numel() == 0:
            return 0
        available = int(max_points) - int(self.get_xyz.shape[0])
        if available <= 0:
            return 0
        candidate_xyz = candidate_xyz.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype).reshape(-1, 3)
        if colors is not None:
            colors = colors.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype).reshape(-1, 3)
            colors = colors[:candidate_xyz.shape[0]]
        if candidate_xyz.shape[0] > available:
            candidate_xyz = candidate_xyz[:available]
            if colors is not None:
                colors = colors[:available]
        # Remove proposals that duplicate an existing canonical Gaussian.
        nearest_dist = torch.cdist(candidate_xyz, self.get_xyz.detach()).min(dim=1).values
        keep = nearest_dist >= float(min_distance)
        candidate_xyz = candidate_xyz[keep]
        if colors is not None:
            colors = colors[keep]
        if candidate_xyz.numel() == 0:
            return 0
        parent_ids = torch.cdist(candidate_xyz, self.get_xyz.detach()).argmin(dim=1)
        parent_scaling = self.get_scaling[parent_ids]
        new_scaling = self.scaling_inverse_activation(
            torch.clamp(parent_scaling * float(scale_ratio), min=1e-6)
        )
        new_rotation = self._rotation[parent_ids]
        new_features_dc = self._features_dc[parent_ids].clone()
        new_features_rest = self._features_rest[parent_ids].clone()
        if colors is not None and colors.shape[0] == candidate_xyz.shape[0]:
            color_sh = RGB2SH(colors.clamp(0.0, 1.0))
            new_features_dc[:, 0, 0] = color_sh[:, 0]
            new_features_dc[:, 0, 1] = color_sh[:, 1]
            new_features_dc[:, 0, 2] = color_sh[:, 2]
        new_opacity = inverse_sigmoid(
            torch.full(
                (candidate_xyz.shape[0], 1),
                float(opacity),
                device=self.get_xyz.device,
                dtype=self.get_xyz.dtype,
            ).clamp(1e-4, 0.99)
        )
        optimizable_tensors = self.cat_tensors_to_optimizer({
            "xyz": candidate_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        })
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        spawned = int(candidate_xyz.shape[0])
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((spawned, 1), device=self.get_xyz.device)], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros((spawned, 1), device=self.get_xyz.device)], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((spawned,), device=self.get_xyz.device)], dim=0)
        self._ensure_point_value_buffers()
        return spawned

    @torch.no_grad()
    def append_cloned_points(self, count, source_ids=None):
        """Append learned Gaussian copies for an isolated final budget correction."""
        count = int(count)
        if count <= 0 or self.get_xyz.shape[0] == 0:
            return 0
        total = int(self.get_xyz.shape[0])
        if source_ids is None:
            source_ids = torch.arange(total, device=self.get_xyz.device, dtype=torch.long)
        else:
            source_ids = source_ids.to(device=self.get_xyz.device, dtype=torch.long).flatten()
            source_ids = source_ids[(source_ids >= 0) & (source_ids < total)]
        if source_ids.numel() == 0:
            return 0
        source_ids = source_ids.repeat((count + source_ids.numel() - 1) // source_ids.numel())[:count]
        self._ensure_point_value_buffers()
        source_opacity_ema = self.point_value_opacity_ema[source_ids].detach().clone()
        source_gradient_ema = self.point_value_gradient_ema[source_ids].detach().clone()
        source_visibility_ema = self.point_value_visibility_ema[source_ids].detach().clone()
        d = {
            "xyz": self._xyz[source_ids].detach().clone(),
            "f_dc": self._features_dc[source_ids].detach().clone(),
            "f_rest": self._features_rest[source_ids].detach().clone(),
            "opacity": self._opacity[source_ids].detach().clone(),
            "scaling": self._scaling[source_ids].detach().clone(),
            "rotation": self._rotation[source_ids].detach().clone(),
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, self.xyz_gradient_accum[source_ids].detach().clone()], dim=0)
        self.denom = torch.cat([self.denom, self.denom[source_ids].detach().clone()], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, self.max_radii2D[source_ids].detach().clone()], dim=0)
        self.point_value_opacity_ema = torch.cat([self.point_value_opacity_ema[:total], source_opacity_ema], dim=0)
        self.point_value_gradient_ema = torch.cat([self.point_value_gradient_ema[:total], source_gradient_ema], dim=0)
        self.point_value_visibility_ema = torch.cat([self.point_value_visibility_ema[:total], source_visibility_ema], dim=0)
        return count

    @torch.no_grad()
    def spawn_from_cached_depth_anchors(
        self,
        parent_attrs,
        children_per_anchor=1,
        offset_scale=0.25,
        scale_ratio=0.7,
        opacity_ratio=0.8,
        opacity_min=0.01,
        max_points=120000,
        surface_max_distance=0.12,
    ):
        if parent_attrs is None:
            return 0
        parent_xyz = parent_attrs["xyz"]
        if parent_xyz.numel() == 0:
            return 0
        children_per_anchor = max(1, int(children_per_anchor))
        available = int(max_points) - int(self.get_xyz.shape[0])
        if available <= 0:
            return 0
        max_parents = max(1, available // children_per_anchor)
        if parent_xyz.shape[0] > max_parents:
            keep = torch.randperm(parent_xyz.shape[0], device=parent_xyz.device)[:max_parents]
            parent_attrs = {name: value[keep] for name, value in parent_attrs.items()}

        repeat = children_per_anchor
        parent_xyz = parent_attrs["xyz"].repeat_interleave(repeat, dim=0)
        parent_scaling = parent_attrs["scaling"].repeat_interleave(repeat, dim=0)
        parent_rotation = parent_attrs["rotation"].repeat_interleave(repeat, dim=0)
        new_features_dc = parent_attrs["features_dc"].repeat_interleave(repeat, dim=0)
        new_features_rest = parent_attrs["features_rest"].repeat_interleave(repeat, dim=0)

        samples = torch.randn_like(parent_scaling) * (parent_scaling * float(offset_scale))
        rots = build_rotation(parent_rotation)
        local_offset = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        new_xyz = parent_xyz + local_offset

        # Keep children in a bounded neighborhood of the canonical SMPL surface.
        if "surface_xyz" in parent_attrs:
            surface_xyz = parent_attrs["surface_xyz"].repeat_interleave(repeat, dim=0)
            delta = new_xyz - surface_xyz
            distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
            max_distance = max(float(surface_max_distance), 1e-6)
            new_xyz = surface_xyz + delta * torch.clamp(
                max_distance / distance.clamp_min(1e-6), max=1.0
            )

        new_scaling = self.scaling_inverse_activation(
            torch.clamp(parent_scaling * float(scale_ratio), min=1e-6)
        )
        new_rotation = parent_rotation
        new_opacity = inverse_sigmoid(
            torch.clamp(
                parent_attrs["opacity"].repeat_interleave(repeat, dim=0) * float(opacity_ratio),
                min=float(opacity_min),
                max=0.99,
            )
        )
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        spawned = int(new_xyz.shape[0])
        self.xyz_gradient_accum = torch.cat(
            [self.xyz_gradient_accum, torch.zeros((spawned, 1), device=self.get_xyz.device)], dim=0
        )
        self.denom = torch.cat(
            [self.denom, torch.zeros((spawned, 1), device=self.get_xyz.device)], dim=0
        )
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros((spawned,), device=self.get_xyz.device)], dim=0
        )
        self._ensure_point_value_buffers()
        return spawned

    @torch.no_grad()
    def replace_low_value_points(
        self,
        count,
        exclude_last=0,
        opacity_w=1.0,
        gradient_w=0.25,
        visibility_w=0.10,
        use_long_term=False,
        protect_mask=None,
        min_keep=1024,
    ):
        count = int(count)
        total = int(self.get_xyz.shape[0])
        eligible = total - max(0, int(exclude_last))
        count = min(count, max(0, eligible - int(min_keep)))
        if count <= 0:
            return 0

        if bool(use_long_term):
            self._ensure_point_value_buffers()
            opacity = self.point_value_opacity_ema[:eligible]
            gradient = self.point_value_gradient_ema[:eligible]
            visibility = self.point_value_visibility_ema[:eligible]
        else:
            opacity = self.get_opacity[:eligible].squeeze(-1)
            gradient = self.xyz_gradient_accum[:eligible] / self.denom[:eligible].clamp_min(1e-6)
            gradient = gradient.squeeze(-1)
            visibility = self.max_radii2D[:eligible]

        def normalize(value):
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return value / value.max().clamp_min(1e-6)

        score = (
            float(opacity_w) * normalize(opacity)
            + float(gradient_w) * normalize(gradient)
            + float(visibility_w) * normalize(visibility)
        )
        if protect_mask is not None:
            protect_mask = protect_mask.to(device=self.get_xyz.device, dtype=torch.bool).flatten()
            if protect_mask.shape[0] < eligible:
                protect_mask = torch.cat(
                    [
                        protect_mask,
                        torch.zeros((eligible - protect_mask.shape[0],), device=self.get_xyz.device, dtype=torch.bool),
                    ],
                    dim=0,
                )
            score = score.masked_fill(protect_mask[:eligible], float("inf"))
            count = min(count, int(torch.isfinite(score).sum().item()))
            if count <= 0:
                return 0
        remove_ids = torch.topk(score, k=count, largest=False).indices
        prune_mask = torch.zeros((total,), dtype=torch.bool, device=self.get_xyz.device)
        prune_mask[remove_ids] = True
        self.prune_points(prune_mask)
        return count

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
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
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    @torch.no_grad()
    def boost_guided_densification(self, guided_mask, min_average_gradient):
        """Raise selected canonical anchors to the densification threshold."""
        if guided_mask is None or guided_mask.numel() != self.get_xyz.shape[0]:
            return 0
        guided_mask = guided_mask.to(device=self.get_xyz.device, dtype=torch.bool)
        count = int(guided_mask.sum().item())
        if count == 0:
            return 0
        denom = torch.clamp(self.denom[guided_mask], min=1.0)
        target = denom * float(min_average_gradient)
        self.xyz_gradient_accum[guided_mask] = torch.maximum(
            self.xyz_gradient_accum[guided_mask],
            target,
        )
        return count

    def get_canon2Tpose_transform(self, cannon_pose_params):
        self.A2T_pose_tranform, _, _, _ = get_transform_params_torch(self.SMPL_NEUTRAL, cannon_pose_params)
        # pose dir
        vertices_num = self.canon_vertices.shape[1]
        posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
        pose_ = cannon_pose_params['poses']
        ident = torch.eye(3).cuda().float()
        batch_size = pose_.shape[0]
        rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
        
        self.canon_pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)

    def coarse_deform_c2source(self, query_pts, params, lbs_weights=None, correct_Rs=None, return_transl=False):
        bs = query_pts.shape[0]
        joints_num = self.SMPL_NEUTRAL['weights'].shape[-1]
        vertices_num = self.canon_vertices.shape[1]
        # Find nearest smpl vertex        
        _, vert_ids = self.knn(self.canon_vertices, query_pts)
        if lbs_weights is None:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)#.cuda() # [bs, points_num, joints_num]
        else:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)
            bweights = torch.log(bweights + 1e-9) + lbs_weights
            bweights = F.softmax(bweights, dim=-1)

        ### From A Pose (canonical space) To T Pose
        A2T_pose_RT = torch.matmul(bweights, self.A2T_pose_tranform.reshape(bs, joints_num, -1))
        A2T_pose_RT = torch.reshape(A2T_pose_RT, (bs, -1, 4, 4))
        query_pts = query_pts - A2T_pose_RT[..., :3, 3]
        A2T_pose_R_inv = torch.inverse(A2T_pose_RT[..., :3, :3].float())
        query_pts = torch.matmul(A2T_pose_R_inv, query_pts[..., None]).squeeze(-1)

        # transforms from A Pose (canonical space) To T Pose
        transforms = A2T_pose_R_inv
        translation = None

        canon_pose_offsets = torch.gather(self.canon_pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
        query_pts = query_pts - canon_pose_offsets

        # From mean shape to normal shape
        betas = params['shapes'].cuda()
        num_shape_coeffs = min(self.SMPL_NEUTRAL['shapedirs'].shape[-1], betas.shape[-1])
        shapedirs = self.SMPL_NEUTRAL['shapedirs'][..., :num_shape_coeffs]#.cuda()
        shapedirs = shapedirs.unsqueeze(0).expand(bs, *shapedirs.shape)
        shape_offset = torch.matmul(shapedirs, torch.reshape(betas[..., :num_shape_coeffs], (bs, 1, -1, 1))).squeeze(-1)
        shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
        query_pts = query_pts + shape_offset

        posedirs = self.SMPL_NEUTRAL['posedirs']#.cuda().float()
        ident = torch.eye(3).cuda().float()
        rot_mats = params['rot_mats']

        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root, correct_Rs)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

        tgt_pose_feature = (rot_mats[:, 1:, :, :] - ident).view([bs, -1])#.cuda()
        tgt_pose_offsets = torch.matmul(tgt_pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(bs, -1, 3)
        tgt_pose_offsets = torch.gather(tgt_pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
        query_pts = query_pts + tgt_pose_offsets

        # T Pose to target Pose (observation space)
        cnt2tgt_rigid_RT, global_R, global_Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params, rot_mats=rot_mats)
        cnt2tgt_rigid_RT = torch.matmul(bweights, cnt2tgt_rigid_RT.reshape(bs, joints_num, -1))
        cnt2tgt_rigid_RT = torch.reshape(cnt2tgt_rigid_RT, (bs, -1, 4, 4))
        smpl_tgt_pts = torch.matmul(cnt2tgt_rigid_RT[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_tgt_pts = smpl_tgt_pts + cnt2tgt_rigid_RT[..., :3, 3]
        transforms = torch.matmul(cnt2tgt_rigid_RT[..., :3, :3], transforms)

        # transform points from the smpl space to the world space
        global_R_inv = torch.inverse(global_R)
        world_pts = torch.matmul(smpl_tgt_pts, global_R_inv) + global_Th.view(bs, 1, -1)
        transforms = torch.matmul(global_R.view(bs, 1, 3, 3), transforms)

        # all transl
        if return_transl: 
            translation = -A2T_pose_RT[..., :3, 3]
            translation = torch.matmul(A2T_pose_R_inv, translation[..., None]).squeeze(-1)
            translation = translation - canon_pose_offsets + shape_offset + tgt_pose_offsets
            translation = torch.matmul(cnt2tgt_rigid_RT[..., :3, :3], translation[..., None]).squeeze(-1) + cnt2tgt_rigid_RT[..., :3, 3]
            translation = torch.matmul(translation, global_R_inv).squeeze(-1) + global_Th
        
        return world_pts, transforms, translation
