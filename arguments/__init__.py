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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.exp_name = ""
        self.smpl_type = "smplx"
        self.actor_gender = "neutral"
        self.motion_offset_flag = False

        # new experiments
        self.non_rigid_flag = True
        self.nonrigid_poseconds_flag = True
        self.nonrigid_deltaposeconds_flag = True
        self.nonrigid_deltaxyzconds_flag = True
        self.non_rigid_mlp_depth = 3
        self.non_rigid_mlp_width = 512
        self.seq_xyz_knn = 5
        self.time_step_num = 5
        self.seq_len = 8
        self.max_time_step = 6
        self.minimal_time_step = 2
        self.use_part_moe = False
        self.use_tri = False
        self.use_tri_part = False
        self.use_tri_gate = False
        self.use_tri_token = False
        self.use_time = False
        self.use_point = False
        self.use_point_anchor = False
        self.use_point_anchor_tb = False
        self.use_part_point = False
        self.use_point_update = False
        self.use_point_cloth_budget = False
        self.use_point_depth = False
        self.point_patch_size = 32
        self.point_start_iter = 800
        self.point_interval = 100
        self.point_topk_patches = 16
        self.point_anchor_radius = 20.0
        self.point_max_anchors = 512
        self.point_grad_boost = 2.0
        self.point_min_patch_coverage = 0.20
        self.point_anchor_start_iter = 800
        self.point_anchor_end_iter = 1800
        self.point_anchor_interval = 100
        self.point_anchor_patch_size = 32
        self.point_anchor_topk = 16
        self.point_anchor_min_coverage = 0.20
        self.point_anchor_radius = 20.0
        self.point_anchor_max_anchors = 512
        self.point_anchor_children_per_anchor = 1
        self.point_anchor_offset_scale = 0.35
        self.point_anchor_scale_ratio = 0.7
        self.point_anchor_opacity_ratio = 0.8
        self.point_anchor_opacity_min = 0.01
        self.point_anchor_max_points = 120000
        self.point_tb_start_iter = 800
        self.point_tb_end_iter = 1800
        self.point_tb_interval = 100
        self.point_tb_patch_size = 32
        self.point_tb_topk = 16
        self.point_tb_min_coverage = 0.20
        self.point_tb_anchor_radius = 20.0
        self.point_tb_max_anchors = 512
        self.point_tb_children_per_anchor = 1
        self.point_tb_offset_scale = 0.35
        self.point_tb_scale_ratio = 0.7
        self.point_tb_opacity_ratio = 0.8
        self.point_tb_opacity_min = 0.01
        self.point_tb_max_points = 120000
        self.point_tb_temporal_alpha = 0.5
        self.point_tb_history_momentum = 0.8
        self.point_tb_min_persistence = 0.0
        self.point_tb_boundary_beta = 0.5
        self.point_tb_boundary_kernel = 5
        self.point_tb_target_points = 0
        self.point_tb_replace_after_iter = 1500
        self.point_tb_replacement_ratio = 1.0
        self.point_tb_replace_opacity_w = 1.0
        self.point_tb_replace_gradient_w = 0.25
        self.point_tb_replace_visibility_w = 0.10
        self.point_update_edge_beta = 0.0
        self.point_update_edge_kernel = 3
        self.point_update_nonrigid_beta = 0.0
        self.point_update_ema_momentum = 0.95
        self.point_update_protect_anchors = True
        self.point_update_min_keep = 1024
        self.point_update_final_clamp_iter = 0
        self.point_update_clamp_interval = 100
        self.point_update_clamp_ratio = 1.0
        self.point_cloth_start_iter = 800
        self.point_cloth_end_iter = 1800
        self.point_cloth_interval = 100
        self.point_cloth_patch_size = 32
        self.point_cloth_topk = 16
        self.point_cloth_min_coverage = 0.20
        self.point_cloth_anchor_radius = 20.0
        self.point_cloth_max_anchors = 512
        self.point_cloth_children_per_anchor = 1
        self.point_cloth_offset_scale = 0.35
        self.point_cloth_scale_ratio = 0.7
        self.point_cloth_opacity_ratio = 0.8
        self.point_cloth_opacity_min = 0.01
        self.point_cloth_max_points = 120000
        self.point_cloth_temporal_alpha = 0.5
        self.point_cloth_history_momentum = 0.8
        self.point_cloth_min_persistence = 0.0
        self.point_cloth_boundary_beta = 1.0
        self.point_cloth_hf_beta = 0.0
        self.point_cloth_nonrigid_beta = 0.0
        self.point_cloth_boundary_kernel = 5
        self.point_cloth_protect_thresh = 0.60
        self.point_cloth_replace_after_iter = 800
        self.point_cloth_replacement_ratio = 1.0
        self.point_depth_start_iter = 800
        self.point_depth_end_iter = 1800
        self.point_depth_interval = 100
        self.point_depth_patch_size = 32
        self.point_depth_topk = 16
        self.point_depth_min_coverage = 0.20
        self.point_depth_anchor_radius = 20.0
        self.point_depth_max_anchors = 512
        self.point_depth_surface_threshold = 0.12
        self.point_depth_depth_threshold = 0.08
        self.point_depth_depth_relative = 0.04
        self.point_depth_children_per_anchor = 1
        self.point_depth_offset_scale = 0.25
        self.point_depth_scale_ratio = 0.7
        self.point_depth_opacity_ratio = 0.8
        self.point_depth_opacity_min = 0.01
        self.point_depth_max_points = 120000
        self.point_depth_fixed_budget = True
        self.point_depth_replace_opacity_w = 1.0
        self.point_depth_replace_gradient_w = 0.25
        self.point_depth_replace_visibility_w = 0.10
        self.tri_plane_dim = 32
        self.tri_plane_res = 64
        self.tri_plane_extent = 1.2
        self.token_tri_dim = 32
        self.token_tri_res = 16
        self.token_tri_extent = 1.0
        self.token_tri_heads = 4
        self.token_tri_layers = 2
        self.token_tri_hidden_dim = 128
        self.token_tri_fusion_mode = "concat"
        self.token_tri_fusion_hidden_dim = 128
        self.token_tri_alpha = 1.0
        self.token_tri_start_iter = 10000
        self.token_tri_warmup = 1000
        self.token_tri_route_boundary_w = 0.0
        self.token_tri_route_boundary_floor = 0.15
        self.token_tri_route_output_alpha = 0.2
        self.token_tri_route_hard_w = 0.0
        self.token_tri_route_hard_boundary_mix = 0.65
        self.token_tri_route_hard_motion_mix = 0.35
        self.token_tri_part_fusion_delta_scale = 0.35
        self.token_tri_part_fusion_spatial_delta_scale = 0.45
        self.token_tri_part_fusion_spatial_w = 0.01
        self.token_tri_part_fusion_spatial_std_floor = 0.08
        self.token_tri_part_fusion_spatial_motion_mix = 0.5
        self.token_tri_part_fusion_spatial_boundary_mix = 0.5
        self.token_tri_part_fusion_spatial_part_mix = 0.5
        self.time_scale_emb_dim = 16
        self.time_scale_temperature = 1.5
        self.tri_gate_alpha = 0.2
        self.tri_gate_init = 0.5
        self.tri_gate_hidden_dim = 128
        self.tri_gate_mode = "additive"
        self.tri_gate_start_iter = 3000
        self.tri_gate_warmup = 3000
        self.tri_part_alpha = 1.0
        self.tri_part_motion_gain = 0.5
        self.tri_part_boundary_gain = 0.5
        self.tri_part_hidden_dim = 64
        self.tri_part_reg_w = 0.0
        self.part_moe_start_iter = 15000
        self.part_moe_warmup = 1000
        self.part_moe_global_keep = 0.1
        self.use_part_budget = False
        self.part_budget_alpha = 1.0
        self.part_budget_start_iter = 16000
        self.part_budget_warmup = 1000
        self.part_budget_hidden_dim = 128
        self.part_budget_token_dim = 32
        self.part_budget_mode = "base"
        self.part_budget_sup_w = 0.02
        self.part_budget_balance_w = 0.005
        self.part_budget_target_mix = 0.6
        self.part_budget_target_sharpness = 2.0
        self.part_budget_router_sharpness = 1.0
        self.use_part_score_route = False
        self.part_score_route_hidden_dim = 128
        self.part_score_route_alpha = 1.0
        self.part_score_route_gate_bias = -2.0
        self.part_score_route_mode = "boost"
        self.part_score_route_use_route_gate = 1
        self.part_score_route_use_unknown_mix = 1
        self.part_score_route_signal_mode = "full"
        self.num_parts = 5
        self.part_label_schema = "anatomy5"
        self.part_max_smpl_dist = 0.08
        self.part_grouping_mode = "prior_only"
        self.part_label_path = ""
        self.smpl_vertex_seg_path = ""
        self.part_log_dir = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = True
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.pose_refine_lr = 0.00005
        self.lbs_offset_lr = 0.00005
        # non-rigid deformation 
        self.non_rigid_deformer_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 400 #500
        self.densify_until_iter = 1500 #15_000
        self.densify_grad_threshold = 0.0002
        self.mlp_lr_ratio = 0.1

        self.lpips_loss_w = 0.1
        self.l1_loss_w = 1.0
        self.ssim_loss_w = 0.1
        self.iospos_w = 1.0
        self.ioscov_w = 100.0
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
