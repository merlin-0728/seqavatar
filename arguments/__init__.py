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
        self.motion_cond_time_step_num = 0
        self.seq_len = 8
        self.max_time_step = 6
        self.minimal_time_step = 2
        self.use_msti = False
        self.msti_mode = "none"
        self.msti_mid_type = "real"
        self.use_amc_pair = False
        self.amc_pair_mode = "baseline_full"
        self.use_amc_causal = False
        self.amc_causal_mode = "gated_residual"
        self.amc_causal_window = 3
        self.amc_motion_gate_alpha = 1.0
        self.amc_motion_gate_temp = 0.5
        self.use_tdp = False
        self.tdp_mode = "keep_base"
        self.use_tdp_semantic_encoder = False
        self.tdp_semantic_mode = "gated_residual"
        self.tdp_gate_init_bias = -4.0
        self.tdp_debug_stats = False
        self.tdp_debug_interval = 1000
        self.use_dif = False
        self.dif_mode = "peak"
        self.dif_sigma_init = -7.0
        self.dif_eps = 1e-6
        self.dif_sigma_min = 1e-4
        self.dif_sigma_max = 0.05
        self.dif_residual_beta = 1.0
        self.dif_residual_warmup = 0
        self.dif_sigma_prior = 0.02
        self.dif_sigma_prior_w = 0.0
        self.dif_uncert_loss_w = 0.0
        self.dif_uncert_s_min = -6.0
        self.dif_uncert_s_max = 3.0
        self.dif_debug_interval = 1000
        self.fix_stms = False
        self.use_acc_cond = False
        self.seq_acc_cond_dim = 64
        self.use_flow_cond = False
        self.flow_cond_mode = "flow"
        self.flow_cond_dim = 32
        self.flow_feature_dim = 3
        self.flow_image_scale = 0.25
        self.flow_mag_scale = 1.0
        self.flow_feature_mode = "mean_max_conf"
        self.flow_knn_agg = "mean"
        self.flow_adapter_mode = "concat"
        self.flow_gate_alpha = 0.0
        self.flow_gate_temp = 0.5
        self.flow_view_token = False
        self.flow_view_attn_dim = 32
        self.use_motion_token = False
        self.motion_token_mode = "none"
        self.motion_token_num = 32
        self.motion_token_dim = 64
        self.motion_token_part_dim = 16
        self.motion_token_acc_dim = 64
        self.motion_token_debug_stats = False
        self.motion_token_debug_interval = 1000
        self.use_part_moe = False
        self.part_moe_start_iter = 15000
        self.part_moe_warmup = 1000
        self.part_moe_global_keep = 0.1
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
        resolve_motion_condition_args(g)
        return g

def resolve_motion_condition_args(args):
    time_step_num = int(getattr(args, "time_step_num", 0))
    if time_step_num <= 0:
        raise ValueError(f"time_step_num must be positive, got {time_step_num}")

    msti_mode = getattr(args, "msti_mode", "none") or "none"
    msti_mode = str(msti_mode).lower()
    valid_modes = {"none", "lite", "full"}
    if msti_mode not in valid_modes:
        raise ValueError(f"msti_mode must be one of {sorted(valid_modes)}, got {msti_mode}")

    use_msti = bool(getattr(args, "use_msti", False)) or msti_mode != "none"
    if not use_msti:
        msti_mode = "none"

    use_amc_pair = bool(getattr(args, "use_amc_pair", False))
    amc_pair_mode = getattr(args, "amc_pair_mode", "baseline_full") or "baseline_full"
    amc_pair_mode = str(amc_pair_mode).lower()
    valid_amc_pair_modes = {"baseline_full"}
    if use_amc_pair and amc_pair_mode not in valid_amc_pair_modes:
        raise ValueError(
            f"amc_pair_mode must be one of {sorted(valid_amc_pair_modes)}, got {amc_pair_mode}"
        )

    use_amc_causal = bool(getattr(args, "use_amc_causal", False))
    amc_causal_mode = getattr(args, "amc_causal_mode", "gated_residual") or "gated_residual"
    amc_causal_mode = str(amc_causal_mode).lower()
    valid_amc_causal_modes = {"gated_residual"}
    if use_amc_causal and amc_causal_mode not in valid_amc_causal_modes:
        raise ValueError(
            f"amc_causal_mode must be one of {sorted(valid_amc_causal_modes)}, got {amc_causal_mode}"
        )

    use_tdp = bool(getattr(args, "use_tdp", False))
    tdp_mode = getattr(args, "tdp_mode", "keep_base") or "keep_base"
    tdp_mode = str(tdp_mode).lower()
    valid_tdp_modes = {"local", "keep_base"}
    if use_tdp and tdp_mode not in valid_tdp_modes:
        raise ValueError(f"tdp_mode must be one of {sorted(valid_tdp_modes)}, got {tdp_mode}")

    use_tdp_semantic_encoder = bool(getattr(args, "use_tdp_semantic_encoder", False))
    tdp_semantic_mode = getattr(args, "tdp_semantic_mode", "gated_residual") or "gated_residual"
    tdp_semantic_mode = str(tdp_semantic_mode).lower()
    valid_tdp_semantic_modes = {"gated_residual", "baseline_residual"}
    if use_tdp_semantic_encoder and tdp_semantic_mode not in valid_tdp_semantic_modes:
        raise ValueError(
            f"tdp_semantic_mode must be one of {sorted(valid_tdp_semantic_modes)}, got {tdp_semantic_mode}"
        )
    if use_tdp_semantic_encoder and (not use_tdp or tdp_mode != "keep_base"):
        raise ValueError("use_tdp_semantic_encoder requires use_tdp=True and tdp_mode=keep_base.")
    tdp_gate_init_bias = float(getattr(args, "tdp_gate_init_bias", -4.0))
    tdp_debug_stats = bool(getattr(args, "tdp_debug_stats", False))
    tdp_debug_interval = int(getattr(args, "tdp_debug_interval", 1000) or 1000)
    if tdp_debug_interval <= 0:
        raise ValueError(f"tdp_debug_interval must be positive, got {tdp_debug_interval}")

    use_flow_cond = bool(getattr(args, "use_flow_cond", False))
    flow_cond_mode = getattr(args, "flow_cond_mode", "flow") or "flow"
    flow_cond_mode = str(flow_cond_mode).lower()
    valid_flow_modes = {"flow", "zero"}
    if use_flow_cond and flow_cond_mode not in valid_flow_modes:
        raise ValueError(f"flow_cond_mode must be one of {sorted(valid_flow_modes)}, got {flow_cond_mode}")
    flow_cond_dim = int(getattr(args, "flow_cond_dim", 32) or 32)
    flow_feature_dim = int(getattr(args, "flow_feature_dim", 3) or 3)
    flow_image_scale = float(getattr(args, "flow_image_scale", 0.25) or 0.25)
    flow_mag_scale = float(getattr(args, "flow_mag_scale", 1.0) or 1.0)
    flow_feature_mode = getattr(args, "flow_feature_mode", "mean_max_conf") or "mean_max_conf"
    flow_feature_mode = str(flow_feature_mode).lower()
    flow_knn_agg = getattr(args, "flow_knn_agg", "mean") or "mean"
    flow_knn_agg = str(flow_knn_agg).lower()
    flow_adapter_mode = getattr(args, "flow_adapter_mode", "concat") or "concat"
    flow_adapter_mode = str(flow_adapter_mode).lower()
    flow_gate_alpha = float(getattr(args, "flow_gate_alpha", 0.0) or 0.0)
    flow_gate_temp = float(getattr(args, "flow_gate_temp", 0.5) or 0.5)
    flow_view_token = bool(getattr(args, "flow_view_token", False))
    flow_view_attn_dim = int(getattr(args, "flow_view_attn_dim", 32) or 32)
    valid_flow_feature_modes = {"mean_max_conf", "mean_max_std", "uv_mag_std", "reproj_residual"}
    valid_flow_knn_aggs = {"mean", "mean_max"}
    valid_flow_adapter_modes = {"concat", "gated_residual"}
    if use_flow_cond and flow_feature_mode not in valid_flow_feature_modes:
        raise ValueError(f"flow_feature_mode must be one of {sorted(valid_flow_feature_modes)}, got {flow_feature_mode}")
    if use_flow_cond and flow_knn_agg not in valid_flow_knn_aggs:
        raise ValueError(f"flow_knn_agg must be one of {sorted(valid_flow_knn_aggs)}, got {flow_knn_agg}")
    if use_flow_cond and flow_adapter_mode not in valid_flow_adapter_modes:
        raise ValueError(f"flow_adapter_mode must be one of {sorted(valid_flow_adapter_modes)}, got {flow_adapter_mode}")
    if flow_cond_dim <= 0:
        raise ValueError(f"flow_cond_dim must be positive, got {flow_cond_dim}")
    if flow_feature_dim <= 0:
        raise ValueError(f"flow_feature_dim must be positive, got {flow_feature_dim}")
    if not (0.0 < flow_image_scale <= 1.0):
        raise ValueError(f"flow_image_scale must be in (0, 1], got {flow_image_scale}")
    if flow_mag_scale <= 0:
        raise ValueError(f"flow_mag_scale must be positive, got {flow_mag_scale}")
    if flow_gate_temp <= 0:
        raise ValueError(f"flow_gate_temp must be positive, got {flow_gate_temp}")
    if flow_view_attn_dim <= 0:
        raise ValueError(f"flow_view_attn_dim must be positive, got {flow_view_attn_dim}")
    if flow_view_token:
        if flow_feature_dim != 4:
            raise ValueError(f"flow_view_token expects flow_feature_dim=4 ([u,v,mag,conf]), got {flow_feature_dim}")
        if flow_knn_agg != "mean":
            raise ValueError("flow_view_token currently supports flow_knn_agg=mean only.")

    use_dif = bool(getattr(args, "use_dif", False))
    dif_mode = getattr(args, "dif_mode", "peak") or "peak"
    dif_mode = str(dif_mode).lower()
    valid_dif_modes = {"peak", "sigma_rectifier", "sigma_rectifier_v2", "uncert_loss"}
    if use_dif and dif_mode not in valid_dif_modes:
        raise ValueError(f"dif_mode must be one of {sorted(valid_dif_modes)}, got {dif_mode}")
    dif_sigma_init = float(getattr(args, "dif_sigma_init", -7.0))
    dif_eps = float(getattr(args, "dif_eps", 1e-6))
    if use_dif and dif_eps <= 0:
        raise ValueError(f"dif_eps must be positive, got {dif_eps}")
    dif_sigma_min = float(getattr(args, "dif_sigma_min", 1e-4))
    dif_sigma_max = float(getattr(args, "dif_sigma_max", 0.05))
    dif_residual_beta = float(getattr(args, "dif_residual_beta", 1.0))
    dif_residual_warmup = int(getattr(args, "dif_residual_warmup", 0) or 0)
    dif_sigma_prior = float(getattr(args, "dif_sigma_prior", 0.02))
    dif_sigma_prior_w = float(getattr(args, "dif_sigma_prior_w", 0.0))
    dif_uncert_loss_w = float(getattr(args, "dif_uncert_loss_w", 0.0))
    dif_uncert_s_min = float(getattr(args, "dif_uncert_s_min", -6.0))
    dif_uncert_s_max = float(getattr(args, "dif_uncert_s_max", 3.0))
    dif_debug_interval = int(getattr(args, "dif_debug_interval", 1000) or 1000)
    if use_dif:
        if dif_sigma_min <= 0:
            raise ValueError(f"dif_sigma_min must be positive, got {dif_sigma_min}")
        if dif_sigma_max <= dif_sigma_min:
            raise ValueError(
                f"dif_sigma_max must be larger than dif_sigma_min, got min={dif_sigma_min}, max={dif_sigma_max}"
            )
        if dif_residual_beta < 0:
            raise ValueError(f"dif_residual_beta must be non-negative, got {dif_residual_beta}")
        if dif_residual_warmup < 0:
            raise ValueError(f"dif_residual_warmup must be non-negative, got {dif_residual_warmup}")
        if dif_sigma_prior <= 0:
            raise ValueError(f"dif_sigma_prior must be positive, got {dif_sigma_prior}")
        if dif_sigma_prior_w < 0:
            raise ValueError(f"dif_sigma_prior_w must be non-negative, got {dif_sigma_prior_w}")
        if dif_uncert_loss_w < 0:
            raise ValueError(f"dif_uncert_loss_w must be non-negative, got {dif_uncert_loss_w}")
        if dif_uncert_s_max <= dif_uncert_s_min:
            raise ValueError(
                f"dif_uncert_s_max must be larger than dif_uncert_s_min, "
                f"got min={dif_uncert_s_min}, max={dif_uncert_s_max}"
            )
        if dif_debug_interval <= 0:
            raise ValueError(f"dif_debug_interval must be positive, got {dif_debug_interval}")
    if use_dif and bool(getattr(args, "use_part_moe", False)):
        raise ValueError("use_dif and use_part_moe are not combined in this ablation.")

    fix_stms = bool(getattr(args, "fix_stms", False))
    use_acc_cond = bool(getattr(args, "use_acc_cond", False))
    seq_acc_cond_dim = int(getattr(args, "seq_acc_cond_dim", 64) or 64)
    if use_acc_cond and seq_acc_cond_dim <= 0:
        raise ValueError(f"seq_acc_cond_dim must be positive, got {seq_acc_cond_dim}")
    if use_acc_cond:
        fix_stms = True

    use_motion_token = bool(getattr(args, "use_motion_token", False))
    motion_token_mode = getattr(args, "motion_token_mode", "none") or "none"
    motion_token_mode = str(motion_token_mode).lower()
    valid_motion_token_modes = {"none", "fix_stms", "acc", "part", "codebook", "full"}
    if motion_token_mode not in valid_motion_token_modes:
        raise ValueError(
            f"motion_token_mode must be one of {sorted(valid_motion_token_modes)}, got {motion_token_mode}"
        )
    use_motion_token = use_motion_token or motion_token_mode != "none"
    if not use_motion_token:
        motion_token_mode = "none"
    if use_motion_token and bool(getattr(args, "use_part_moe", False)):
        raise ValueError("use_motion_token and use_part_moe are not combined in this ablation.")
    if use_motion_token and use_dif:
        raise ValueError("use_motion_token and use_dif are not combined in this ablation.")
    if use_motion_token and any([use_msti, use_amc_pair, use_amc_causal, use_tdp]):
        raise ValueError("use_motion_token is mutually exclusive with MSTI, AMC, and TDP ablations.")
    if use_acc_cond and any([use_msti, use_amc_pair, use_amc_causal, use_tdp, use_dif, use_motion_token]):
        raise ValueError("use_acc_cond is a standalone ablation and is mutually exclusive with MSTI, AMC, TDP, DIF, and token ablations.")
    if use_flow_cond and any([use_msti, use_amc_pair, use_amc_causal, use_tdp, use_dif, use_motion_token, use_acc_cond]):
        raise ValueError("use_flow_cond is a standalone ablation and is mutually exclusive with MSTI, AMC, TDP, DIF, token, and acc ablations.")

    motion_token_num = int(getattr(args, "motion_token_num", 32) or 32)
    motion_token_dim = int(getattr(args, "motion_token_dim", 64) or 64)
    motion_token_part_dim = int(getattr(args, "motion_token_part_dim", 16) or 16)
    motion_token_acc_dim = int(getattr(args, "motion_token_acc_dim", 64) or 64)
    motion_token_debug_stats = bool(getattr(args, "motion_token_debug_stats", False))
    motion_token_debug_interval = int(getattr(args, "motion_token_debug_interval", 1000) or 1000)
    if use_motion_token:
        for name, value in {
            "motion_token_num": motion_token_num,
            "motion_token_dim": motion_token_dim,
            "motion_token_part_dim": motion_token_part_dim,
            "motion_token_acc_dim": motion_token_acc_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if motion_token_debug_interval <= 0:
            raise ValueError(
                f"motion_token_debug_interval must be positive, got {motion_token_debug_interval}"
            )

    if sum([use_msti, use_amc_pair, use_amc_causal, use_tdp]) > 1:
        raise ValueError("use_msti, use_amc_pair, use_amc_causal, and use_tdp are mutually exclusive in these ablations.")

    amc_causal_window = int(getattr(args, "amc_causal_window", 3) or 3)
    if use_amc_causal and amc_causal_window <= 0:
        raise ValueError(f"amc_causal_window must be positive, got {amc_causal_window}")
    args.amc_causal_window = amc_causal_window

    msti_mid_type = getattr(args, "msti_mid_type", "real") or "real"
    msti_mid_type = str(msti_mid_type).lower()
    valid_mid_types = {"real"}
    if use_msti and msti_mid_type not in valid_mid_types:
        raise ValueError(
            f"msti_mid_type={msti_mid_type} is not implemented yet; "
            f"supported types: {sorted(valid_mid_types)}"
        )

    if use_amc_pair:
        expected_cond_steps = time_step_num + time_step_num * (time_step_num - 1) // 2
    elif use_amc_causal:
        expected_cond_steps = time_step_num
    elif use_tdp:
        if tdp_mode == "local":
            expected_cond_steps = 2 * time_step_num
        else:
            expected_cond_steps = time_step_num + 2 * time_step_num - 1
    elif msti_mode == "none":
        expected_cond_steps = time_step_num
    elif msti_mode == "lite":
        expected_cond_steps = time_step_num + 2
    else:
        expected_cond_steps = time_step_num * 3

    requested_cond_steps = int(getattr(args, "motion_cond_time_step_num", 0) or 0)
    if requested_cond_steps > 0 and requested_cond_steps != expected_cond_steps:
        raise ValueError(
            "motion_cond_time_step_num mismatch: "
            f"requested={requested_cond_steps}, expected={expected_cond_steps} "
            f"for time_step_num={time_step_num}, msti_mode={msti_mode}, "
            f"use_amc_pair={use_amc_pair}, use_amc_causal={use_amc_causal}, "
            f"use_tdp={use_tdp}, tdp_mode={tdp_mode}"
        )

    args.use_msti = use_msti
    args.msti_mode = msti_mode
    args.msti_mid_type = msti_mid_type
    args.use_amc_pair = use_amc_pair
    args.amc_pair_mode = amc_pair_mode
    args.use_amc_causal = use_amc_causal
    args.amc_causal_mode = amc_causal_mode
    args.use_tdp = use_tdp
    args.tdp_mode = tdp_mode
    args.use_tdp_semantic_encoder = use_tdp_semantic_encoder
    args.tdp_semantic_mode = tdp_semantic_mode
    args.tdp_gate_init_bias = tdp_gate_init_bias
    args.tdp_debug_stats = tdp_debug_stats
    args.tdp_debug_interval = tdp_debug_interval
    args.use_flow_cond = use_flow_cond
    args.flow_cond_mode = flow_cond_mode
    args.flow_cond_dim = flow_cond_dim
    args.flow_feature_dim = flow_feature_dim
    args.flow_image_scale = flow_image_scale
    args.flow_mag_scale = flow_mag_scale
    args.flow_feature_mode = flow_feature_mode
    args.flow_knn_agg = flow_knn_agg
    args.flow_adapter_mode = flow_adapter_mode
    args.flow_gate_alpha = flow_gate_alpha
    args.flow_gate_temp = flow_gate_temp
    args.flow_view_token = flow_view_token
    args.flow_view_attn_dim = flow_view_attn_dim
    args.use_dif = use_dif
    args.dif_mode = dif_mode
    args.dif_sigma_init = dif_sigma_init
    args.dif_eps = dif_eps
    args.dif_sigma_min = dif_sigma_min
    args.dif_sigma_max = dif_sigma_max
    args.dif_residual_beta = dif_residual_beta
    args.dif_residual_warmup = dif_residual_warmup
    args.dif_sigma_prior = dif_sigma_prior
    args.dif_sigma_prior_w = dif_sigma_prior_w
    args.dif_uncert_loss_w = dif_uncert_loss_w
    args.dif_uncert_s_min = dif_uncert_s_min
    args.dif_uncert_s_max = dif_uncert_s_max
    args.dif_debug_interval = dif_debug_interval
    args.fix_stms = fix_stms
    args.use_acc_cond = use_acc_cond
    args.seq_acc_cond_dim = seq_acc_cond_dim
    args.use_motion_token = use_motion_token
    args.motion_token_mode = motion_token_mode
    args.motion_token_fix_stms = use_motion_token and motion_token_mode in {"fix_stms", "acc", "part", "codebook", "full"}
    args.motion_token_use_acc = use_motion_token and motion_token_mode in {"acc", "full"}
    args.motion_token_use_part = use_motion_token and motion_token_mode in {"part", "full"}
    args.motion_token_use_codebook = use_motion_token and motion_token_mode in {"codebook", "full"}
    args.motion_token_num = motion_token_num
    args.motion_token_dim = motion_token_dim
    args.motion_token_part_dim = motion_token_part_dim
    args.motion_token_acc_dim = motion_token_acc_dim
    args.motion_token_debug_stats = motion_token_debug_stats
    args.motion_token_debug_interval = motion_token_debug_interval
    args.motion_cond_time_step_num = expected_cond_steps
    return args

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
