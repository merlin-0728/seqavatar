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

    msti_mid_type = getattr(args, "msti_mid_type", "real") or "real"
    msti_mid_type = str(msti_mid_type).lower()
    valid_mid_types = {"real"}
    if use_msti and msti_mid_type not in valid_mid_types:
        raise ValueError(
            f"msti_mid_type={msti_mid_type} is not implemented yet; "
            f"supported types: {sorted(valid_mid_types)}"
        )

    if msti_mode == "none":
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
            f"for time_step_num={time_step_num}, msti_mode={msti_mode}"
        )

    args.use_msti = use_msti
    args.msti_mode = msti_mode
    args.msti_mid_type = msti_mid_type
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
