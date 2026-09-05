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
from scene import Scene
import os
import time
import pickle
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from part_label.common import default_part_label_dir
from ablations.part_moe_controller import build_part_moe_controller

from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    smpl_rot = {}
    smpl_rot_path = model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle'
    if os.path.exists(smpl_rot_path):
        with open(smpl_rot_path, 'rb') as handle:
            smpl_rot = pickle.load(handle)
    else:
        print(f"[Render] Missing cached SMPL rotations: {smpl_rot_path}. Recomputing during render.")

    elapsed_time = 0
    psnrs, ssims, lpipss = 0.0, 0.0, 0.0

    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        bound_mask = view.bound_mask

        # Start timer
        start_time = time.time() 
        cached_pose = smpl_rot.get(name, {}).get(view.pose_id)
        if cached_pose is not None:
            d_nonrigid = cached_pose['d_nonrigid']
            transforms = cached_pose['transforms']
            translation = cached_pose['translation']
            render_output = render(
                view,
                gaussians,
                pipeline,
                background,
                transforms=transforms,
                translation=translation,
                d_nonrigid=d_nonrigid,
                iteration=iteration,
            )
        else:
            render_output = render(view, gaussians, pipeline, background, iteration=iteration)
        # end time
        end_time = time.time()
        rendering = render_output["render"]
        
        # Calculate elapsed time
        elapsed_time += end_time - start_time
        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        torchvision.utils.save_image(rendering.cpu(), os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt.cpu(), os.path.join(gts_path, view.image_name + ".png"))

        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

        del render_output, rendering, gt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate elapsed time
    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)  

    metrics_dir = os.path.join(model_path, "metrics")
    makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"results_{name}_{iteration}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "iteration": int(iteration),
                "name": name,
                "num_views": int(len(views)),
                "psnr": float(psnrs),
                "ssim": float(ssims),
                "lpips": float(lpipss),
            },
            f,
            indent=2,
        )

    # evalution metrics
    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if getattr(dataset, "use_mapo_all_dynamic", False):
            gaussians.non_rigid_deformer.mapo_set_training_level(
                getattr(dataset, "mapo_max_partition_level", 0)
            )
            print(
                "[MAPO_ALL_DYNAMIC] render configured: "
                f"level={gaussians.non_rigid_deformer.mapo_active_level} "
                f"branches={1 << gaussians.non_rigid_deformer.mapo_active_level}"
            )
        if getattr(dataset, "use_temporal_conditioned_part_moe", False):
            gaussians.init_temporal_conditioned_part_moe()
            print(
                "[TEMPORAL_CONDITIONED_PART] render activation enabled; "
                "using temporal-conditioned Part predictors."
            )
        if getattr(dataset, "use_part_moe", False):
            gaussians.part_moe_alpha = max(0.0, min(1.0, 1.0 - float(getattr(dataset, "part_moe_global_keep", 0.1))))
        if getattr(dataset, "use_part_budget", False):
            gaussians.part_budget_alpha_scale = 1.0
        if getattr(dataset, "use_tri_gate", False):
            gaussians.tri_gate_alpha_scale = 1.0
        if getattr(dataset, "use_tri_token", False):
            gaussians.tri_token_alpha_scale = 1.0
        if getattr(dataset, "use_part_moe", False):
            part_controller = build_part_moe_controller(dataset)
            current_part_dir = default_part_label_dir(dataset.model_path, iteration)
            current_part_label_path = os.path.join(current_part_dir, "gaussian_part_label.npy")
            candidate_paths = []
            if dataset.part_label_path:
                candidate_paths.append(dataset.part_label_path)
            candidate_paths.append(current_part_label_path)
            part_labels_root = os.path.join(dataset.model_path, "part_labels")
            refreshed_label_dirs = []
            if os.path.isdir(part_labels_root):
                for name in os.listdir(part_labels_root):
                    if not name.startswith("iteration_"):
                        continue
                    try:
                        refreshed_label_dirs.append((int(name.split("_")[-1]), name))
                    except ValueError:
                        continue
                for _, name in sorted(refreshed_label_dirs, reverse=True):
                    candidate_paths.append(
                        os.path.join(part_labels_root, name, "gaussian_part_label.npy")
                    )
            part_iter = getattr(dataset, "part_moe_start_iter", 15000)
            candidate_paths.append(
                os.path.join(
                    dataset.model_path,
                    "part_labels",
                    f"iteration_{part_iter}",
                    "gaussian_part_label.npy",
                )
            )

            loaded = False
            rebuild_dir = current_part_dir
            for part_label_path in candidate_paths:
                if not os.path.exists(part_label_path):
                    continue
                part_conf_path = os.path.join(os.path.dirname(part_label_path), "gaussian_part_conf.npy")
                labels = torch.from_numpy(np.load(part_label_path)).to(gaussians.device)
                if labels.shape[0] != gaussians.get_xyz.shape[0]:
                    print(
                        f"[Render] Part label mismatch at {part_label_path}: "
                        f"{labels.shape[0]} vs {gaussians.get_xyz.shape[0]}; rebuilding for iteration {iteration}."
                    )
                    break
                gaussians.load_part_labels(
                    part_label_path,
                    part_conf_path if os.path.exists(part_conf_path) else None,
                )
                print(f"[Render] Loaded part labels: {part_label_path}")
                loaded = True
                break

            if not loaded:
                print(f"[Render] Rebuilding part labels at {rebuild_dir} for iteration {iteration}.")
                rebuild_dir.mkdir(parents=True, exist_ok=True)
                label_path, conf_path = part_controller._build_prior_only_labels(
                    rebuild_dir,
                    iteration,
                    gaussians,
                    dataset.model_path,
                )
                gaussians.load_part_labels(label_path, conf_path)
                print(f"[Render] Loaded rebuilt part labels: {label_path}")
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            for key in scene.getTestCameras().keys():
                render_set(dataset.model_path, key, scene.loaded_iter, scene.getTestCameras()[key], gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if getattr(args, "use_part_moe", False):
        from part_label.common import enable_part_stdout_logging
        enable_part_stdout_logging(args, "render")
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
