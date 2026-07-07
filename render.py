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
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, use_cached_smpl_rot=True):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    smpl_rot = {}
    smpl_rot_path = model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle'
    if not use_cached_smpl_rot:
        print("[Render] Disabled cached SMPL rotations for view-dependent deformation. Recomputing per view.")
    elif os.path.exists(smpl_rot_path):
        with open(smpl_rot_path, 'rb') as handle:
            smpl_rot = pickle.load(handle)
    else:
        print(f"[Render] Missing cached SMPL rotations: {smpl_rot_path}. Recomputing during render.")

    rgbs = []
    rgbs_gt = []
    elapsed_time = 0

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
            )
        else:
            render_output = render(view, gaussians, pipeline, background)
        # end time
        end_time = time.time()
        rendering = render_output["render"]
        
        # Calculate elapsed time
        elapsed_time += end_time - start_time
        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        rgbs.append(rendering)
        rgbs_gt.append(gt)

    # Calculate elapsed time
    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs, ssims, lpipss = 0.0, 0.0, 0.0

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        torchvision.utils.save_image(rendering, os.path.join(render_path, views[id].image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, views[id].image_name + ".png"))

        # metrics
        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)  

    # evalution metrics
    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if getattr(dataset, "use_part_moe", False):
            gaussians.part_moe_alpha = max(0.0, min(1.0, 1.0 - float(getattr(dataset, "part_moe_global_keep", 0.1))))
        if getattr(dataset, "use_part_moe", False):
            part_label_path = dataset.part_label_path
            if not part_label_path:
                part_iter = getattr(dataset, "part_moe_start_iter", 15000)
                part_label_path = os.path.join(
                    dataset.model_path,
                    "part_labels",
                    f"iteration_{part_iter}",
                    "gaussian_part_label.npy",
                )
            if os.path.exists(part_label_path):
                gaussians.load_part_labels(part_label_path)
                print(f"[Render] Loaded part labels: {part_label_path}")
            else:
                print(f"[Render] Part label file not found, cached d_nonrigid must be available: {part_label_path}")
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, use_cached_smpl_rot=not getattr(dataset, "flow_view_token", False))

        if not skip_test:
            for key in scene.getTestCameras().keys():
                render_set(dataset.model_path, key, scene.loaded_iter, scene.getTestCameras()[key], gaussians, pipeline, background, use_cached_smpl_rot=not getattr(dataset, "flow_view_token", False))

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
