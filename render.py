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
import datetime # [新增] 用于获取当前年月日
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

<<<<<<< Updated upstream
# --- [WANDB 新增] 引入库 ---
=======
# --- [WANDB 引入] ---
>>>>>>> Stashed changes
import wandb

loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
<<<<<<< Updated upstream
    # 【修改点 1】注释掉本地创建文件夹的逻辑，不再占用本地硬盘
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
=======
    # [修改点 1 & 2] 获取当前年月日时分秒，恢复并修改本地创建文件夹逻辑
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    render_folder_name = f"renders_{current_time}"
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), render_folder_name)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
>>>>>>> Stashed changes

    # Load data (deserialize)
    with open(model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle', 'rb') as handle:
        smpl_rot = pickle.load(handle)

    rgbs = []
    rgbs_gt = []
    elapsed_time = 0

    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        bound_mask = view.bound_mask

        d_nonrigid = smpl_rot[name][view.pose_id]['d_nonrigid']
        transforms, translation = smpl_rot[name][view.pose_id]['transforms'], smpl_rot[name][view.pose_id]['translation']

        # Start timer
        start_time = time.time() 
        render_output = render(view, gaussians, pipeline, background, transforms=transforms, translation=translation, d_nonrigid=d_nonrigid)
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
    
    # --- [WANDB 新增] 用于收集上传图片的列表 ---
    wandb_renders = []
    wandb_gts = []

    # [修改点 3] 移除了 wandb_renders 和 wandb_gts 列表的初始化

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

<<<<<<< Updated upstream
        # 【修改点 2】注释掉本地保存图片的逻辑
        # torchvision.utils.save_image(rendering, os.path.join(render_path, views[id].image_name + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, views[id].image_name + ".png"))

        # --- [WANDB 新增] 将张量打包为 wandb.Image 对象，附带文件名作为标注 ---
        wandb_renders.append(wandb.Image(rendering, caption=f"Render_{views[id].image_name}"))
        wandb_gts.append(wandb.Image(gt, caption=f"GT_{views[id].image_name}"))
=======
        # [修改点 2] 恢复本地保存图片的逻辑
        torchvision.utils.save_image(rendering, os.path.join(render_path, views[id].image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, views[id].image_name + ".png"))
>>>>>>> Stashed changes

        # metrics
        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)  

    # evalution metrics
    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

<<<<<<< Updated upstream
    # --- [WANDB 新增] 一次性将所有图片和当前视角的指标上传到 WandB ---
    wandb.log({
        f"Eval_{name}/Rendered_Images": wandb_renders,
        f"Eval_{name}/Ground_Truth": wandb_gts,
=======
    # [修改点 3] 移除了图像上传逻辑，仅保留客观指标的 WandB 同步
    wandb.log({
>>>>>>> Stashed changes
        f"Eval_{name}/Avg_PSNR": psnrs,
        f"Eval_{name}/Avg_SSIM": ssims,
        f"Eval_{name}/Avg_LPIPS": lpipss,
        "iteration": iteration
    })

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            for key in scene.getTestCameras().keys():
                # [修改点 4] 过滤掉 circle 视角的渲染任务
                if "circle" in key.lower():
                    print(f"\nSkipping {key} rendering as requested...")
                    continue
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
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

<<<<<<< Updated upstream
    # --- [WANDB 新增] 初始化渲染阶段的看板 ---
    # 我们用 job_type="eval" 来区分它和训练任务，并用文件夹名作为实验名
=======
    # --- [WANDB] 初始化渲染阶段的看板 ---
>>>>>>> Stashed changes
    exp_name = os.path.basename(args.model_path.rstrip('/'))
    wandb.init(
        project="SeqAvatar", 
        name=f"Eval_{exp_name}", 
        job_type="eval",
        config=vars(args)
    )

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

<<<<<<< Updated upstream
    # --- [WANDB 新增] 结束并同步数据 ---
=======
    # --- [WANDB] 结束并同步数据 ---
>>>>>>> Stashed changes
    wandb.finish()