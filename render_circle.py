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
import math
import numpy as np
import copy
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix_refine

# === 修正后的 LookAt 函数 (默认使用 B组逻辑) ===
def get_look_at_rotation(camera_position, target_position, up_vector=np.array([0, 1, 0])):
    # 使用 B组逻辑：Z 轴指向前方 (Target - Camera)
    # 这样可以修正 "全灰/背对" 的问题
    z_axis = target_position - camera_position 
        
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0, 0, 1])
    else:
        z_axis = z_axis / z_norm
    
    # 计算右向量 (Right Axis)
    x_axis = np.cross(up_vector, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = x_axis / x_norm
    
    # 计算上向量 (True Up Axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 构建旋转矩阵 R (Row-Major)
    R = np.eye(3)
    R[0, :] = x_axis
    R[1, :] = y_axis
    R[2, :] = z_axis
    
    # 3DGS 需要转置
    return R.transpose()

def render_circle_path(dataset, iteration, pipeline, background, scene, gaussians):
    """
    渲染左右30度圆弧视角的正式函数
    """
    # 1. 获取参考相机
    test_cameras = scene.getTestCameras()
    if isinstance(test_cameras, dict):
        seq_name = list(test_cameras.keys())[0]
        base_cam = test_cameras[seq_name][0]
        data_name = seq_name
    else:
        base_cam = test_cameras[0]
        data_name = "test"

    print(f"\n[Circle Render] Reference camera: '{base_cam.image_name}'")

    # 获取坐标信息
    cam_center = base_cam.camera_center.cpu().numpy()
    # 计算水平半径 (XZ平面)
    radius = np.linalg.norm(np.array([cam_center[0], cam_center[2]]))
    # 保持高度 (Y轴)
    height = cam_center[1]
    
    # 2. 设置保存路径
    render_path = os.path.join(dataset.model_path, data_name, "ours_{}".format(iteration), "renders_circle_72")
    makedirs(render_path, exist_ok=True)

    # 3. 加载动作数据
    with open(dataset.model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle', 'rb') as handle:
        smpl_rot = pickle.load(handle)
    
    # 使用参考相机的动作 ID (定格动画)
    pose_id = base_cam.pose_id
    d_nonrigid = smpl_rot[data_name][pose_id]['d_nonrigid']
    transforms = smpl_rot[data_name][pose_id]['transforms']
    translation = smpl_rot[data_name][pose_id]['translation']

    # 4. 生成 72 个相机轨迹
    render_cameras = []
    
    # 角度范围：-30度 到 +30度
    num_views = 72
    angles = np.linspace(-30, 30, num_views)
    
    # 计算初始角度
    # === 修改处：增加 180 度，将相机移动到对面（从背面移到正面） ===
    base_angle = np.degrees(np.arctan2(cam_center[0], cam_center[2])) + 180.0
    
    print("Generating camera path...")
    
    for i, angle_deg in enumerate(angles):
        current_angle = base_angle + angle_deg
        rad = np.radians(current_angle)
        
        # 计算新位置 (圆弧轨迹)
        new_x = radius * np.sin(rad)
        new_z = radius * np.cos(rad)
        new_pos = np.array([new_x, height, new_z])
        
        # 目标点：假设人站在原点 (0,0,0)
        target_pos = np.array([0.0, 0.0, 0.0]) 

        # 创建新相机
        new_view = copy.deepcopy(base_cam)
        new_view.uid = 200000 + i
        new_view.image_name = f"circle_{i:03d}_angle_{int(angle_deg)}"
        
        # 计算旋转 (使用修正后的 LookAt)
        new_R = get_look_at_rotation(new_pos, target_pos)
        
        # 更新外参
        new_view.R = new_R
        new_view.T = -np.dot(new_R.transpose(), new_pos)
        
        # 更新变换矩阵
        new_view.world_view_transform = torch.tensor(
            getWorld2View2(new_view.R, new_view.T, new_view.trans, new_view.scale)
        ).transpose(0, 1).cuda()
        
        new_view.full_proj_transform = (
            new_view.world_view_transform.unsqueeze(0).bmm(new_view.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
        new_view.camera_center = new_view.world_view_transform.inverse()[3, :3]
        render_cameras.append(new_view)

    # 5. 开始渲染
    print(f"Rendering {len(render_cameras)} frames to {render_path}...")
    for idx, view in enumerate(tqdm(render_cameras, desc="Rendering Circle")):
        
        render_output = render(view, gaussians, pipeline, background, 
                               transforms=transforms, 
                               translation=translation, 
                               d_nonrigid=d_nonrigid)
        
        rendering = render_output["render"]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        
    print("Done.")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_circle : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # 恢复正常的黑/白背景
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 触发圆弧渲染
        if render_circle:
            render_circle_path(dataset, iteration, pipeline, background, scene, gaussians)
            return

        # 原有逻辑
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            for key in scene.getTestCameras().keys():
                render_set(dataset.model_path, key, scene.loaded_iter, scene.getTestCameras()[key], gaussians, pipeline, background)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # 原有的渲染函数，保持不变
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

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

        start_time = time.time() 
        render_output = render(view, gaussians, pipeline, background, transforms=transforms, translation=translation, d_nonrigid=d_nonrigid)
        end_time = time.time()
        rendering = render_output["render"]
        
        elapsed_time += end_time - start_time
        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        rgbs.append(rendering)
        rgbs_gt.append(gt)

    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs, ssims, lpipss = 0.0, 0.0, 0.0

    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    import lpips
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        torchvision.utils.save_image(rendering, os.path.join(render_path, views[id].image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, views[id].image_name + ".png"))

        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)  

    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # 添加一个新参数来触发圆弧渲染
    parser.add_argument("--render_circle", action="store_true", help="Render a 60-degree arc around the subject")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.render_circle)