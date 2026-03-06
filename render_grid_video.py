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
from PIL import Image
import imageio

# === 修正后的 LookAt 函数 (Z轴指向前方) ===
def get_look_at_rotation(camera_position, target_position, up_vector=np.array([0, 1, 0])):
    z_axis = target_position - camera_position 
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0, 0, 1])
    else:
        z_axis = z_axis / z_norm
    
    x_axis = np.cross(up_vector, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = x_axis / x_norm
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    R = np.eye(3)
    R[0, :] = x_axis
    R[1, :] = y_axis
    R[2, :] = z_axis
    return R.transpose()

def create_grid_image(image_list, grid_rows=6, grid_cols=12):
    """
    在内存中将 tensor 列表拼接成 PIL Grid 图片
    """
    if not image_list:
        return None
    
    # 转换第一张图获取尺寸 (C, H, W) -> PIL
    def to_pil(tensor_img):
        arr = tensor_img.cpu().numpy().transpose(1, 2, 0)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    first_img = to_pil(image_list[0])
    w, h = first_img.size
    
    total_w = w * grid_cols
    total_h = h * grid_rows
    
    grid = Image.new('RGB', (total_w, total_h), (0, 0, 0))
    
    for idx, tensor_img in enumerate(image_list):
        if idx >= grid_rows * grid_cols: break
        
        row = idx // grid_cols
        col = idx % grid_cols
        
        pil_img = to_pil(tensor_img)
        grid.paste(pil_img, (col * w, row * h))
        
    return grid

def render_dynamic_grid_video(dataset, iteration, pipeline, background, scene, gaussians, fps=24):
    """
    渲染动态宫格视频：时间 x 空间 (72 views)
    使用训练集数据以获取完整的 100 帧连续动作
    """
    # 1. 切换数据源为训练集 (Train Set)
    # 因为 DNA-Rendering 的训练集包含了完整的 0-99 帧，而测试集是抽样的
    print("\n[Grid Video] Switching to TRAINING sequence to get full 100 continuous frames...")
    train_cameras = scene.getTrainCameras()
    
    if isinstance(train_cameras, dict):
        seq_name = list(train_cameras.keys())[0]
        base_cam_list = train_cameras[seq_name]
        data_name = seq_name
    else:
        base_cam_list = train_cameras
        data_name = "train"

    if len(base_cam_list) == 0:
        print("Error: No training cameras found! Please check dataset path.")
        return

    # 使用列表中的第一个相机来确定圆弧轨迹的“基准位置”
    ref_cam = base_cam_list[0]
    
    # 修改输出路径，避免覆盖
    output_dir = os.path.join(dataset.model_path, data_name, "ours_{}".format(iteration), "grid_video_100frames")
    images_dir = os.path.join(output_dir, "frames") # 用于存放每一张宫格图
    makedirs(output_dir, exist_ok=True)
    makedirs(images_dir, exist_ok=True)
    
    video_path = os.path.join(output_dir, "dynamic_grid_view_100.mp4")
    
    print(f"[Grid Video] Output directory: {output_dir}")
    print(f"[Grid Video] Saving individual frames to: {images_dir}")

    # 加载 SMPL 动作数据
    # 注意：这里需要确保 pickle 文件里包含 'train' 或对应 seq_name 的 key
    # 通常 smpl_rot.pickle 结构是 {seq_name: {pose_id: ...}}，所以只要 seq_name 对就行
    with open(dataset.model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle', 'rb') as handle:
        smpl_rot = pickle.load(handle)

    # 2. 预计算 72 个相机的相对位置 (几何轨迹)
    cam_center = ref_cam.camera_center.cpu().numpy()
    radius = np.linalg.norm(np.array([cam_center[0], cam_center[2]]))
    height = cam_center[1]
    
    # 增加 180 度修正，确保是正面
    base_angle = np.degrees(np.arctan2(cam_center[0], cam_center[2])) + 180.0
    
    num_views = 72
    angles = np.linspace(-30, 30, num_views)
    
    orbit_params = []
    target_pos = np.array([0.0, 0.0, 0.0])
    
    print("Pre-calculating orbit path...")
    for angle_deg in angles:
        current_angle = base_angle + angle_deg
        rad = np.radians(current_angle)
        
        new_x = radius * np.sin(rad)
        new_z = radius * np.cos(rad)
        new_pos = np.array([new_x, height, new_z])
        
        new_R = get_look_at_rotation(new_pos, target_pos)
        new_T = -np.dot(new_R.transpose(), new_pos)
        
        orbit_params.append((new_R, new_T, new_pos))

    # 3. 关键修复：筛选唯一的时间帧 (Unique Pose Frames)
    print(f"Raw camera list size: {len(base_cam_list)}")
    
    unique_pose_frames = []
    seen_poses = set()
    
    # 先按 Pose ID 排序，确保时间顺序 (0, 1, 2, ... 99)
    sorted_cams = sorted(base_cam_list, key=lambda x: x.pose_id)
    
    for cam in sorted_cams:
        if cam.pose_id not in seen_poses:
            unique_pose_frames.append(cam)
            seen_poses.add(cam.pose_id)
    
    processing_frames = unique_pose_frames
    print(f"Filtered to {len(processing_frames)} unique temporal frames (Should be around 100).")

    # 4. 初始化视频写入器
    # 帧数增多到100帧，FPS 建议设为 24 或 30 以获得流畅的 3-4 秒视频
    writer = imageio.get_writer(video_path, fps=fps, quality=8, macro_block_size=None)
    
    # 5. 双重循环：时间 (Frames) -> 空间 (72 Views)
    print(f"Start rendering {len(processing_frames)} frames. Each frame contains {num_views} sub-views.")
    
    for frame_idx, current_cam in enumerate(tqdm(processing_frames, desc="Rendering Sequence")):
        
        # === A. 获取当前帧的动作数据 ===
        pose_id = current_cam.pose_id
        try:
            # 注意：如果 smpl_rot 的 key 是 'train' 或 'test' 而不是 seq_name，这里可能需要调整
            # 通常 DNA-Rendering 的 pickle 结构是 { 'sequence_name': ... }
            if data_name in smpl_rot:
                target_dict = smpl_rot[data_name]
            elif "train" in smpl_rot: # Fallback guessing
                target_dict = smpl_rot["train"]
            else:
                # 尝试用第一个 key
                target_dict = smpl_rot[list(smpl_rot.keys())[0]]

            d_nonrigid = target_dict[pose_id]['d_nonrigid']
            transforms = target_dict[pose_id]['transforms']
            translation = target_dict[pose_id]['translation']
        except KeyError:
            print(f"Skipping frame {frame_idx} (pose_id {pose_id}) due to missing motion data.")
            continue

        # === B. 渲染当前帧的 72 个视角 ===
        frame_images = []
        
        for view_idx, (R, T, pos) in enumerate(orbit_params):
            view_cam = copy.deepcopy(current_cam) 
            
            view_cam.R = R
            view_cam.T = T
            view_cam.camera_center = torch.tensor(pos, dtype=torch.float32, device="cuda")
            
            view_cam.world_view_transform = torch.tensor(
                getWorld2View2(R, T, view_cam.trans, view_cam.scale)
            ).transpose(0, 1).cuda()
            
            view_cam.full_proj_transform = (
                view_cam.world_view_transform.unsqueeze(0).bmm(view_cam.projection_matrix.unsqueeze(0))
            ).squeeze(0)
            
            # 渲染
            render_pkg = render(view_cam, gaussians, pipeline, background, 
                                transforms=transforms, 
                                translation=translation, 
                                d_nonrigid=d_nonrigid)
            
            img = render_pkg["render"]
            img = torch.clamp(img, 0.0, 1.0)
            frame_images.append(img)
        
        # === C. 拼图 (内存操作) ===
        grid_img_pil = create_grid_image(frame_images, grid_rows=6, grid_cols=12)
        
        # === D. 保存单帧图片 (新增需求) ===
        # 文件名包含 pose_id 以便核对
        frame_filename = f"grid_frame_{frame_idx:04d}_pose{pose_id}.png"
        grid_img_pil.save(os.path.join(images_dir, frame_filename), optimize=True)

        # === E. 写入视频 ===
        writer.append_data(np.array(grid_img_pil))

    writer.close()
    print(f"\nDone! Video saved to: {video_path}")
    print(f"Individual frames saved in: {images_dir}")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_circle : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if render_circle:
            # 调用新的视频生成函数
            render_dynamic_grid_video(dataset, iteration, pipeline, background, scene, gaussians)
            return

        # 原有逻辑 (保留)
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            for key in scene.getTestCameras().keys():
                render_set(dataset.model_path, key, scene.loaded_iter, scene.getTestCameras()[key], gaussians, pipeline, background)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # 原有逻辑 (保留)
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
    parser.add_argument("--render_circle", action="store_true", help="Render 6x12 grid video")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.render_circle)