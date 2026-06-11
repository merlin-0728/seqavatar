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

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
# ================= User Settings =================
# Change this value when you want to use a different physical GPU.
GPU_ID = "2"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
PYTHON_BIN_DIR = Path(sys.executable).resolve().parent
os.environ["PATH"] = f"{PYTHON_BIN_DIR}:{os.environ.get('PATH', '')}"

import torch
from scene import Scene
import argparse
import json
import time
import pickle
import math
import numpy as np
import copy
from datetime import datetime
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

from freeview import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_ROOT,
    build_dataset_args,
    build_pipeline_args,
    pose_target,
    prepare_output_dir,
    read_metric,
    resolve_sequence,
    select_run,
    tensor_to_numpy,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "freeview" / "DNA-Rendering"
DEFAULT_SEQUENCES = ["0044"]

# ================= Camera Extrinsic Defaults =================
# 相机整体绕人体水平旋转的偏移角度，单位是度。
# 0.0 表示沿用 reference_index 对应训练相机所在方向；如果看到的是背面，
# 改成 180.0 就会转到对面看正面；90.0/-90.0 可以看左右侧面。
DEFAULT_BASE_YAW_OFFSET = 180.0


def serializable_args(args):
    output = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def write_metadata(out_dir, metadata):
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    lines = [
        "SeqAvatar render_grid_video render",
        "",
        f"sequence: {metadata.get('sequence')}",
        f"dataset: {metadata.get('dataset')}",
        f"model: {metadata.get('model')}",
        f"experiment: {metadata.get('experiment')}",
        f"run: {metadata.get('run')}",
        f"iteration: {metadata.get('iteration')}",
        "",
        f"split: {metadata.get('split')}",
        f"reference_camera: {metadata.get('reference_camera')}",
        f"reference_index: {metadata.get('reference_index')}",
        f"target_mode: {metadata.get('target_mode')}",
        f"target: {metadata.get('target')}",
        f"frame_count: {metadata.get('frame_count')}",
        f"rendered_count: {metadata.get('rendered_count')}",
        f"num_views: {metadata.get('num_views')}",
        f"angle_start: {metadata.get('angle_start')}",
        f"angle_end: {metadata.get('angle_end')}",
        f"radius: {metadata.get('radius')}",
        f"height: {metadata.get('height')}",
        f"base_yaw: {metadata.get('base_yaw')}",
        f"base_yaw_offset: {metadata.get('base_yaw_offset')}",
        f"grid_rows: {metadata.get('grid_rows')}",
        f"grid_cols: {metadata.get('grid_cols')}",
        "",
        f"video: {metadata.get('video')}",
        f"frames_dir: {metadata.get('frames_dir')}",
        "",
        "command:",
        f"  {metadata.get('command')}",
    ]
    with open(os.path.join(out_dir, "render_info.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

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

    # 使用参考相机继承内参/分辨率，并按 freeview.py 的方式确定相机半径和高度
    reference_index = getattr(dataset, "render_reference_index", 0)
    if reference_index < 0 or reference_index >= len(base_cam_list):
        raise IndexError(f"reference_index {reference_index} out of range 0..{len(base_cam_list)-1}")
    ref_cam = base_cam_list[reference_index]
    
    # 修改输出路径，避免覆盖
    output_dir = getattr(dataset, "render_output_dir", None)
    if output_dir is None:
        output_dir = os.path.join(dataset.model_path, data_name, "ours_{}".format(iteration), "grid_video_100frames")
    else:
        output_dir = str(output_dir)

    images_dir = os.path.join(output_dir, "frames") # 用于存放每一张宫格图
    makedirs(output_dir, exist_ok=True)
    save_frames = getattr(dataset, "render_save_frames", True)
    make_video = getattr(dataset, "render_make_video", True)
    if save_frames:
        makedirs(images_dir, exist_ok=True)
    
    video_name = getattr(dataset, "render_video_name", "render_grid_video.mp4")
    video_path = os.path.join(output_dir, video_name)
    fps = getattr(dataset, "render_fps", fps)
    quality = getattr(dataset, "render_quality", 8)
    
    print(f"[Grid Video] Output directory: {output_dir}")
    if save_frames:
        print(f"[Grid Video] Saving individual frames to: {images_dir}")

    # 2. 按 freeview.py 的方式计算目标点、半径、高度和基准 yaw
    ref_center = tensor_to_numpy(ref_cam.camera_center).astype(np.float64)
    ref_target = pose_target(gaussians, int(ref_cam.pose_id), dataset)
    ref_delta = ref_center - ref_target
    base_radius = math.hypot(float(ref_delta[0]), float(ref_delta[2]))
    if base_radius < 1e-8:
        base_radius = 1.0
    radius = (
        getattr(dataset, "render_radius", None)
        if getattr(dataset, "render_radius", None) is not None
        else base_radius * getattr(dataset, "render_radius_scale", 1.0)
    )
    # base_height_offset: 参考相机相对人体中心的高度差。
    # 默认保持这个高度差；如果想整体抬高/降低相机，看下面 render_height_offset。
    base_height_offset = float(ref_delta[1])

    # base_yaw: 参考相机绕人体的水平角度，是所有自由视角相机的中心方向。
    # render_base_yaw_offset 会整体旋转这一组相机，比如 180 度从背面转到正面。
    base_yaw = math.degrees(math.atan2(float(ref_delta[0]), float(ref_delta[2]))) + getattr(dataset, "render_base_yaw_offset", 0.0)

    # num_views: 每个时间帧渲染多少个相邻相机视角。
    # grid_rows/grid_cols: 输出视频中拼接宫格的行列数，必须满足 rows * cols >= num_views。
    # angle_start/angle_end: 这些相机相对 base_yaw 的左右展开范围。
    # 例如 30 个相机、-10 到 10 表示 30 个相机均匀分布在正面左右 20 度范围内。
    # 想相机间距更小，就缩小角度范围；想覆盖更宽，就扩大角度范围。
    num_views = getattr(dataset, "render_num_views", 72)
    grid_rows = getattr(dataset, "render_grid_rows", 6)
    grid_cols = getattr(dataset, "render_grid_cols", 12)
    angle_start = getattr(dataset, "render_angle_start", -30.0)
    angle_end = getattr(dataset, "render_angle_end", 30.0)
    if grid_rows * grid_cols < num_views:
        raise ValueError(f"grid_rows * grid_cols must be >= num_views, got {grid_rows}x{grid_cols} for {num_views}")

    angles = np.linspace(angle_start, angle_end, num_views)

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
    writer = None
    if make_video:
        writer = imageio.get_writer(video_path, fps=fps, quality=quality, macro_block_size=None)
    
    # 5. 双重循环：时间 (Frames) -> 空间 (72 Views)
    print(f"Start rendering {len(processing_frames)} frames. Each frame contains {num_views} sub-views.")
    
    rendered_count = 0
    for frame_idx, current_cam in enumerate(tqdm(processing_frames, desc="Rendering Sequence")):
        
        # === A. 获取当前帧的目标点，保持和 freeview.py 一样围绕运动人体中心 ===
        pose_id = current_cam.pose_id
        target_pos = pose_target(gaussians, int(pose_id), dataset)

        # === B. 渲染当前帧的相邻视角 ===
        frame_images = []
        
        for view_idx, angle_deg in enumerate(angles):
            # yaw 是当前相机绕人体中心的水平角度。
            # base_yaw 决定这一组相机朝向哪一侧，angle_deg 决定组内从左到右的相邻视角。
            yaw = math.radians(base_yaw + float(angle_deg))

            # pos_y 是相机高度。
            # render_height=None 时，沿用参考相机相对人体中心的高度；
            # render_height 给具体数值时，会固定使用这个世界坐标高度。
            pos_y = (
                getattr(dataset, "render_height", None)
                if getattr(dataset, "render_height", None) is not None
                else target_pos[1] + base_height_offset
            )

            # pos 是当前相机的世界坐标位置，也就是外参里的 camera center。
            # radius 控制相机离人体中心的水平距离，height_offset 控制相机上下偏移。
            # x/z 由 yaw 决定，y 由 pos_y + height_offset 决定。
            pos = np.array(
                [
                    target_pos[0] + radius * math.sin(yaw),
                    pos_y + getattr(dataset, "render_height_offset", 0.0),
                    target_pos[2] + radius * math.cos(yaw),
                ],
                dtype=np.float64,
            )

            # R/T 是真正传给渲染器的相机外参。
            # R: 相机旋转矩阵，让相机始终看向当前帧人体中心 target_pos。
            # T: 相机平移向量，由 R 和相机世界坐标 pos 推出。
            R = get_look_at_rotation(pos, target_pos)
            T = -np.dot(R.transpose(), pos)

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
            render_pkg = render(view_cam, gaussians, pipeline, background)
            
            img = render_pkg["render"]
            img = torch.clamp(img, 0.0, 1.0)
            frame_images.append(img)
        
        # === C. 拼图 (内存操作) ===
        grid_img_pil = create_grid_image(frame_images, grid_rows=grid_rows, grid_cols=grid_cols)
        
        # === D. 保存单帧图片 (新增需求) ===
        # 文件名包含 pose_id 以便核对
        frame_filename = f"grid_frame_{frame_idx:04d}_pose{pose_id}.png"
        if save_frames:
            grid_img_pil.save(os.path.join(images_dir, frame_filename), optimize=True)

        # === E. 写入视频 ===
        if writer is not None:
            writer.append_data(np.array(grid_img_pil))
        rendered_count += 1

    if writer is not None:
        writer.close()

    metadata = {
        "sequence": getattr(dataset, "render_sequence", None),
        "dataset": dataset.source_path,
        "model": dataset.model_path,
        "experiment": getattr(dataset, "render_experiment", None),
        "run": getattr(dataset, "render_run", None),
        "iteration": iteration,
        "renderer": "render_grid_video",
        "split": data_name,
        "reference_camera": ref_cam.image_name,
        "reference_index": reference_index,
        "target_mode": getattr(dataset, "target_mode", None),
        "target": list(getattr(dataset, "target", [])),
        "frame_count": len(processing_frames),
        "rendered_count": rendered_count,
        "motion_mode": "renderer_on_the_fly",
        "num_views": num_views,
        "angle_start": float(angles[0]),
        "angle_end": float(angles[-1]),
        "radius": float(radius),
        "height": getattr(dataset, "render_height", None),
        "height_offset": getattr(dataset, "render_height_offset", 0.0),
        "base_yaw": float(base_yaw),
        "base_yaw_offset": getattr(dataset, "render_base_yaw_offset", 0.0),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "fps": fps,
        "video": video_path if make_video else None,
        "frames_dir": images_dir if save_frames else None,
        "metric": getattr(dataset, "render_metric", None),
        "parameters": getattr(dataset, "render_parameters", None),
        "command": getattr(dataset, "render_command", " ".join([Path(sys.executable).name] + sys.argv)),
    }
    write_metadata(output_dir, metadata)

    print(f"\nDone! Output saved to: {output_dir}")
    if make_video:
        print(f"Video saved to: {video_path}")
    if save_frames:
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


def apply_dna_defaults(args):
    def set_default(name, value):
        if getattr(args, name, None) is None:
            setattr(args, name, value)

    if args.iteration in (None, -1):
        args.iteration = 25000
    set_default("sh_degree", 3)
    set_default("images", "images")
    set_default("resolution", -1)
    set_default("white_background", False)
    set_default("data_device", "cuda")
    set_default("eval", True)
    set_default("smpl_type", "smplx")
    set_default("actor_gender", "neutral")
    set_default("motion_offset_flag", True)
    set_default("non_rigid_flag", True)
    set_default("nonrigid_poseconds_flag", True)
    set_default("nonrigid_deltaposeconds_flag", True)
    set_default("nonrigid_deltaxyzconds_flag", True)
    set_default("seq_len", 8)
    set_default("seq_xyz_knn", 8)
    set_default("time_step_num", 3)
    set_default("max_time_step", 3)
    set_default("minimal_time_step", 1)


def run_sequence_mode(args, model, pipeline):
    args.dataset_root = Path(args.dataset_root).resolve()
    args.model_root = Path(args.model_root).resolve()
    args.output_root = Path(args.output_root).resolve()
    if args.timestamp is None:
        args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    apply_dna_defaults(args)
    sequences = [resolve_sequence(sequence, args.dataset_root) for sequence in args.sequences]
    print("Resolved sequences:", ", ".join(sequences))

    selected = []
    for sequence in sequences:
        run_dir = select_run(sequence, args)
        selected.append((sequence, run_dir))
        print(f"[dry] {sequence}: {run_dir}")
    if args.dry_run:
        return

    safe_state(args.quiet)
    for sequence, run_dir in selected:
        source_path = args.dataset_root / sequence
        out_dir = prepare_output_dir(sequence, run_dir, args)
        metric = read_metric(run_dir, args.iteration)
        dataset = build_dataset_args(args, source_path, run_dir)
        dataset.render_output_dir = str(out_dir)
        dataset.render_video_name = args.video_name
        dataset.render_fps = args.fps
        dataset.render_quality = args.quality
        dataset.render_save_frames = args.save_frames
        dataset.render_make_video = args.make_video
        dataset.render_num_views = args.num_views
        dataset.render_grid_rows = args.grid_rows
        dataset.render_grid_cols = args.grid_cols
        dataset.render_angle_start = args.angle_start
        dataset.render_angle_end = args.angle_end
        dataset.render_reference_index = args.reference_index
        dataset.render_base_yaw_offset = args.base_yaw_offset
        dataset.render_radius = args.radius
        dataset.render_radius_scale = args.radius_scale
        dataset.render_height = args.height
        dataset.render_height_offset = args.height_offset
        dataset.target_mode = args.target_mode
        dataset.target = args.target
        dataset.render_sequence = sequence
        dataset.render_experiment = args.experiment
        dataset.render_run = run_dir.name
        dataset.render_metric = metric
        dataset.render_parameters = serializable_args(args)
        dataset.render_command = " ".join([Path(sys.executable).name] + sys.argv)

        print(f"\n[{sequence}] dataset: {source_path}")
        print(f"[{sequence}] model:   {run_dir}")
        print(f"[{sequence}] output:  {out_dir}")
        render_sets(dataset, args.iteration, build_pipeline_args(args), True, True, True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_circle", action="store_true", help="Render 6x12 grid video")

    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_subdir", default="render_grid_video")
    parser.add_argument("--experiment", default="no_depth_no_split")
    parser.add_argument("--run", default="auto_best")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--gpu", default=GPU_ID)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--video_name", default="render_grid_video.mp4")
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make_video", action=argparse.BooleanOptionalAction, default=True)
    # ===== 相机外参/宫格参数：直接改 default 就能改变每次默认输出 =====
    # 每一帧渲染多少个相机视角；例如 30 表示每个动作帧有 30 个相邻相机。
    parser.add_argument("--num_views", type=int, default=30)
    # 输出宫格视频的行数和列数；要求 grid_rows * grid_cols >= num_views。
    parser.add_argument("--grid_rows", type=int, default=3)
    parser.add_argument("--grid_cols", type=int, default=10)
    # 相机组相对中心方向的角度范围，单位是度。
    # -10 到 10 表示从左到右覆盖 20 度；缩小范围会让相邻相机更近。
    parser.add_argument("--angle_start", type=float, default=-15.0)
    parser.add_argument("--angle_end", type=float, default=15.0)
    # 参考训练相机的索引，用它继承分辨率、内参、默认半径和默认高度。
    parser.add_argument("--reference_index", type=int, default=0)
    # 整组相机的水平朝向偏移；180 通常用于从背面转到正面。
    parser.add_argument("--base_yaw_offset", type=float, default=DEFAULT_BASE_YAW_OFFSET)
    # 相机到人体中心的水平距离。None 表示用参考相机距离；radius_scale 可以缩放默认距离。
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--radius_scale", type=float, default=1) # 调小后相机离得近点
    # 相机高度。None 表示沿用参考相机高度；height_offset 用于整体抬高/降低。
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--height_offset", type=float, default=1)
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--target_mode", choices=["origin", "smpl_center"], default="smpl_center")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    raw_args = parser.parse_args()
    if raw_args.sequences is not None or (not raw_args.source_path and not raw_args.model_path):
        if raw_args.sequences is None:
            raw_args.sequences = DEFAULT_SEQUENCES
        raw_args.render_circle = True
        run_sequence_mode(raw_args, model, pipeline)
        sys.exit(0)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.render_circle)
