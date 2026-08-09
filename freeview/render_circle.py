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
import imageio.v2 as imageio
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix_refine
from visual_effects import compose_torch_render, visual_metadata

from freeview import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_ROOT,
    build_dataset_args,
    build_pipeline_args,
    prepare_output_dir,
    read_metric,
    resolve_sequence,
    select_run,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "freeview" / "DNA-Rendering"
DEFAULT_SEQUENCES = ["0007", "0019", "0044", "0051", "0206", "0813"]


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
        "SeqAvatar render_circle render",
        "",
        f"sequence: {metadata.get('sequence')}",
        f"dataset: {metadata.get('dataset')}",
        f"model: {metadata.get('model')}",
        f"experiment: {metadata.get('experiment')}",
        f"run: {metadata.get('run')}",
        f"iteration: {metadata.get('iteration')}",
        "",
        f"reference_split: {metadata.get('reference_split')}",
        f"reference_camera: {metadata.get('reference_camera')}",
        f"pose_id: {metadata.get('pose_id')}",
        f"num_views: {metadata.get('num_views')}",
        f"angle_start: {metadata.get('angle_start')}",
        f"angle_end: {metadata.get('angle_end')}",
        f"radius: {metadata.get('radius')}",
        f"radius_scale: {metadata.get('radius_scale')}",
        f"height: {metadata.get('height')}",
        f"height_offset: {metadata.get('height_offset')}",
        f"base_angle: {metadata.get('base_angle')}",
        "",
        f"video: {metadata.get('video')}",
        f"frames_dir: {metadata.get('frames_dir')}",
        "",
        "command:",
        f"  {metadata.get('command')}",
    ]
    with open(os.path.join(out_dir, "render_info.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

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
    radius = np.linalg.norm(np.array([cam_center[0], cam_center[2]])) * getattr(dataset, "render_radius_scale", 1.0)
    # 保持参考高度，并允许整体上下平移相机。
    # 负数会让相机整体向下，减少从头顶往下看的视角。
    height = cam_center[1] + getattr(dataset, "render_height_offset", 0.0)
    
    # 2. 设置保存路径
    output_dir = getattr(dataset, "render_output_dir", None)
    if output_dir is None:
        output_dir = os.path.join(dataset.model_path, data_name, "ours_{}".format(iteration), "renders_circle_72")
    else:
        output_dir = str(output_dir)

    render_path = os.path.join(output_dir, "frames")
    makedirs(output_dir, exist_ok=True)
    makedirs(render_path, exist_ok=True)

    video_name = getattr(dataset, "render_video_name", "render_circle.mp4")
    video_path = os.path.join(output_dir, video_name)
    fps = getattr(dataset, "render_fps", 24)
    quality = getattr(dataset, "render_quality", 8)
    make_video = getattr(dataset, "render_make_video", True)
    save_frames = getattr(dataset, "render_save_frames", True)

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
    writer = None
    if make_video:
        writer = imageio.get_writer(video_path, fps=fps, quality=quality, macro_block_size=None)

    print(f"Rendering {len(render_cameras)} frames to {output_dir}...")
    for idx, view in enumerate(tqdm(render_cameras, desc="Rendering Circle")):
        
        render_output = render(view, gaussians, pipeline, background, 
                               transforms=transforms, 
                               translation=translation, 
                               d_nonrigid=d_nonrigid)
        
        rendering = render_output["render"]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        frame = compose_torch_render(
            rendering,
            render_output.get("render_alpha"),
            background_path=getattr(dataset, "visual_background_path", None),
            enable_background=getattr(dataset, "visual_background", True),
            enable_shadow=getattr(dataset, "visual_shadow", True),
            background_fit=getattr(dataset, "visual_background_fit", "cover"),
            shadow_offset=getattr(dataset, "visual_shadow_offset", [46, 30]),
            shadow_blur=getattr(dataset, "visual_shadow_blur", 14.0),
            shadow_opacity=getattr(dataset, "visual_shadow_opacity", 0.33),
        )
        if save_frames:
            imageio.imwrite(os.path.join(render_path, view.image_name + ".png"), frame)
        if writer is not None:
            writer.append_data(frame)

    if writer is not None:
        writer.close()

    metadata = {
        "sequence": getattr(dataset, "render_sequence", None),
        "dataset": dataset.source_path,
        "model": dataset.model_path,
        "experiment": getattr(dataset, "render_experiment", None),
        "run": getattr(dataset, "render_run", None),
        "iteration": iteration,
        "renderer": "render_circle",
        "reference_split": data_name,
        "reference_camera": base_cam.image_name,
        "pose_id": int(pose_id),
        "num_views": num_views,
        "angle_start": float(angles[0]),
        "angle_end": float(angles[-1]),
        "radius": float(radius),
        "radius_scale": getattr(dataset, "render_radius_scale", 1.0),
        "height": float(height),
        "height_offset": getattr(dataset, "render_height_offset", 0.0),
        "base_angle": float(base_angle),
        "fps": fps,
        "video": video_path if make_video else None,
        "frames_dir": render_path if save_frames else None,
        "metric": getattr(dataset, "render_metric", None),
        "parameters": getattr(dataset, "render_parameters", None),
        "visual_effects": getattr(dataset, "visual_effects", None),
        "command": getattr(dataset, "render_command", " ".join([Path(sys.executable).name] + sys.argv)),
    }
    write_metadata(output_dir, metadata)

    print(f"Done. Output saved to: {output_dir}")

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
        dataset.render_radius_scale = args.radius_scale
        dataset.render_height_offset = args.height_offset
        dataset.visual_background = args.visual_background
        dataset.visual_background_path = str(args.visual_background_path) if args.visual_background_path else None
        dataset.visual_background_fit = args.visual_background_fit
        dataset.visual_shadow = args.visual_shadow
        dataset.visual_shadow_offset = args.visual_shadow_offset
        dataset.visual_shadow_blur = args.visual_shadow_blur
        dataset.visual_shadow_opacity = args.visual_shadow_opacity
        dataset.visual_effects = visual_metadata(args)
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
    # 添加一个新参数来触发圆弧渲染
    parser.add_argument("--render_circle", action="store_true", help="Render a 60-degree arc around the subject")

    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_subdir", default="render_circle")
    parser.add_argument("--experiment", default="part_moe_leg")
    parser.add_argument("--run", default="latest")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--gpu", default=GPU_ID)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--video_name", default="render_circle.mp4")
    parser.add_argument("--save_frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radius_scale", type=float, default=0.75)
    parser.add_argument("--height_offset", type=float, default=-0.6)
    parser.add_argument("--visual_background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual_background_path", type=Path, default=None)
    parser.add_argument("--visual_background_fit", choices=["cover", "contain"], default="cover")
    parser.add_argument("--visual_shadow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual_shadow_offset", nargs=2, type=int, default=[46, 30])
    parser.add_argument("--visual_shadow_blur", type=float, default=14.0)
    parser.add_argument("--visual_shadow_opacity", type=float, default=0.33)
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
