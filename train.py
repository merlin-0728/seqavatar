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

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, l1_loss_masked, l2_loss_masked, ssim, full_aiap_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_rotation, inverse_sigmoid
import json
import numpy as np
import pickle
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import shutil
from torchvision.ops import masks_to_boxes
import time
torch.backends.cudnn.enabled = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))


# ------------------ Depth loss / depth-guided split helpers ------------------
def as_hw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert depth tensor to HxW."""
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor.squeeze()


def load_train_fs_depth(dataset, image_name):
    """
    只读取 filtered_npy 中的深度。
    如果 filtered_npy 没有当前视角对应文件，则不使用该视角深度。
    """
    depth_path_filtered = os.path.join(
        dataset.source_path,
        "render_depth",
        "train",
        "filtered_npy",
        image_name + ".npy"
    )

    if os.path.exists(depth_path_filtered):
        return np.load(depth_path_filtered).astype(np.float32)

    return None


@torch.no_grad()
def select_projected_gaussians_by_depth(view, render_pkg, split_mask: torch.Tensor, error_map: torch.Tensor, max_points: int):
    """
    Select Gaussians whose projected pixels fall into high depth-error regions.
    """
    points = render_pkg["deformed_means3D"].detach()

    if points.ndim != 2 or points.shape[0] == 0 or not torch.any(split_mask):
        return torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device), 0

    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)

    clip = points_h @ view.full_proj_transform
    w = clip[:, 3]
    valid_w = torch.abs(w) > 1e-8
    denom = torch.where(valid_w, w, torch.ones_like(w) * 1e-8)
    ndc = clip[:, :3] / denom[:, None]

    width = int(view.image_width)
    height = int(view.image_height)

    px = ((ndc[:, 0] + 1.0) * 0.5 * width).long()
    py = ((ndc[:, 1] + 1.0) * 0.5 * height).long()

    inside = valid_w & (px >= 0) & (px < width) & (py >= 0) & (py < height)

    if "visibility_filter" in render_pkg:
        inside = inside & render_pkg["visibility_filter"].detach()

    selected = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)

    idx = torch.where(inside)[0]
    if idx.numel() == 0:
        return selected, 0

    pix_x = px[idx]
    pix_y = py[idx]
    hit = split_mask[pix_y, pix_x]
    idx = idx[hit]

    if idx.numel() == 0:
        return selected, 0

    if max_points > 0 and idx.numel() > max_points:
        scores = error_map[py[idx], px[idx]]
        _, order = torch.topk(scores, k=max_points, largest=True)
        idx = idx[order]

    selected[idx] = True
    return selected, int(idx.numel())


@torch.no_grad()
def split_selected_gaussians_by_depth(gaussians: GaussianModel, selected: torch.Tensor, split_count: int, split_scale: float):
    """
    Add extra Gaussians around selected parents along the largest local scale axis.
    """
    idx = torch.where(selected)[0]
    if idx.numel() == 0 or split_count <= 0:
        return 0

    parent_xyz = gaussians._xyz.detach()[idx]
    parent_scale = gaussians.get_scaling.detach()[idx]
    parent_rot = gaussians._rotation.detach()[idx]

    rots = build_rotation(parent_rot)
    largest_axis = torch.argmax(parent_scale, dim=1)

    axis = torch.zeros_like(parent_xyz)
    axis[torch.arange(idx.numel(), device=idx.device), largest_axis] = 1.0

    world_axis = torch.bmm(rots, axis[..., None]).squeeze(-1)
    world_axis = world_axis / torch.clamp(
        torch.linalg.norm(world_axis, dim=1, keepdim=True),
        min=1e-8
    )

    offsets = torch.linspace(
        -1.0, 1.0,
        steps=split_count + 2,
        device=idx.device,
        dtype=parent_xyz.dtype
    )[1:-1]

    step = parent_scale.max(dim=1).values * split_scale
    new_xyz = parent_xyz.repeat_interleave(split_count, dim=0)
    new_xyz = new_xyz + (
        world_axis.repeat_interleave(split_count, dim=0)
        * (step[:, None] * offsets[None, :]).reshape(-1, 1)
    )

    shrink = 0.8 * (split_count + 1)
    new_scaling = gaussians.scaling_inverse_activation(
        parent_scale.repeat_interleave(split_count, dim=0) / shrink
    )

    new_rotation = gaussians._rotation.detach()[idx].repeat_interleave(split_count, dim=0)
    new_features_dc = gaussians._features_dc.detach()[idx].repeat_interleave(split_count, dim=0)
    new_features_rest = gaussians._features_rest.detach()[idx].repeat_interleave(split_count, dim=0)

    parent_opacity = gaussians.get_opacity.detach()[idx].repeat_interleave(split_count, dim=0)
    new_opacity_value = torch.clamp(parent_opacity / (split_count + 1), min=1e-4, max=1.0 - 1e-4)
    new_opacity = inverse_sigmoid(new_opacity_value)

    gaussians.densification_postfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
    )

    return int(new_xyz.shape[0])
# ---------------------------------------------------------------------------

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    Ll1_loss_for_log = 0.0
    mask_loss_for_log = 0.0
    ssim_loss_for_log = 0.0
    lpips_loss_for_log = 0.0
    depth_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training")
    first_iter += 1

    elapsed_time = 0

    print("========== Depth Ablation Settings ==========")
    print("Enable depth loss:", getattr(opt, "enable_depth_loss", False))
    print("Enable gaussian split:", getattr(opt, "enable_gaussian_split", False))
    print("Depth loss start:", getattr(opt, "depth_loss_start", None))
    print("Depth split start/end:", getattr(opt, "depth_split_start", None), getattr(opt, "depth_split_end", None))
    print("Lambda depth:", getattr(opt, "lambda_depth", None))
    print("Depth split threshold:", getattr(opt, "depth_split_threshold", None))
    print("=============================================")

    for iteration in range(first_iter, opt.iterations + 1):
        use_depth_loss = (
            getattr(opt, "enable_depth_loss", False)
            and iteration >= getattr(opt, "depth_loss_start", 15000)
        )
        use_depth_split = (
            getattr(opt, "enable_gaussian_split", False)
            and iteration >= getattr(opt, "depth_split_start", 17000)
            and iteration < getattr(opt, "depth_split_end", 18000)
        )
        need_depth = use_depth_loss or use_depth_split

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree.
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        start_time = time.time()

        # Pick a random camera.
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render.
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        alpha = render_pkg["render_alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # Original RGB / alpha / SSIM / LPIPS loss.
        gt_image = viewpoint_cam.original_image.cuda()
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()
        bound_mask = viewpoint_cam.bound_mask.cuda()

        x1, y1, x2, y2 = masks_to_boxes(bound_mask).int().squeeze(0)
        img_pred_rect = image[:, y1:y2 + 1, x1:x2 + 1].unsqueeze(0)
        img_gt_rect = gt_image[:, y1:y2 + 1, x1:x2 + 1].unsqueeze(0)
        bound_mask = bound_mask[0] == 1

        Ll1 = l1_loss_masked(image, gt_image, bound_mask)
        alpha_loss = l2_loss_masked(alpha, bkgd_mask, bound_mask)
        ssim_loss = ssim(img_pred_rect, img_gt_rect)
        lpips_loss = loss_fn_vgg(img_pred_rect, img_gt_rect).squeeze()

        loss = (
            opt.l1_loss_w * Ll1
            + 0.1 * alpha_loss
            + opt.ssim_loss_w * (1.0 - ssim_loss)
            + opt.lpips_loss_w * lpips_loss
        )

        # Depth loss and depth split mask.
        loss_depth_raw = torch.zeros((), device=image.device)
        depth_error_map = None
        depth_split_mask = None

        if need_depth:
            fs_depth_np = load_train_fs_depth(dataset, viewpoint_cam.image_name)
            if fs_depth_np is not None and "depth" in render_pkg:
                fs_depth = torch.from_numpy(fs_depth_np).to(device=image.device, dtype=torch.float32)
                render_depth = as_hw(render_pkg["depth"])

                if fs_depth.shape != render_depth.shape:
                    fs_depth = F.interpolate(
                        fs_depth[None, None],
                        size=render_depth.shape,
                        mode="nearest",
                    )[0, 0]

                valid_depth = (
                    torch.isfinite(render_depth)
                    & torch.isfinite(fs_depth)
                    & (render_depth > 0)
                    & (fs_depth > 0)
                    & bound_mask
                )

                if torch.any(valid_depth):
                    depth_error_map = torch.zeros_like(render_depth)
                    depth_error_map[valid_depth] = torch.abs(render_depth[valid_depth] - fs_depth[valid_depth])
                    depth_split_mask = valid_depth & (depth_error_map > opt.depth_split_threshold)

                    if use_depth_loss:
                        loss_depth_raw = depth_error_map[valid_depth].mean()
                        loss = loss + opt.lambda_depth * loss_depth_raw

        # Original AIAP loss.
        loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(
            scene.gaussians.get_xyz,
            render_pkg["deformed_means3D"],
            scene.gaussians.get_covariance(),
            render_pkg["deformed_cov3D"],
        )
        loss = loss + opt.iospos_w * loss_aiap_xyz + opt.ioscov_w * loss_aiap_cov

        loss.backward()

        end_time = time.time()
        elapsed_time += (end_time - start_time)

        if iteration in testing_iterations:
            print("[Elapsed time]: ", elapsed_time)

        iter_end.record()

        with torch.no_grad():
            # Progress bar.
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            Ll1_loss_for_log = 0.4 * Ll1.item() + 0.6 * Ll1_loss_for_log
            mask_loss_for_log = 0.4 * alpha_loss.item() + 0.6 * mask_loss_for_log
            ssim_loss_for_log = 0.4 * ssim_loss.item() + 0.6 * ssim_loss_for_log
            lpips_loss_for_log = 0.4 * lpips_loss.item() + 0.6 * lpips_loss_for_log
            depth_loss_for_log = 0.4 * loss_depth_raw.item() + 0.6 * depth_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "#pts": gaussians._xyz.shape[0],
                    "Ll1 Loss": f"{Ll1_loss_for_log:.3f}",
                    "mask Loss": f"{mask_loss_for_log:.2f}",
                    "ssim": f"{ssim_loss_for_log:.2f}",
                    "lpips": f"{lpips_loss_for_log:.2f}",
                    "depth_raw": f"{depth_loss_for_log:.4f}",
                    "Dloss": int(use_depth_loss),
                    "Dsplit": int(use_depth_split),
                })
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                saving_iterations,
            )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            start_time = time.time()

            # Original densification. Keep it unchanged.
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                    and len(gaussians.get_xyz) < 120000
                ):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Extra depth-guided split. It is independent of densify_until_iter,
            # so it can run in 17000-18000 even if original densification stopped earlier.
            if (
                use_depth_split
                and depth_split_mask is not None
                and depth_error_map is not None
                and iteration % opt.depth_split_interval == 0
                and len(gaussians.get_xyz) < opt.max_depth_split_points
            ):
                selected, selected_count = select_projected_gaussians_by_depth(
                    viewpoint_cam,
                    render_pkg,
                    depth_split_mask.detach(),
                    depth_error_map.detach(),
                    opt.max_depth_split_points_per_frame,
                )
                if selected_count > 0:
                    new_points = split_selected_gaussians_by_depth(
                        gaussians,
                        selected,
                        opt.depth_split_count,
                        opt.depth_split_scale,
                    )
                    if new_points > 0:
                        print(
                            f"[DEPTH-SPLIT] iter={iteration}, "
                            f"selected={selected_count}, "
                            f"new={new_points}, "
                            f"total={len(gaussians.get_xyz)}"
                        )

            # Optimizer step.
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                gaussians.mlp_optimizer.step()
                gaussians.mlp_optimizer.zero_grad()
                gaussians.mlp_scheduler.step()

            end_time = time.time()
            elapsed_time += (end_time - start_time)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", args.exp_name)

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, saving_iterations):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        smpl_rot = {}
        validation_configs = [{'name': 'train', 'cameras' : scene.getTrainCameras()}]
        smpl_rot['train'] = {}
        for key in scene.getTestCameras().keys():
            validation_configs.append({'name': key, 'cameras' : scene.getTestCameras()[key]})
            smpl_rot[key] = {}        
        for config in validation_configs:
            if config['name'] != 'train' and config['cameras'] and len(config['cameras']) > 0: 
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                ssims, psnrs, lpipss, img_names, full_dict, per_view_dict = [], [], [], [], {}, {}
                    
                for idx, viewpoint in enumerate(config['cameras']):
                    smpl_rot[config['name']][viewpoint.pose_id] = {}
                    render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, return_smpl_rot=True)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    cur_psnr = psnr(image, gt_image).mean().double()
                    psnr_test += cur_psnr
                    cur_ssim = ssim(image, gt_image).mean().double()
                    ssim_test += cur_ssim
                    cur_lpips = loss_fn_vgg(image, gt_image).mean().double()
                    lpips_test += cur_lpips

                    img_names.append(viewpoint.image_name)
                    psnrs.append(cur_psnr)
                    ssims.append(cur_ssim)
                    lpipss.append(cur_lpips)

                    smpl_rot[config['name']][viewpoint.pose_id]['d_nonrigid'] = render_output['d_nonrigid']
                    smpl_rot[config['name']][viewpoint.pose_id]['transforms'] = render_output['transforms']
                    smpl_rot[config['name']][viewpoint.pose_id]['translation'] = render_output['translation']

                full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                                  "PSNR": torch.tensor(psnrs).mean().item(),
                                  "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), img_names)},
                                      "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), img_names)},
                                      "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), img_names)}})
                os.makedirs(scene.model_path + '/metrics', exist_ok=True)
                with open(scene.model_path + '/metrics/results_' + config['name'] + '_' + str(iteration) + '.json', 'w') as fp:
                    json.dump(full_dict, fp, indent=True)
                with open(scene.model_path + '/metrics/per_view' + config['name'] + '_' + str(iteration) + '.json', 'w') as fp:
                    json.dump(per_view_dict, fp, indent=True)

                l1_test /= len(config['cameras']) 
                psnr_test /= len(config['cameras'])   
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])      
                print("\n[ITER {}] Evaluating {} #{}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], len(config['cameras']), l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # Store data (serialize)
        if iteration in saving_iterations:
            save_path = os.path.join(scene.model_path, 'smpl_rot', f'iteration_{iteration}')
            os.makedirs(save_path, exist_ok=True)
            with open(save_path+"/smpl_rot.pickle", 'wb') as handle:
                pickle.dump(smpl_rot, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--enable_depth_loss", action="store_true", help="Use training-set depth loss after depth_loss_start")
    parser.add_argument("--enable_gaussian_split", action="store_true", help="Use depth-guided Gaussian split in [depth_split_start, depth_split_end)")
    parser.add_argument("--depth_loss_start", type=int, default=15000)
    parser.add_argument("--depth_split_start", type=int, default=17000)
    parser.add_argument("--depth_split_end", type=int, default=18000)
    parser.add_argument("--lambda_depth", type=float, default=0.1)
    parser.add_argument("--depth_split_threshold", type=float, default=0.02)
    parser.add_argument("--depth_split_interval", type=int, default=100)
    parser.add_argument("--depth_split_count", type=int, default=3)
    parser.add_argument("--depth_split_scale", type=float, default=0.5)
    parser.add_argument("--max_depth_split_points_per_frame", type=int, default=8)
    parser.add_argument("--max_depth_split_points", type=int, default=120000)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 15_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 15_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--mono_test", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    # Attach custom depth-ablation arguments to OptimizationParams output.
    opt_args.enable_depth_loss = args.enable_depth_loss
    opt_args.enable_gaussian_split = args.enable_gaussian_split
    opt_args.depth_loss_start = args.depth_loss_start
    opt_args.depth_split_start = args.depth_split_start
    opt_args.depth_split_end = args.depth_split_end
    opt_args.lambda_depth = args.lambda_depth
    opt_args.depth_split_threshold = args.depth_split_threshold
    opt_args.depth_split_interval = args.depth_split_interval
    opt_args.depth_split_count = args.depth_split_count
    opt_args.depth_split_scale = args.depth_split_scale
    opt_args.max_depth_split_points_per_frame = args.max_depth_split_points_per_frame
    opt_args.max_depth_split_points = args.max_depth_split_points

    training(
        dataset_args,
        opt_args,
        pipe_args,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")