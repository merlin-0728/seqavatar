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
from utils.general_utils import safe_state
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

# --- WANDB 新增：引入库 ---
import wandb

torch.backends.cudnn.enabled = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
    scene = Scene(dataset, gaussians)
    print(
        f"Densify settings: grad_threshold={opt.densify_grad_threshold}, "
        f"prune_opacity_threshold={opt.prune_opacity_threshold}"
    )
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log, Ll1_loss_for_log, mask_loss_for_log, ssim_loss_for_log, lpips_loss_for_log = 0.0, 0.0, 0.0, 0.0, 0.0
    total_train_iters = opt.iterations - first_iter
    progress_bar = tqdm(total=total_train_iters, desc="Training")
    first_iter += 1
    progress_steps = 0
    last_logged_percent = -1

    elapsed_time = 0
    for iteration in range(first_iter, opt.iterations + 1):  
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if gaussians.non_rigid_flag and hasattr(gaussians.non_rigid_deformer, "set_phase"):
            phase1_active = opt.phase1_vggt_iters > 0 and iteration <= opt.phase1_vggt_iters
            gaussians.non_rigid_deformer.set_phase(phase1_active)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        start_time = time.time()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()
        bound_mask = viewpoint_cam.bound_mask.cuda()
        
        x1, y1, x2, y2 = masks_to_boxes(bound_mask).int().squeeze(0)
        img_pred_rect = image[:, y1:y2+1, x1:x2+1].unsqueeze(0)
        img_gt_rect = gt_image[:, y1:y2+1, x1:x2+1].unsqueeze(0)
        bound_mask = bound_mask[0] == 1
        Ll1 = l1_loss_masked(image, gt_image, bound_mask)
        alpha_loss = l2_loss_masked(alpha, bkgd_mask, bound_mask)
        
        ssim_loss = ssim(img_pred_rect, img_gt_rect)
        lpips_loss = loss_fn_vgg(img_pred_rect, img_gt_rect).squeeze()

        loss = opt.l1_loss_w * Ll1 + 0.1 * alpha_loss + opt.ssim_loss_w * (1.0 - ssim_loss) + opt.lpips_loss_w * lpips_loss

        loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians.get_xyz, render_pkg["deformed_means3D"], scene.gaussians.get_covariance(), render_pkg["deformed_cov3D"])
        loss = loss + opt.iospos_w * loss_aiap_xyz + opt.ioscov_w * loss_aiap_cov

        loss_vggt = torch.zeros((), dtype=loss.dtype, device=loss.device)
        if opt.lambda_vggt > 0 and gaussians.has_vggt_points and "canonical_means3D" in render_pkg:
            vggt_mask = gaussians.vggt_mask
            canonical_pred = render_pkg["canonical_means3D"]
            if canonical_pred.ndim == 3:
                canonical_pred = canonical_pred.squeeze(0)
            if vggt_mask is not None and vggt_mask.any() and gaussians._vggt_target.shape[0] == canonical_pred.shape[0]:
                loss_vggt = F.mse_loss(canonical_pred[vggt_mask], gaussians._vggt_target[vggt_mask])
                loss = loss + opt.lambda_vggt * loss_vggt
        
        # ==========================================
        # 🛡️ 【防爆盾 A】：检查 Loss 本身
        # ==========================================
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ 警告: 第 {iteration} 步的 Loss 出现 NaN/Inf！已跳过该步梯度更新以保护 CUDA 算子。")
            gaussians.optimizer.zero_grad(set_to_none=True)
            if gaussians.mlp_optimizer:
                gaussians.mlp_optimizer.zero_grad()
            continue
            
        loss.backward()

        # ==========================================
        # 🛡️ 【防爆盾 B】：终极物理级梯度清洗 (Gradient Scrubber)
        # ==========================================
        # 1. 强制清洗基础高斯参数的异常梯度 (NaN/Inf 全部置 0)
        for param_group in gaussians.optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. 强制清洗 MLP 网络的异常梯度 (这是 CUBLAS 崩溃的直接源头)
        if gaussians.mlp_optimizer is not None:
            for param_group in gaussians.mlp_optimizer.param_groups:
                for p in param_group['params']:
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. 再加上全局梯度裁剪，防止更新步子太大
        torch.nn.utils.clip_grad_norm_(gaussians.optimizer.param_groups[0]['params'], max_norm=1.0)
        if gaussians.mlp_optimizer is not None:
            for param_group in gaussians.mlp_optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=1.0)
        # ==========================================

        end_time = time.time()
        elapsed_time += (end_time - start_time)

        if (iteration in testing_iterations):
            print("[Elapsed time]: ", elapsed_time) 

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            Ll1_loss_for_log = 0.4 * Ll1.item() + 0.6 * Ll1_loss_for_log
            mask_loss_for_log = 0.4 * alpha_loss.item() + 0.6 * mask_loss_for_log
            ssim_loss_for_log = 0.4 * ssim_loss.item() + 0.6 * ssim_loss_for_log
            lpips_loss_for_log = 0.4 * lpips_loss.item() + 0.6 * lpips_loss_for_log
            
            if iteration % 10 == 0: 
                wandb.log({
                    "Train/Total_Loss": loss.item(),
                    "Train/L1_Loss": Ll1.item(),
                    "Train/Alpha_Loss": alpha_loss.item(),
                    "Train/SSIM_Loss": ssim_loss.item(),
                    "Train/LPIPS_Loss": lpips_loss.item(),
                    "Train/VGGT_Geo_Loss": loss_vggt.item(),
                    "Train/AIAP_XYZ_Loss": loss_aiap_xyz.item(),
                    "Train/AIAP_COV_Loss": loss_aiap_cov.item(),
                    "Train/Num_Points": gaussians._xyz.shape[0],
                    "Train/Phase1_VGGT_Only": int(opt.phase1_vggt_iters > 0 and iteration <= opt.phase1_vggt_iters),
                    "iteration": iteration
                })
            
            completed_iters = iteration - first_iter + 1
            curr_percent = int(100 * completed_iters / total_train_iters)
            if curr_percent > last_logged_percent:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}", "mask Loss": f"{mask_loss_for_log:.{2}f}",
                                          "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}",
                                          "vggt": f"{loss_vggt.item():.{3}f}"})
                progress_bar.update(completed_iters - progress_steps)
                progress_steps = completed_iters
                last_logged_percent = curr_percent
            if iteration == opt.iterations:
                if progress_steps < total_train_iters:
                    progress_bar.update(total_train_iters - progress_steps)
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), saving_iterations)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            start_time = time.time()
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and len(gaussians.get_xyz) < 120000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.prune_opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                    )
                
                # [3DHGS]: 这里的重置逻辑我们已经在 gaussian_model 里用 torch.min 适配了双通道
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                gaussians.mlp_optimizer.step()
                gaussians.mlp_optimizer.zero_grad()
                gaussians.mlp_scheduler.step()

            end_time = time.time()
            elapsed_time += (end_time - start_time)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", args.exp_name)

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

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
                    
                wandb.log({
                    f"Eval_{config['name']}/L1_Loss": l1_test,
                    f"Eval_{config['name']}/PSNR": psnr_test,
                    f"Eval_{config['name']}/SSIM": ssim_test,
                    f"Eval_{config['name']}/LPIPS": lpips_test,
                    "iteration": iteration
                })

        if iteration in saving_iterations:
            save_path = os.path.join(scene.model_path, 'smpl_rot', f'iteration_{iteration}')
            os.makedirs(save_path, exist_ok=True)
            with open(save_path+"/smpl_rot.pickle", 'wb') as handle:
                pickle.dump(smpl_rot, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if tb_writer:
            # [3DHGS 修改]: 这里因为 get_opacity 返回的是 (N, 2)，而直方图绘制只能接受 1D 数组，所以加了一个 .view(-1) 将其打平。
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity.view(-1), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
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
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    wandb.init(config=vars(args))
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
    wandb.finish()
