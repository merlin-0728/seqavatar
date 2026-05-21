#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

gpu_id = 1
SEQAVATAR_ROOT = Path("/media/image/mxz/human/SeqAvatar")
FS_ROOT = Path("/media/image/mxz/human/FoundationStereo")
DATA_ROOT = SEQAVATAR_ROOT / "DNA-Rendering"

DEFAULT_SEQUENCES = ["0007_04", "0019_10", "0044_11", "0051_09", "0206_04", "0813_05"]
DEFAULT_CKPT = FS_ROOT / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"

sys.path.insert(0, str(FS_ROOT))
from core.foundation_stereo import FoundationStereo  # noqa
from core.utils.utils import InputPadder  # noqa


# -----------------------------
# GPU 设置
# -----------------------------
def set_cuda_device(gpu_id: int):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("FoundationStereo requires CUDA device.")
    print(f"[INFO] Using GPU {gpu_id}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    return device


# -----------------------------
# 工具函数（mask, depth colorize 等）
# -----------------------------
def colorize_depth(depth: np.ndarray):
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = (np.clip(depth, lo, hi) - lo) / (hi - lo + 1e-8)
        vis[valid] = (255 * (1.0 - norm[valid])).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def read_intrinsic_file(path: Path):
    with path.open("r") as f:
        lines = [x.strip() for x in f if x.strip()]
    K = np.array(list(map(float, lines[0].split())), dtype=np.float32).reshape(3, 3)
    baseline = float(lines[1])
    return K, baseline


def parse_frame_view(stem: str):
    m = re.match(r"frame_(\d+)_view_(\d+)$", stem)
    if not m:
        raise RuntimeError(f"Cannot parse frame/view from {stem}")
    return m.group(1), m.group(2)


def load_human_mask(seq: str, stem: str, target_hw):
    frame_id, view_id = parse_frame_view(stem)
    mask_file = DATA_ROOT / seq / "bkgd_masks" / view_id / f"{frame_id}.png"
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    mask = imageio.imread(mask_file)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_human = mask > 128
    # resize
    th, tw = target_hw
    if mask_human.shape != (th, tw):
        mask_human = cv2.resize(mask_human.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask_human, str(mask_file)


def load_model(ckpt_path: Path, device: torch.device):
    cfg_path = ckpt_path.parent / "cfg.yaml"
    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def infer_depth_and_disp(model, left_file, right_file, intrinsic_file, scale, valid_iters, remove_invisible, min_disp, max_depth, device):
    img0 = imageio.imread(left_file)
    img1 = imageio.imread(right_file)
    H_big, W_big = img0.shape[:2]

    img0_resized = cv2.resize(img0, (0,0), fx=scale, fy=scale)
    img1_resized = cv2.resize(img1, (0,0), fx=scale, fy=scale)
    h, w = img0_resized.shape[:2]

    img0_tensor = torch.as_tensor(img0_resized, device=device).float()[None].permute(0,3,1,2)
    img1_tensor = torch.as_tensor(img1_resized, device=device).float()[None].permute(0,3,1,2)

    padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
    img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            disp = model.forward(img0_tensor, img1_tensor, iters=valid_iters, test_mode=True)

    disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(h, w)
    disp_big = cv2.resize(disp, (W_big,H_big), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    if remove_invisible:
        yy, xx = np.meshgrid(np.arange(H_big), np.arange(W_big), indexing='ij')
        invalid = (xx - disp_big) < 0
        disp_big[invalid] = 0.0

    K, baseline = read_intrinsic_file(intrinsic_file)
    valid_disp = np.isfinite(disp_big) & (disp_big > min_disp)
    depth = np.zeros_like(disp_big, dtype=np.float32)
    depth[valid_disp] = (K[0,0]*scale*baseline)/disp_big[valid_disp]

    if max_depth>0:
        valid_depth = (depth>0) & (depth<=max_depth)
        depth[~valid_depth] = 0.0
    return depth, disp_big, valid_disp


# -----------------------------
# 主处理函数
# -----------------------------
def process_sequence(model, seq, split, scale, valid_iters, remove_invisible, min_disp, max_depth, save_disp, max_frames, device):
    root = DATA_ROOT/seq/"render_depth"/split
    rgb_dir = root/"rgb"
    right_dir = root/"right"
    npy_dir = root/"npy"
    vis_dir = root/"vis"
    valid_mask_dir = root/"valid_mask"
    npy_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    valid_mask_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    if max_frames:
        rgb_files = rgb_files[:max_frames]

    for left_file in tqdm(rgb_files, desc=f"{seq} {split} FS depth"):
        stem = left_file.stem
        right_file = right_dir/left_file.name
        k_file = root/"K"/f"{stem}.txt"
        if not right_file.exists():
            raise FileNotFoundError(f"Right image not found: {right_file}")
        if not k_file.exists():
            raise FileNotFoundError(f"Intrinsic file not found: {k_file}")

        left_img = imageio.imread(left_file)
        target_hw = left_img.shape[:2]
        mask_human, mask_path = load_human_mask(seq, stem, target_hw)

        depth, disp, valid_disp = infer_depth_and_disp(
            model, left_file, right_file, k_file, scale, valid_iters, remove_invisible, 1.0, max_depth, device
        )

        valid_final = mask_human & (depth>0)
        depth_masked = np.zeros_like(depth)
        depth_masked[valid_final] = depth[valid_final]

        np.save(npy_dir/f"{stem}.npy", depth_masked)
        imageio.imwrite(vis_dir/f"{stem}.png", colorize_depth(depth_masked))
        np.save(valid_mask_dir/f"{stem}.npy", valid_final.astype(np.bool_))
        imageio.imwrite(valid_mask_dir/f"{stem}.png", (valid_final.astype(np.uint8)*255))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="选择要使用的 GPU 卡")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--split", choices=["train", "novelview"], default="novelview")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--save_disp", action="store_true")
    parser.add_argument("--max_depth", type=float, default=0.0)
    args = parser.parse_args()

    device = set_cuda_device(args.gpu_id)
    model = load_model(DEFAULT_CKPT, device)

    for seq in args.sequences:
        process_sequence(model, seq, args.split, args.scale, args.valid_iters, True, 1.0, args.max_depth, args.save_disp, args.max_frames, device)

    print("[ALL DONE]")


if __name__=="__main__":
    main()
