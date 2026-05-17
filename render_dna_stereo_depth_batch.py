import json
import os
from argparse import ArgumentParser, Namespace

import cv2
import imageio
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import generate_time_steps, safe_state
from utils.smpl_utils import dict_to_device


SEQAVATAR_DIR = os.path.dirname(os.path.abspath(__file__))
DNA_ROOT = os.path.join(SEQAVATAR_DIR, "DNA-Rendering")
OUTPUT_ROOT = os.path.join(SEQAVATAR_DIR, "output", "DNA-Rendering")
DEFAULT_SEQUENCES = ["0007_04", "0019_10", "0044_11", "0051_09", "0206_04", "0813_05"]
DEVICE = "cuda"


def load_cfg_args(model_path):
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing cfg_args: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = eval(f.read(), {"Namespace": Namespace})
    cfg.model_path = os.path.abspath(os.path.join(SEQAVATAR_DIR, cfg.model_path))
    cfg.source_path = os.path.abspath(cfg.source_path)
    return cfg


def camera_pose_from_info(info):
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = info.R.transpose()
    w2c[:3, 3] = info.T
    c2w = np.linalg.inv(w2c)
    return {
        "R": np.asarray(info.R, dtype=np.float32),
        "T": np.asarray(info.T, dtype=np.float32),
        "w2c": w2c.astype(np.float32),
        "c2w": c2w.astype(np.float32),
        "camera_center_world": c2w[:3, 3].astype(np.float32),
    }


def make_right_info(base_info, baseline):
    base_pose = camera_pose_from_info(base_info)
    c2w = base_pose["c2w"]
    shifted_c2w = c2w.copy()
    shifted_c2w[:3, 3] = c2w[:3, 3] + baseline * c2w[:3, 0]
    shifted_w2c = np.linalg.inv(shifted_c2w)
    R = shifted_w2c[:3, :3].transpose().astype(np.float32)
    T = shifted_w2c[:3, 3].astype(np.float32)
    pose = {
        "R": R,
        "T": T,
        "w2c": shifted_w2c.astype(np.float32),
        "c2w": shifted_c2w.astype(np.float32),
        "camera_center_world": shifted_c2w[:3, 3].astype(np.float32),
    }
    right_info = base_info._replace(
        uid=1,
        R=R,
        T=T,
        image_name=f"{base_info.image_name}_right_shift_{baseline:.3f}",
        bkgd_mask=None,
        bound_mask=None,
    )
    return right_info, base_pose, pose


def pose_to_jsonable(pose):
    return {key: value.tolist() for key, value in pose.items()}


def write_k_file(path, K, baseline):
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(f"{float(v):.12f}" for v in K.reshape(-1)) + "\n")
        f.write(f"{float(baseline):.12f}\n")


def write_pose_files(path_prefix, base_info, left_pose, right_pose, baseline):
    payload = {
        "coordinate_convention": {
            "R": "SeqAvatar Camera.R, stored as world-to-camera rotation transposed",
            "T": "world-to-camera translation, matching SeqAvatar Camera.T",
            "w2c": "world-to-camera 4x4 matrix",
            "c2w": "camera-to-world 4x4 matrix",
        },
        "base": {
            "image_name": base_info.image_name,
            "pose_id": int(base_info.pose_id),
            "view_uid": int(base_info.uid),
            "K": np.asarray(base_info.K, dtype=np.float32).tolist(),
            "width": int(base_info.width),
            "height": int(base_info.height),
        },
        "baseline_meters": float(baseline),
        "left": pose_to_jsonable(left_pose),
        "right": pose_to_jsonable(right_pose),
    }
    with open(f"{path_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    np.savez(
        f"{path_prefix}.npz",
        K=np.asarray(base_info.K, dtype=np.float32),
        baseline_meters=np.asarray(baseline, dtype=np.float32),
        left_R=left_pose["R"],
        left_T=left_pose["T"],
        left_w2c=left_pose["w2c"],
        left_c2w=left_pose["c2w"],
        right_R=right_pose["R"],
        right_T=right_pose["T"],
        right_w2c=right_pose["w2c"],
        right_c2w=right_pose["c2w"],
    )


def colorize_depth(depth, valid):
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        vals = depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        norm = (np.clip(depth, lo, hi) - lo) / (hi - lo + 1e-8)
        vis = (255 * (1.0 - norm)).astype(np.uint8)
        vis[~valid] = 0
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def save_render_outputs(render_output, view, background, rgb_path, depth_path, vis_path):
    image = torch.clamp(render_output["render"], 0.0, 1.0)
    depth = render_output["depth"].detach().float().squeeze()
    alpha = render_output["render_alpha"].detach().float().squeeze()

    valid = torch.isfinite(depth) & (alpha > 1e-4)
    if view.bound_mask is not None:
        bound = view.bound_mask[0].to(depth.device) > 0
        image.permute(1, 2, 0)[~bound] = 0 if background.sum().item() == 0 else 1
        valid = valid & bound

    depth = torch.where(valid, depth, torch.zeros_like(depth))
    torchvision.utils.save_image(image.cpu(), rgb_path)

    depth_np = depth.cpu().numpy().astype(np.float32)
    valid_np = valid.cpu().numpy()
    np.save(depth_path, depth_np)
    imageio.imwrite(vis_path, colorize_depth(depth_np, valid_np))


def save_right_render(render_output, rgb_path):
    image = torch.clamp(render_output["render"], 0.0, 1.0)
    torchvision.utils.save_image(image.cpu(), rgb_path)


def make_preview(left_path, right_path, preview_path):
    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)
    if left is None or right is None:
        return
    preview = np.zeros((left.shape[0], left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    preview[:, : left.shape[1]] = left
    preview[:, left.shape[1] :] = right
    for y in range(0, preview.shape[0], 32):
        cv2.line(preview, (0, y), (preview.shape[1] - 1, y), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(preview_path, preview)


def ensure_output_dirs(root):
    dirs = {
        "rgb": os.path.join(root, "rgb"),
        "right": os.path.join(root, "right"),
        "npy": os.path.join(root, "npy"),
        "vis": os.path.join(root, "vis"),
        "pose": os.path.join(root, "pose"),
        "K": os.path.join(root, "K"),
        "preview": os.path.join(root, "preview"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def load_sequence_model(cfg, iteration):
    gaussians = GaussianModel(
        cfg.sh_degree,
        cfg.smpl_type,
        cfg.motion_offset_flag,
        cfg.actor_gender,
        cfg,
    )
    time_steps = generate_time_steps(
        cfg.minimal_time_step,
        cfg.max_time_step,
        1,
        cfg.time_step_num,
        cfg.seq_len,
    )
    scene_info = sceneLoadTypeCallbacks["DNARendering"](
        cfg.source_path,
        cfg.white_background,
        cfg.eval,
        time_steps,
    )
    gaussians.canon_params = dict_to_device(scene_info.canon_params, cfg.data_device)
    gaussians.canon_vertices = torch.tensor(scene_info.canon_vertices).to(cfg.data_device).unsqueeze(0)
    gaussians.smpl_params_dict = dict_to_device(scene_info.smpl_params_dict, cfg.data_device)
    gaussians.cond_dict = dict_to_device(scene_info.cond_dict, cfg.data_device)
    gaussians.get_canon2Tpose_transform(gaussians.canon_params)
    gaussians.load_ply(os.path.join(cfg.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))

    ckpt_path = os.path.join(cfg.model_path, "mlp_ckpt", f"iteration_{iteration}", "ckpt.pth")
    if cfg.motion_offset_flag and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        gaussians.pose_decoder.load_state_dict(ckpt["pose_decoder"])
        gaussians.lweight_offset_decoder.load_state_dict(ckpt["lweight_offset_decoder"])
        if gaussians.non_rigid_flag:
            gaussians.non_rigid_deformer.load_state_dict(ckpt["non_rigid_deformer"])

    return gaussians, scene_info


def render_sequence(seq, iteration, baseline, max_views=None, skip_existing=True, make_previews=True):
    model_path = os.path.join(OUTPUT_ROOT, seq)
    cfg = load_cfg_args(model_path)
    cfg.source_path = os.path.join(DNA_ROOT, seq)
    cfg.model_path = model_path

    print(f"\n=== Rendering {seq} iteration {iteration} ===")
    gaussians, scene_info = load_sequence_model(cfg, iteration)
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=DEVICE)

    views_info = list(scene_info.test_cameras.get("novelview", []))
    if max_views is not None:
        views_info = views_info[:max_views]

    out_root = os.path.join(cfg.source_path, "render_depth", "novelview")
    dirs = ensure_output_dirs(out_root)

    summary = {
        "sequence": seq,
        "iteration": int(iteration),
        "baseline_meters": float(baseline),
        "num_views": len(views_info),
        "source_path": cfg.source_path,
        "model_path": cfg.model_path,
        "note": "Left RGB/depth are freshly rendered from the SeqAvatar checkpoint; existing novelview render PNGs are not used.",
    }
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for base_info in tqdm(views_info, desc=f"{seq} novelview"):
        name = base_info.image_name
        left_rgb = os.path.join(dirs["rgb"], f"{name}.png")
        right_rgb = os.path.join(dirs["right"], f"{name}.png")
        depth_npy = os.path.join(dirs["npy"], f"{name}.npy")
        depth_vis = os.path.join(dirs["vis"], f"{name}.png")
        pose_prefix = os.path.join(dirs["pose"], name)
        k_path = os.path.join(dirs["K"], f"{name}.txt")
        preview_path = os.path.join(dirs["preview"], f"{name}.png")

        expected = [left_rgb, right_rgb, depth_npy, depth_vis, f"{pose_prefix}.json", f"{pose_prefix}.npz", k_path]
        if make_previews:
            expected.append(preview_path)
        if skip_existing and all(os.path.exists(path) for path in expected):
            continue

        right_info, left_pose, right_pose = make_right_info(base_info, baseline)
        left_info = base_info._replace(uid=0)
        left_view, right_view = cameraList_from_camInfos([left_info, right_info], 1.0, cfg)

        with torch.no_grad():
            left_output = render(left_view, gaussians, PipelineParams(ArgumentParser()).extract(Namespace(
                convert_SHs_python=False,
                compute_cov3D_python=True,
                debug=False,
            )), background)
            right_output = render(right_view, gaussians, PipelineParams(ArgumentParser()).extract(Namespace(
                convert_SHs_python=False,
                compute_cov3D_python=True,
                debug=False,
            )), background)

        save_render_outputs(left_output, left_view, background, left_rgb, depth_npy, depth_vis)
        save_right_render(right_output, right_rgb)
        write_pose_files(pose_prefix, base_info, left_pose, right_pose, baseline)
        write_k_file(k_path, np.asarray(base_info.K, dtype=np.float32), baseline)
        if make_previews:
            make_preview(left_rgb, right_rgb, preview_path)

        del left_view, right_view, left_output, right_output
        torch.cuda.empty_cache()


def main():
    os.chdir(SEQAVATAR_DIR)
    parser = ArgumentParser(description="Render virtual right stereo views and left depths for DNA-Rendering novelview sets")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--iteration", type=int, default=25000)
    parser.add_argument("--baseline", type=float, default=0.06)
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_preview", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    for seq in args.sequences:
        render_sequence(
            seq=seq,
            iteration=args.iteration,
            baseline=args.baseline,
            max_views=args.max_views,
            skip_existing=not args.overwrite,
            make_previews=not args.no_preview,
        )


if __name__ == "__main__":
    main()
