#!/usr/bin/env python3
"""Subset metrics and attention diagnostics for flow_view_token_gate.

This script is read-only: it does not train, render to disk, or modify outputs.
It compares existing baseline and flow_view_token_gate PNG renders on:
  1) high-motion novel frames,
  2) foreground boundary bands,
and it can forward the trained flow_view_token_gate model to inspect view-token
attention weights.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


SEQ_DEFAULT = "0007_04 0019_10 0044_11 0051_09 0206_04 0813_05"
FLOW_RUN_DEFAULT = "20260707_130935"
FLOW_CACHE_NAME = "dna_train_farneback_viewtoken_s0p250_ms100_v24_fd4.npz"


def parse_list(raw: str) -> List[str]:
    return [x for x in re.split(r"[\s,]+", raw.strip()) if x]


def parse_frame_view(name: str) -> Tuple[int, int]:
    match = re.search(r"frame_(\d+)_view_(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse frame/view from {name}")
    return int(match.group(1)), int(match.group(2))


def read_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return TF.to_tensor(image).float()


def read_mask(path: Path, size_hw: Tuple[int, int]) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    if mask.size != (size_hw[1], size_hw[0]):
        mask = mask.resize((size_hw[1], size_hw[0]), Image.Resampling.NEAREST)
    arr = np.asarray(mask) > 0
    return torch.from_numpy(arr)


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def erode_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def boundary_band(mask: torch.Tensor, radius: int) -> torch.Tensor:
    arr = mask.cpu().numpy().astype(bool)
    band = dilate_bool(arr, radius) & ~erode_bool(arr, radius)
    return torch.from_numpy(band)


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def mean_dict(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(items)
    if not items:
        return {}
    keys = sorted(items[0].keys())
    return {k: float(np.mean([x[k] for x in items if k in x])) for k in keys}


def load_lpips(device: torch.device):
    import lpips

    return lpips.LPIPS(net="vgg").to(device).eval()


def image_metrics(pred: torch.Tensor, gt: torch.Tensor, loss_fn, device: torch.device) -> Dict[str, float]:
    from utils.loss_utils import ssim

    pred = pred.clamp(0.0, 1.0)
    gt = gt.clamp(0.0, 1.0)
    mse = float(((pred - gt) ** 2).mean().item())
    with torch.no_grad():
        ssim_v = float(ssim(pred.to(device), gt.to(device)).mean().item())
        lpips_v = float(loss_fn(pred.to(device), gt.to(device)).mean().item())
    return {"psnr": psnr_from_mse(mse), "ssim": ssim_v, "lpips": lpips_v}


def masked_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    loss_fn,
    device: torch.device,
) -> Dict[str, float]:
    from utils.loss_utils import ssim

    pred = pred.clamp(0.0, 1.0)
    gt = gt.clamp(0.0, 1.0)
    mask = mask.to(torch.bool)
    if int(mask.sum().item()) == 0:
        return {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan"), "pixel_count": 0.0}
    diff = pred[:, mask] - gt[:, mask]
    mse = float((diff * diff).mean().item())
    mask3 = mask.unsqueeze(0).expand_as(pred)
    pred_focus = torch.where(mask3, pred, gt)
    with torch.no_grad():
        ssim_v = float(ssim(pred_focus.to(device), gt.to(device)).mean().item())
        lpips_v = float(loss_fn(pred_focus.to(device), gt.to(device)).mean().item())
    return {
        "psnr": psnr_from_mse(mse),
        "ssim": ssim_v,
        "lpips": lpips_v,
        "pixel_count": float(mask.sum().item()),
    }


def high_motion_scores(data_root: Path, sequence: str) -> Dict[int, float]:
    cache = data_root / sequence / "flow_features" / FLOW_CACHE_NAME
    if not cache.exists():
        raise FileNotFoundError(cache)
    features = np.load(cache)["features"]
    mag = features[..., 2]
    conf = features[..., 3]
    scores = np.mean(mag * conf, axis=(1, 2))
    return {idx: float(v) for idx, v in enumerate(scores)}


def evaluate_subset_sequence(args, sequence: str, loss_fn, device: torch.device) -> Dict[str, object]:
    baseline_dir = args.output_root / sequence / "novelview" / "ours_25000" / "renders"
    flow_dir = args.output_root / sequence / "flow_view_token_gate" / args.flow_run / "novelview" / "ours_25000" / "renders"
    gt_dir = args.output_root / sequence / "flow_view_token_gate" / args.flow_run / "novelview" / "ours_25000" / "gt"
    if not gt_dir.exists():
        gt_dir = args.output_root / sequence / "novelview" / "ours_25000" / "gt"
    if not (baseline_dir.exists() and flow_dir.exists() and gt_dir.exists()):
        raise FileNotFoundError(f"Missing renders for {sequence}: {baseline_dir}, {flow_dir}, {gt_dir}")

    names = sorted(p.name for p in flow_dir.glob("*.png") if (baseline_dir / p.name).exists() and (gt_dir / p.name).exists())
    motion = high_motion_scores(args.data_root, sequence)
    ranked = sorted(names, key=lambda n: motion.get(parse_frame_view(n)[0], -1.0), reverse=True)
    high_count = max(1, int(round(len(ranked) * args.high_motion_fraction)))
    high_names = set(ranked[:high_count])

    out: Dict[str, object] = {
        "sequence": sequence,
        "image_count": len(names),
        "high_motion_count": high_count,
        "high_motion_fraction": args.high_motion_fraction,
        "boundary_radius": args.boundary_radius,
    }
    accum = {
        "all": {"baseline": [], "flow": []},
        "high_motion": {"baseline": [], "flow": []},
        "boundary": {"baseline": [], "flow": []},
    }

    for name in names:
        gt = read_rgb(gt_dir / name)
        baseline = read_rgb(baseline_dir / name)
        flow = read_rgb(flow_dir / name)
        accum["all"]["baseline"].append(image_metrics(baseline, gt, loss_fn, device))
        accum["all"]["flow"].append(image_metrics(flow, gt, loss_fn, device))

        if name in high_names:
            accum["high_motion"]["baseline"].append(image_metrics(baseline, gt, loss_fn, device))
            accum["high_motion"]["flow"].append(image_metrics(flow, gt, loss_fn, device))

        frame_id, view_id = parse_frame_view(name)
        mask_path = args.data_root / sequence / "bkgd_masks" / f"{view_id:02d}" / f"{frame_id:06d}.png"
        if mask_path.exists():
            fg = read_mask(mask_path, gt.shape[-2:])
            band = boundary_band(fg, args.boundary_radius)
            accum["boundary"]["baseline"].append(masked_metrics(baseline, gt, band, loss_fn, device))
            accum["boundary"]["flow"].append(masked_metrics(flow, gt, band, loss_fn, device))

    for subset_name, methods in accum.items():
        out[subset_name] = {}
        for method_name, values in methods.items():
            out[subset_name][method_name] = mean_dict(values)
        b = out[subset_name].get("baseline", {})
        f = out[subset_name].get("flow", {})
        out[subset_name]["delta_flow_minus_baseline"] = {
            k: float(f[k] - b[k])
            for k in ("psnr", "ssim", "lpips")
            if k in b and k in f
        }
    return out


def run_subset(args) -> Dict[str, object]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    loss_fn = load_lpips(device)
    results = [evaluate_subset_sequence(args, seq, loss_fn, device) for seq in args.sequences]

    summary: Dict[str, object] = {"mode": "subset", "sequences": args.sequences, "results": results}
    for subset_name in ("all", "high_motion", "boundary"):
        summary[subset_name] = {}
        for method in ("baseline", "flow"):
            vals = [r[subset_name][method] for r in results if method in r[subset_name]]
            summary[subset_name][method] = mean_dict(vals)
        b = summary[subset_name]["baseline"]
        f = summary[subset_name]["flow"]
        summary[subset_name]["delta_flow_minus_baseline"] = {
            k: float(f[k] - b[k])
            for k in ("psnr", "ssim", "lpips")
            if k in b and k in f
        }
    return summary


def load_cfg_args(model_path: Path) -> Namespace:
    raw = (model_path / "cfg_args").read_text()
    tree = ast.parse(raw, mode="eval")
    if not isinstance(tree.body, ast.Call) or getattr(tree.body.func, "id", "") != "Namespace":
        raise ValueError(f"Unexpected cfg_args format: {model_path / 'cfg_args'}")
    values = {}
    for kw in tree.body.keywords:
        values[kw.arg] = ast.literal_eval(kw.value)
    return Namespace(**values)


def prepare_model_args(model_path: Path) -> Tuple[object, object]:
    from arguments import ModelParams, PipelineParams

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    defaults = parser.parse_args([])
    cfg = load_cfg_args(model_path)
    merged = vars(defaults).copy()
    merged.update(vars(cfg))
    merged["model_path"] = str(model_path)
    args = Namespace(**merged)
    return model.extract(args), pipeline.extract(args)


def attention_sequence(args, sequence: str) -> Dict[str, object]:
    from gaussian_renderer import GaussianModel, render
    from scene import Scene

    model_path = args.output_root / sequence / "flow_view_token_gate" / args.flow_run
    dataset, pipeline = prepare_model_args(model_path)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, dataset)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    flow_cache = np.load(args.data_root / sequence / "flow_features" / FLOW_CACHE_NAME)
    train_views = flow_cache["train_view"].astype(int).tolist()
    view_count = len(train_views)
    top1_counts = torch.zeros(view_count, dtype=torch.float64)
    top3_counts = torch.zeros(view_count, dtype=torch.float64)

    entropy_sum = 0.0
    max_sum = 0.0
    match_sum = 0.0
    selected_cos_sum = 0.0
    nearest_cos_sum = 0.0
    pearson_sum = 0.0
    point_count = 0
    image_count = 0

    cameras = []
    for cam_list in scene.getTestCameras().values():
        cameras.extend(cam_list)
    if args.max_views > 0:
        cameras = cameras[: args.max_views]

    with torch.no_grad():
        for cam in cameras:
            render(cam, gaussians, pipeline, background)
            encoder = getattr(gaussians.non_rigid_deformer, "FlowEncoder", None)
            weights = getattr(encoder, "last_weights", None)
            if weights is None:
                continue
            weights = weights.squeeze(0).detach().float().cpu()
            if weights.numel() == 0:
                continue
            pose_id = cam.pose_id
            train_centers = gaussians.cond_dict[pose_id]["flow_train_camera_centers"].detach()
            points = gaussians.get_xyz.detach()
            current_dirs = torch.nn.functional.normalize(
                cam.camera_center.view(1, 3).to(points.device) - points,
                dim=-1,
                eps=1e-6,
            )
            train_dirs = torch.nn.functional.normalize(
                train_centers.view(1, train_centers.shape[0], 3) - points.unsqueeze(1),
                dim=-1,
                eps=1e-6,
            )
            cos = (current_dirs.unsqueeze(1) * train_dirs).sum(dim=-1).detach().float().cpu()
            selected = weights.argmax(dim=1)
            nearest = cos.argmax(dim=1)
            topk = torch.topk(weights, k=min(3, weights.shape[1]), dim=1).indices

            eps = 1e-8
            entropy = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=1) / math.log(weights.shape[1])
            max_prob = weights.max(dim=1).values
            selected_cos = cos.gather(1, selected[:, None]).squeeze(1)
            nearest_cos = cos.gather(1, nearest[:, None]).squeeze(1)
            wc = weights - weights.mean(dim=1, keepdim=True)
            cc = cos - cos.mean(dim=1, keepdim=True)
            pearson = (wc * cc).sum(dim=1) / (
                torch.sqrt((wc * wc).sum(dim=1).clamp_min(eps))
                * torch.sqrt((cc * cc).sum(dim=1).clamp_min(eps))
            )

            n = int(weights.shape[0])
            entropy_sum += float(entropy.sum().item())
            max_sum += float(max_prob.sum().item())
            match_sum += float((selected == nearest).float().sum().item())
            selected_cos_sum += float(selected_cos.sum().item())
            nearest_cos_sum += float(nearest_cos.sum().item())
            pearson_sum += float(pearson.sum().item())
            point_count += n
            image_count += 1
            top1_counts += torch.bincount(selected, minlength=view_count).double()
            top3_counts += torch.bincount(topk.reshape(-1), minlength=view_count).double()

    if point_count == 0:
        raise RuntimeError(f"No attention weights collected for {sequence}")
    top1_total = float(top1_counts.sum().item())
    top3_total = float(top3_counts.sum().item())
    return {
        "sequence": sequence,
        "image_count": image_count,
        "point_count": point_count,
        "train_views": train_views,
        "entropy_norm": entropy_sum / point_count,
        "max_weight": max_sum / point_count,
        "selected_nearest_match": match_sum / point_count,
        "selected_view_cos": selected_cos_sum / point_count,
        "nearest_view_cos": nearest_cos_sum / point_count,
        "weight_viewcos_pearson": pearson_sum / point_count,
        "top1_usage": {str(v): float(c / top1_total) for v, c in zip(train_views, top1_counts.tolist())},
        "top3_usage": {str(v): float(c / top3_total) for v, c in zip(train_views, top3_counts.tolist())},
    }


def run_attention(args) -> Dict[str, object]:
    results = [attention_sequence(args, seq) for seq in args.sequences]
    avg_keys = [
        "entropy_norm",
        "max_weight",
        "selected_nearest_match",
        "selected_view_cos",
        "nearest_view_cos",
        "weight_viewcos_pearson",
    ]
    summary = {"mode": "attention", "sequences": args.sequences, "results": results}
    summary["average"] = {k: float(np.mean([r[k] for r in results])) for k in avg_keys}
    return summary


def write_report(path: Path, data: Dict[str, object]) -> None:
    lines = [f"# Flow view-token diagnostic: {data['mode']}", ""]
    if data["mode"] == "subset":
        for subset in ("all", "high_motion", "boundary"):
            d = data[subset]["delta_flow_minus_baseline"]
            b = data[subset]["baseline"]
            f = data[subset]["flow"]
            lines.append(f"## {subset}")
            lines.append(
                f"baseline: PSNR {b.get('psnr')} SSIM {b.get('ssim')} LPIPS*1000 {b.get('lpips', 0) * 1000}"
            )
            lines.append(
                f"flow: PSNR {f.get('psnr')} SSIM {f.get('ssim')} LPIPS*1000 {f.get('lpips', 0) * 1000}"
            )
            lines.append(
                f"delta: PSNR {d.get('psnr')} SSIM {d.get('ssim')} LPIPS*1000 {d.get('lpips', 0) * 1000}"
            )
            lines.append("")
    else:
        avg = data["average"]
        lines.append("## average")
        for k, v in avg.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## per sequence")
        for r in data["results"]:
            lines.append(
                f"- {r['sequence']}: entropy_norm {r['entropy_norm']}, max_weight {r['max_weight']}, "
                f"selected_nearest_match {r['selected_nearest_match']}, weight_viewcos_pearson {r['weight_viewcos_pearson']}"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["subset", "attention"], required=True)
    parser.add_argument("--sequences", default=SEQ_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("DNA-Rendering"))
    parser.add_argument("--output-root", type=Path, default=Path("output/DNA-Rendering"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/flow"))
    parser.add_argument("--flow-run", default=FLOW_RUN_DEFAULT)
    parser.add_argument("--iteration", type=int, default=25000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--high-motion-fraction", type=float, default=0.30)
    parser.add_argument("--boundary-radius", type=int, default=5)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.data_root = (args.repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    args.output_root = (args.repo_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    args.log_dir = (args.repo_root / args.log_dir).resolve() if not args.log_dir.is_absolute() else args.log_dir
    args.sequences = parse_list(args.sequences)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo_root))

    data = run_subset(args) if args.mode == "subset" else run_attention(args)
    suffix = f"_{args.tag}" if args.tag else ""
    base = args.log_dir / f"flow_view_token_gate_{args.mode}{suffix}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(data, indent=2))
    write_report(md_path, data)
    print(f"[DONE] wrote {json_path}")
    print(f"[DONE] wrote {md_path}")


if __name__ == "__main__":
    main()
