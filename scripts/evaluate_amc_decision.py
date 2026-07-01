#!/usr/bin/env python3
"""Evaluate whether AMC is worth continuing as a motion innovation.

The script is intentionally evaluation-only. It reads existing metric JSONs,
SMPL-X pose files, and optionally rendered images for a silhouette-boundary
PSNR check. It does not import training code or modify any experiment output.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_SEQUENCES = ("0044_11", "0051_09", "0206_04", "0813_05", "0007_04", "0019_10")
DEFAULT_BASELINE_RUN_OVERRIDES = {"0044_11": "20260701_162750"}
IMAGE_NAME_RE = re.compile(r"^frame_(?P<pose>\d{6})_view_(?P<view>\d{2})(?:\.png)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline and AMC outputs on overall, high-motion, and optional "
            "silhouette-boundary subsets, then emit a continuation decision."
        )
    )
    script_path = Path(__file__).resolve()
    default_repo_root = script_path.parents[1]

    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument("--dataset-name", default="DNA-Rendering")
    parser.add_argument("--data-root", type=Path, default=None, help="Defaults to <repo-root>/<dataset-name>.")
    parser.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES), help="Comma or space separated sequence list.")
    parser.add_argument("--baseline-exp", default="orginal", help="Experiment folder for baseline. Kept as repo spelling.")
    parser.add_argument("--candidate-exp", default="amc_causal")
    parser.add_argument("--baseline-run", default="20260701_130518")
    parser.add_argument(
        "--baseline-run-map",
        default="0044_11=20260701_162750",
        help="Comma separated overrides, e.g. 0044_11=20260701_162750,*=20260701_130518.",
    )
    parser.add_argument("--candidate-run", default="20260701_180553")
    parser.add_argument("--candidate-run-map", default="", help="Optional per-sequence candidate run overrides.")
    parser.add_argument("--iteration", type=int, default=25000)
    parser.add_argument("--split", default="novelview")

    parser.add_argument("--top-motion-ratio", type=float, default=0.2)
    parser.add_argument(
        "--motion-step",
        type=int,
        default=5,
        help="Former pose offset used for motion score. DNA novelview uses pose interval 5.",
    )
    parser.add_argument("--exclude-root", action="store_true", help="Exclude global/root joint from pose motion score.")
    parser.add_argument(
        "--motion-score",
        choices=("pose", "xyz", "pose_xyz"),
        default="pose",
        help="Motion score source for top-motion subset selection.",
    )
    parser.add_argument(
        "--xyz-scale",
        type=float,
        default=1.0,
        help="Scale xyz velocity before combining with pose score for --motion-score pose_xyz.",
    )

    parser.add_argument(
        "--skip-boundary",
        action="store_true",
        help="Skip silhouette-boundary PSNR. Overall and high-motion metrics are still computed.",
    )
    parser.add_argument("--boundary-radius", type=int, default=3)

    parser.add_argument("--lpips-threshold", type=float, default=0.5, help="Meaningful LPIPS*1000 improvement.")
    parser.add_argument("--ssim-threshold", type=float, default=0.0002, help="Meaningful SSIM improvement.")
    parser.add_argument("--boundary-psnr-threshold", type=float, default=0.10, help="Meaningful boundary PSNR dB gain.")
    parser.add_argument("--max-overall-psnr-drop", type=float, default=0.10)
    parser.add_argument("--max-overall-lpips-regress", type=float, default=0.25, help="LPIPS*1000 regression limit.")

    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <repo-root>/logs/AMC.")
    parser.add_argument("--report-prefix", default="amc_decision")
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args()


def parse_sequence_list(raw: str) -> List[str]:
    return [item for item in re.split(r"[\s,]+", raw.strip()) if item]


def parse_run_map(raw: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not raw.strip():
        return result
    for item in re.split(r"[\s,]+", raw.strip()):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Run map item must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def resolve_run(sequence: str, default_run: str, run_map: Dict[str, str]) -> str:
    return run_map.get(sequence, run_map.get("*", default_run))


def metric_path(
    repo_root: Path,
    dataset_name: str,
    sequence: str,
    exp_name: str,
    run_name: str,
    split: str,
    iteration: int,
    per_view: bool,
) -> Path:
    prefix = "per_view" if per_view else "results_"
    filename = f"{prefix}{split}_{iteration}.json" if per_view else f"results_{split}_{iteration}.json"
    return repo_root / "output" / dataset_name / sequence / exp_name / run_name / "metrics" / filename


def render_dir(
    repo_root: Path,
    dataset_name: str,
    sequence: str,
    exp_name: str,
    run_name: str,
    split: str,
    iteration: int,
) -> Path:
    return repo_root / "output" / dataset_name / sequence / exp_name / run_name / split / f"ours_{iteration}"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def parse_image_name(name: str) -> Tuple[int, int]:
    match = IMAGE_NAME_RE.match(Path(name).stem)
    if not match:
        raise ValueError(f"Cannot parse image name: {name}")
    return int(match.group("pose")), int(match.group("view"))


def common_names(base_per_view: dict, cand_per_view: dict) -> List[str]:
    names = set(base_per_view["PSNR"].keys()) & set(cand_per_view["PSNR"].keys())
    return sorted(names, key=lambda n: parse_image_name(n))


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(np.mean(vals))


def mean_metrics(per_view: dict, names: Sequence[str]) -> Dict[str, Optional[float]]:
    return {metric: mean_or_none(per_view[metric][name] for name in names) for metric in METRICS}


def delta_metrics(candidate: dict, baseline: dict) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for metric in METRICS:
        if candidate.get(metric) is None or baseline.get(metric) is None:
            out[metric] = None
        else:
            out[metric] = float(candidate[metric]) - float(baseline[metric])
    return out


def aggregate_sequence_metrics(seq_rows: Sequence[dict], section: str, side: str) -> Dict[str, Optional[float]]:
    return {
        metric: mean_or_none(row[section][side].get(metric) for row in seq_rows if row[section][side].get(metric) is not None)
        for metric in METRICS
    }


def aggregate_delta(seq_rows: Sequence[dict], section: str) -> Dict[str, Optional[float]]:
    return {
        metric: mean_or_none(row[section]["delta"].get(metric) for row in seq_rows if row[section]["delta"].get(metric) is not None)
        for metric in METRICS
    }


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    small = np.abs(angles) < 1e-8
    sin_half_over_angle = np.empty_like(angles)
    sin_half_over_angle[~small] = np.sin(half_angles[~small]) / angles[~small]
    sin_half_over_angle[small] = 0.5 - (angles[small] * angles[small]) / 48.0
    quat = np.concatenate([np.cos(half_angles), axis_angle * sin_half_over_angle], axis=-1)
    return quaternion_to_matrix(quat)


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    r, i, j, k = np.moveaxis(quat, -1, 0)
    two_s = 2.0 / np.sum(quat * quat, axis=-1)
    matrix = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return matrix.reshape(quat.shape[:-1] + (3, 3))


def load_pose_xyz(data_root: Path, sequence: str, pose_id: int) -> Tuple[np.ndarray, np.ndarray]:
    model_file = data_root / sequence / "model" / f"{pose_id:06d}.npz"
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    loaded = np.load(model_file, allow_pickle=True)
    poses = loaded["poses"].reshape(-1, 3)
    xyz = loaded["obs_xyz"]
    return poses, xyz


def motion_scores_for_poses(
    data_root: Path,
    sequence: str,
    pose_ids: Sequence[int],
    motion_step: int,
    exclude_root: bool,
    motion_score: str,
    xyz_scale: float,
) -> Dict[int, float]:
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def get(pose_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if pose_id not in cache:
            cache[pose_id] = load_pose_xyz(data_root, sequence, pose_id)
        return cache[pose_id]

    scores: Dict[int, float] = {}
    for pose_id in sorted(set(pose_ids)):
        former_id = max(pose_id - motion_step, 0)
        cur_pose, cur_xyz = get(pose_id)
        former_pose, former_xyz = get(former_id)
        dt = max(pose_id - former_id, 1)

        score_parts = []
        if motion_score in ("pose", "pose_xyz"):
            cur_mat = axis_angle_to_matrix(cur_pose)
            former_mat = axis_angle_to_matrix(former_pose)
            rel = np.matmul(cur_mat, np.swapaxes(former_mat, -1, -2))
            trace = np.trace(rel, axis1=-2, axis2=-1)
            angles = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
            if exclude_root and angles.shape[0] > 1:
                angles = angles[1:]
            score_parts.append(float(np.mean(angles) / dt))

        if motion_score in ("xyz", "pose_xyz"):
            score_parts.append(float(np.mean(np.linalg.norm(cur_xyz - former_xyz, axis=-1)) / dt) * xyz_scale)

        scores[pose_id] = float(np.sum(score_parts))
    return scores


def select_pose_subset(scores: Dict[int, float], ratio: float, high: bool) -> List[int]:
    if not scores:
        return []
    ratio = min(max(ratio, 0.0), 1.0)
    k = max(1, int(math.ceil(len(scores) * ratio)))
    items = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=high)
    return sorted(pose_id for pose_id, _ in items[:k])


def names_for_poses(names: Sequence[str], pose_ids: Sequence[int]) -> List[str]:
    pose_set = set(pose_ids)
    return [name for name in names if parse_image_name(name)[0] in pose_set]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom == 0:
        return None
    return float(np.sum(x * y) / denom)


def ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks_out = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks_out[order[start:end]] = rank
        start = end
    return ranks_out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def metric_motion_correlations(
    base_per_view: dict,
    cand_per_view: dict,
    names: Sequence[str],
    pose_scores: Dict[int, float],
) -> Dict[str, Dict[str, Optional[float]]]:
    x = [pose_scores[parse_image_name(name)[0]] for name in names]
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for metric in METRICS:
        y = [float(cand_per_view[metric][name]) - float(base_per_view[metric][name]) for name in names]
        out[metric] = {"pearson": pearson(x, y), "spearman": spearman(x, y)}
    return out


def load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(radius, 0)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        shifted = []
        for dy in range(3):
            for dx in range(3):
                shifted.append(padded[dy : dy + out.shape[0], dx : dx + out.shape[1]])
        out = np.logical_or.reduce(shifted)
    return out


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    fg = mask.astype(bool)
    dilated = dilate_bool(fg, radius)
    eroded = ~dilate_bool(~fg, radius)
    return dilated & ~eroded


def psnr_on_mask(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if pred.shape != gt.shape:
        raise ValueError(f"Image shape mismatch: pred={pred.shape}, gt={gt.shape}")
    if mask.shape != pred.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {pred.shape[:2]}")
    if not np.any(mask):
        return None
    diff = pred[mask] - gt[mask]
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 100.0
    return float(-10.0 * math.log10(mse))


def mask_path(data_root: Path, sequence: str, name: str) -> Path:
    pose_id, view_id = parse_image_name(name)
    return data_root / sequence / "bkgd_masks" / f"{view_id:02d}" / f"{pose_id:06d}.png"


def load_boundary_mask(data_root: Path, sequence: str, name: str, target_hw: Tuple[int, int], radius: int) -> np.ndarray:
    path = mask_path(data_root, sequence, name)
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("L")
    target_h, target_w = target_hw
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), Image.NEAREST)
    mask = np.asarray(image) != 0
    return boundary_band(mask, radius)


def boundary_psnr_metrics(
    data_root: Path,
    repo_root: Path,
    dataset_name: str,
    sequence: str,
    baseline_exp: str,
    baseline_run: str,
    candidate_exp: str,
    candidate_run: str,
    split: str,
    iteration: int,
    names: Sequence[str],
    radius: int,
) -> Dict[str, Optional[float]]:
    base_dir = render_dir(repo_root, dataset_name, sequence, baseline_exp, baseline_run, split, iteration)
    cand_dir = render_dir(repo_root, dataset_name, sequence, candidate_exp, candidate_run, split, iteration)
    base_scores: List[float] = []
    cand_scores: List[float] = []

    mask_cache: Dict[str, np.ndarray] = {}
    for name in names:
        filename = f"{name}.png" if not name.endswith(".png") else name
        gt_path = base_dir / "gt" / filename
        base_path = base_dir / "renders" / filename
        cand_path = cand_dir / "renders" / filename
        if not (gt_path.exists() and base_path.exists() and cand_path.exists()):
            continue
        gt = load_rgb(gt_path)
        base_pred = load_rgb(base_path)
        cand_pred = load_rgb(cand_path)
        if name not in mask_cache:
            mask_cache[name] = load_boundary_mask(data_root, sequence, name, gt.shape[:2], radius)
        mask = mask_cache[name]
        base_psnr = psnr_on_mask(base_pred, gt, mask)
        cand_psnr = psnr_on_mask(cand_pred, gt, mask)
        if base_psnr is not None and cand_psnr is not None:
            base_scores.append(base_psnr)
            cand_scores.append(cand_psnr)

    base_mean = mean_or_none(base_scores)
    cand_mean = mean_or_none(cand_scores)
    return {
        "PSNR": cand_mean,
        "baseline_PSNR": base_mean,
        "delta_PSNR": None if base_mean is None or cand_mean is None else cand_mean - base_mean,
        "count": len(base_scores),
    }


def fmt_float(value: Optional[float], digits: int = 4, scale: float = 1.0) -> str:
    if value is None:
        return "NA"
    return f"{value * scale:.{digits}f}"


def markdown_metric_table(title: str, rows: Sequence[dict], section: str) -> List[str]:
    lines = [f"## {title}", "", "| Sequence | Base PSNR | Cand PSNR | dPSNR | Base SSIM | Cand SSIM | dSSIM | Base LPIPS*1000 | Cand LPIPS*1000 | dLPIPS*1000 | Count |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        sec = row[section]
        count = sec.get("count", "")
        lines.append(
            "| {sequence} | {bpsnr} | {cpsnr} | {dpsnr} | {bssim} | {cssim} | {dssim} | {blpips} | {clpips} | {dlpips} | {count} |".format(
                sequence=row["sequence"],
                bpsnr=fmt_float(sec["baseline"].get("PSNR")),
                cpsnr=fmt_float(sec["candidate"].get("PSNR")),
                dpsnr=fmt_float(sec["delta"].get("PSNR")),
                bssim=fmt_float(sec["baseline"].get("SSIM"), 6),
                cssim=fmt_float(sec["candidate"].get("SSIM"), 6),
                dssim=fmt_float(sec["delta"].get("SSIM"), 6),
                blpips=fmt_float(sec["baseline"].get("LPIPS"), scale=1000.0),
                clpips=fmt_float(sec["candidate"].get("LPIPS"), scale=1000.0),
                dlpips=fmt_float(sec["delta"].get("LPIPS"), scale=1000.0),
                count=count,
            )
        )
    lines.append("")
    return lines


def decide(aggregate: dict, args: argparse.Namespace) -> Tuple[str, List[str]]:
    overall = aggregate["overall"]["delta"]
    high = aggregate["high_motion"]["delta"]
    boundary = aggregate.get("boundary_all", {})

    high_positive = (
        high.get("LPIPS") is not None
        and high["LPIPS"] * 1000.0 <= -args.lpips_threshold
    ) or (high.get("SSIM") is not None and high["SSIM"] >= args.ssim_threshold)
    boundary_positive = (
        boundary.get("delta_PSNR") is not None
        and boundary["delta_PSNR"] >= args.boundary_psnr_threshold
    )
    overall_positive = (
        overall.get("LPIPS") is not None
        and overall["LPIPS"] * 1000.0 <= -args.lpips_threshold
    ) or (overall.get("SSIM") is not None and overall["SSIM"] >= args.ssim_threshold)
    overall_bad = (
        overall.get("PSNR") is not None
        and overall["PSNR"] < -args.max_overall_psnr_drop
    ) or (
        overall.get("LPIPS") is not None
        and overall["LPIPS"] * 1000.0 > args.max_overall_lpips_regress
    )

    reasons = []
    if high_positive:
        reasons.append("high-motion subset has a meaningful SSIM or LPIPS improvement")
    if boundary_positive:
        reasons.append("silhouette-boundary PSNR has a meaningful gain")
    if overall_positive:
        reasons.append("overall metrics have a meaningful SSIM or LPIPS improvement")
    if overall_bad:
        reasons.append("overall regression exceeds the configured tolerance")

    if (high_positive or boundary_positive) and not overall_bad:
        return "CONTINUE_AS_CANDIDATE", reasons
    if overall_positive and not overall_bad:
        return "HOLD_AND_DIAGNOSE", reasons or ["overall is mildly positive, but high-motion evidence is weak"]
    return "STOP_AS_MAIN", reasons or ["no meaningful high-motion/local gain under the current thresholds"]


def evaluate_sequence(args: argparse.Namespace, sequence: str, baseline_run: str, candidate_run: str) -> dict:
    repo_root = args.repo_root
    data_root = args.data_root or (repo_root / args.dataset_name)

    base_result = load_json(
        metric_path(repo_root, args.dataset_name, sequence, args.baseline_exp, baseline_run, args.split, args.iteration, False)
    )
    cand_result = load_json(
        metric_path(repo_root, args.dataset_name, sequence, args.candidate_exp, candidate_run, args.split, args.iteration, False)
    )
    base_per = load_json(
        metric_path(repo_root, args.dataset_name, sequence, args.baseline_exp, baseline_run, args.split, args.iteration, True)
    )
    cand_per = load_json(
        metric_path(repo_root, args.dataset_name, sequence, args.candidate_exp, candidate_run, args.split, args.iteration, True)
    )

    names = common_names(base_per, cand_per)
    pose_ids = sorted({parse_image_name(name)[0] for name in names})
    pose_scores = motion_scores_for_poses(
        data_root,
        sequence,
        pose_ids,
        args.motion_step,
        args.exclude_root,
        args.motion_score,
        args.xyz_scale,
    )
    high_poses = select_pose_subset(pose_scores, args.top_motion_ratio, high=True)
    low_poses = select_pose_subset(pose_scores, args.top_motion_ratio, high=False)
    high_names = names_for_poses(names, high_poses)
    low_names = names_for_poses(names, low_poses)

    overall = {
        "baseline": {metric: float(base_result[metric]) for metric in METRICS},
        "candidate": {metric: float(cand_result[metric]) for metric in METRICS},
        "count": len(names),
    }
    overall["delta"] = delta_metrics(overall["candidate"], overall["baseline"])
    high_motion = {
        "baseline": mean_metrics(base_per, high_names),
        "candidate": mean_metrics(cand_per, high_names),
        "count": len(high_names),
    }
    high_motion["delta"] = delta_metrics(high_motion["candidate"], high_motion["baseline"])
    low_motion = {
        "baseline": mean_metrics(base_per, low_names),
        "candidate": mean_metrics(cand_per, low_names),
        "count": len(low_names),
    }
    low_motion["delta"] = delta_metrics(low_motion["candidate"], low_motion["baseline"])

    row = {
        "sequence": sequence,
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "overall": overall,
        "high_motion": high_motion,
        "low_motion": low_motion,
        "motion": {
            "scores": {str(k): v for k, v in pose_scores.items()},
            "high_poses": high_poses,
            "low_poses": low_poses,
            "high_score_mean": mean_or_none(pose_scores[p] for p in high_poses),
            "low_score_mean": mean_or_none(pose_scores[p] for p in low_poses),
        },
        "correlation": metric_motion_correlations(base_per, cand_per, names, pose_scores),
    }

    if not args.skip_boundary:
        row["boundary_all"] = boundary_psnr_metrics(
            data_root,
            repo_root,
            args.dataset_name,
            sequence,
            args.baseline_exp,
            baseline_run,
            args.candidate_exp,
            candidate_run,
            args.split,
            args.iteration,
            names,
            args.boundary_radius,
        )
        row["boundary_high_motion"] = boundary_psnr_metrics(
            data_root,
            repo_root,
            args.dataset_name,
            sequence,
            args.baseline_exp,
            baseline_run,
            args.candidate_exp,
            candidate_run,
            args.split,
            args.iteration,
            high_names,
            args.boundary_radius,
        )

    return row


def aggregate(rows: Sequence[dict]) -> dict:
    result = {}
    for section in ("overall", "high_motion", "low_motion"):
        baseline = aggregate_sequence_metrics(rows, section, "baseline")
        candidate = aggregate_sequence_metrics(rows, section, "candidate")
        result[section] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": aggregate_delta(rows, section),
        }
    for section in ("boundary_all", "boundary_high_motion"):
        if any(section in row for row in rows):
            base = mean_or_none(row[section].get("baseline_PSNR") for row in rows if section in row)
            cand = mean_or_none(row[section].get("PSNR") for row in rows if section in row)
            result[section] = {
                "baseline_PSNR": base,
                "PSNR": cand,
                "delta_PSNR": None if base is None or cand is None else cand - base,
            }
    return result


def build_markdown(args: argparse.Namespace, rows: Sequence[dict], aggregate_row: dict, decision: str, reasons: Sequence[str]) -> str:
    lines: List[str] = [
        "# AMC Decision Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dataset: {args.dataset_name}",
        f"- Baseline: {args.baseline_exp}",
        f"- Candidate: {args.candidate_exp}",
        f"- Split / iteration: {args.split} / {args.iteration}",
        "- Metric source: metrics/results and metrics/per_view JSON from training eval; render-log totals are not used because they do not provide per-view subset values.",
        f"- Motion score: {args.motion_score}, step={args.motion_step}, top_ratio={args.top_motion_ratio}",
        f"- Decision: **{decision}**",
        "",
    ]
    if reasons:
        lines.append("Reasons:")
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines.extend(markdown_metric_table("Overall", rows, "overall"))
    lines.extend(markdown_metric_table("High-Motion Subset", rows, "high_motion"))
    lines.extend(markdown_metric_table("Low-Motion Subset", rows, "low_motion"))

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Section | dPSNR | dSSIM | dLPIPS*1000 |")
    lines.append("|---|---:|---:|---:|")
    for section in ("overall", "high_motion", "low_motion"):
        delta = aggregate_row[section]["delta"]
        lines.append(
            f"| {section} | {fmt_float(delta.get('PSNR'))} | {fmt_float(delta.get('SSIM'), 6)} | {fmt_float(delta.get('LPIPS'), scale=1000.0)} |"
        )
    lines.append("")

    if "boundary_all" in aggregate_row:
        lines.append("## Boundary PSNR")
        lines.append("")
        lines.append("| Section | Baseline PSNR | Candidate PSNR | dPSNR |")
        lines.append("|---|---:|---:|---:|")
        for section in ("boundary_all", "boundary_high_motion"):
            if section not in aggregate_row:
                continue
            sec = aggregate_row[section]
            lines.append(
                f"| {section} | {fmt_float(sec.get('baseline_PSNR'))} | {fmt_float(sec.get('PSNR'))} | {fmt_float(sec.get('delta_PSNR'))} |"
            )
        lines.append("")

    lines.append("## Motion Selection")
    lines.append("")
    lines.append("| Sequence | High poses | High score mean | Low poses | Low score mean |")
    lines.append("|---|---|---:|---|---:|")
    for row in rows:
        motion = row["motion"]
        lines.append(
            f"| {row['sequence']} | {','.join(map(str, motion['high_poses']))} | {fmt_float(motion['high_score_mean'], 6)} | "
            f"{','.join(map(str, motion['low_poses']))} | {fmt_float(motion['low_score_mean'], 6)} |"
        )
    lines.append("")

    lines.append("## Motion-Delta Correlation")
    lines.append("")
    lines.append("Correlation is between per-image motion score and candidate-minus-baseline metric delta.")
    lines.append("For LPIPS, a negative correlation means higher-motion frames tend to improve more.")
    lines.append("")
    lines.append("| Sequence | PSNR pearson | SSIM pearson | LPIPS pearson | PSNR spearman | SSIM spearman | LPIPS spearman |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        corr = row["correlation"]
        lines.append(
            f"| {row['sequence']} | {fmt_float(corr['PSNR']['pearson'])} | {fmt_float(corr['SSIM']['pearson'])} | {fmt_float(corr['LPIPS']['pearson'])} | "
            f"{fmt_float(corr['PSNR']['spearman'])} | {fmt_float(corr['SSIM']['spearman'])} | {fmt_float(corr['LPIPS']['spearman'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.data_root = (args.data_root.resolve() if args.data_root else (args.repo_root / args.dataset_name).resolve())
    sequences = parse_sequence_list(args.sequences)
    baseline_run_map = parse_run_map(args.baseline_run_map)
    candidate_run_map = parse_run_map(args.candidate_run_map)

    rows = []
    for sequence in sequences:
        baseline_run = resolve_run(sequence, args.baseline_run, baseline_run_map)
        candidate_run = resolve_run(sequence, args.candidate_run, candidate_run_map)
        rows.append(evaluate_sequence(args, sequence, baseline_run, candidate_run))

    aggregate_row = aggregate(rows)
    decision, reasons = decide(aggregate_row, args)
    report = {
        "config": {
            "repo_root": str(args.repo_root),
            "data_root": str(args.data_root),
            "dataset_name": args.dataset_name,
            "baseline_exp": args.baseline_exp,
            "candidate_exp": args.candidate_exp,
            "iteration": args.iteration,
            "split": args.split,
            "top_motion_ratio": args.top_motion_ratio,
            "motion_step": args.motion_step,
            "motion_score": args.motion_score,
            "exclude_root": args.exclude_root,
            "skip_boundary": args.skip_boundary,
            "metric_source": "metrics/results and metrics/per_view JSON from training eval",
        },
        "decision": decision,
        "reasons": reasons,
        "aggregate": aggregate_row,
        "sequences": rows,
    }

    markdown = build_markdown(args, rows, aggregate_row, decision, reasons)
    print(markdown)

    if not args.no_write_report:
        output_dir = (args.output_dir or (args.repo_root / "logs" / "AMC")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"{args.report_prefix}_{timestamp}.json"
        md_path = output_dir / f"{args.report_prefix}_{timestamp}.md"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        md_path.write_text(markdown + "\n")
        print("")
        print(f"[INFO] Wrote JSON report: {json_path}")
        print(f"[INFO] Wrote Markdown report: {md_path}")


if __name__ == "__main__":
    main()
