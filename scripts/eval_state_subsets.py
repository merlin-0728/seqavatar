import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.loss_utils import ssim


NAME_RE = re.compile(r"frame_(\d+)_view_(\d+)\.png$")
DEFAULT_BASELINE_RUNS = {
    "0044_11": "20260701_162750",
    "0051_09": "20260701_130518",
    "0206_04": "20260701_130518",
    "0007_04": "20260701_130518",
    "0813_05": "20260701_130518",
    "0019_10": "20260701_130518",
}


def parse_name(name):
    match = NAME_RE.match(name)
    if match is None:
        raise ValueError(f"Unexpected image name: {name}")
    return int(match.group(1)), int(match.group(2))


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def read_mask(path, shape):
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def to_tensor(img, device):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)


def l1_np(pred, gt, mask=None):
    err = np.abs(pred - gt)
    if mask is not None:
        if mask.sum() == 0:
            return None
        return float(err[mask].mean())
    return float(err.mean())


def psnr_np(pred, gt, mask=None):
    err = (pred - gt) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return None
        mse = float(err[mask].mean())
    else:
        mse = float(err.mean())
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def ssim_torch(pred, gt, device):
    with torch.no_grad():
        return float(ssim(to_tensor(pred, device), to_tensor(gt, device)).item())


def lpips_torch(pred, gt, loss_fn, device):
    with torch.no_grad():
        return float(loss_fn(to_tensor(pred, device), to_tensor(gt, device)).mean().double().item())


def boundary_band(mask, width):
    k = 2 * width + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    m = mask.astype(np.uint8)
    dilated = cv2.dilate(m, kernel, iterations=1) > 0
    eroded = cv2.erode(m, kernel, iterations=1) > 0
    return np.logical_and(dilated, np.logical_not(eroded))


def max_error_crop_box(base, gt, patch):
    err = np.abs(base - gt).mean(axis=2).astype(np.float32)
    h, w = err.shape
    ph = min(patch, h)
    pw = min(patch, w)
    score = cv2.boxFilter(err, ddepth=-1, ksize=(pw, ph), normalize=False, borderType=cv2.BORDER_CONSTANT)
    cy, cx = np.unravel_index(int(score.argmax()), score.shape)
    y0 = int(np.clip(cy - ph // 2, 0, h - ph))
    x0 = int(np.clip(cx - pw // 2, 0, w - pw))
    return x0, y0, x0 + pw, y0 + ph


def image_metrics(pred, gt, loss_fn, device, with_lpips=True):
    out = {
        "L1": l1_np(pred, gt),
        "PSNR": psnr_np(pred, gt),
        "SSIM": ssim_torch(pred, gt, device),
    }
    if with_lpips:
        out["LPIPS"] = lpips_torch(pred, gt, loss_fn, device)
    return out


def mean_metric(rows, key):
    vals = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarize_method(rows):
    keys = sorted({key for row in rows for key in row.keys() if key not in {"seq", "name", "frame", "view", "x0", "y0", "x1", "y1"}})
    return {key: mean_metric(rows, key) for key in keys}


def add_delta(summary, method_name):
    if "baseline" not in summary or method_name not in summary:
        return {}
    delta = {}
    for key, val in summary[method_name].items():
        base = summary["baseline"].get(key)
        if val is not None and base is not None:
            delta[key] = val - base
    return delta


def collect_names(base_dir, state_dir):
    base_names = {p.name for p in (base_dir / "renders").glob("*.png")}
    state_names = {p.name for p in (state_dir / "renders").glob("*.png")}
    gt_names = {p.name for p in (state_dir / "gt").glob("*.png")}
    return sorted(base_names & state_names & gt_names)


def mask_path(data_root, seq, frame, view):
    return data_root / seq / "bkgd_masks" / f"{view:02d}" / f"{frame:06d}.png"


def compute_motion_scores(data_root, seq, names):
    by_frame_view = {}
    for name in names:
        frame, view = parse_name(name)
        by_frame_view[(frame, view)] = name
    views = sorted({view for _, view in by_frame_view.keys()})
    frames = sorted({frame for frame, _ in by_frame_view.keys()})
    masks = {}
    for frame in frames:
        for view in views:
            if (frame, view) not in by_frame_view:
                continue
            p = mask_path(data_root, seq, frame, view)
            masks[(frame, view)] = np.asarray(Image.open(p).convert("L")) > 0

    scores = defaultdict(list)
    for i, frame in enumerate(frames):
        neighbor_frames = []
        if i > 0:
            neighbor_frames.append(frames[i - 1])
        if i + 1 < len(frames):
            neighbor_frames.append(frames[i + 1])
        for view in views:
            cur = masks.get((frame, view))
            if cur is None:
                continue
            for nf in neighbor_frames:
                other = masks.get((nf, view))
                if other is None:
                    continue
                union = np.logical_or(cur, other).sum()
                if union == 0:
                    continue
                scores[frame].append(np.logical_xor(cur, other).sum() / union)
    return {frame: float(np.mean(vals)) for frame, vals in scores.items() if vals}


def evaluate_sequence(seq, args, loss_fn, device):
    baseline_run = args.baseline_runs.get(seq, args.baseline_run_default)
    if baseline_run is None:
        raise RuntimeError(f"No baseline run configured for {seq}")
    base_dir = args.output_root / seq / args.baseline_experiment / baseline_run / "novelview" / "ours_25000"
    state_dir = args.output_root / seq / args.state_experiment / args.state_run / "novelview" / "ours_25000"
    names = collect_names(base_dir, state_dir)
    if not names:
        raise RuntimeError(f"No common novelview images found for {seq}")

    motion_scores = compute_motion_scores(args.data_root, seq, names)
    frames = sorted({parse_name(name)[0] for name in names})
    top_n = max(1, int(math.ceil(len(frames) * args.high_motion_frac)))
    high_motion_frames = set(
        frame for frame, _ in sorted(motion_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    )

    rows = {
        "full": {"baseline": [], args.method_name: []},
        "high_motion": {"baseline": [], args.method_name: []},
        "boundary": {"baseline": [], args.method_name: []},
        "high_error_crop": {"baseline": [], args.method_name: []},
    }

    for name in tqdm(names, desc=seq):
        frame, view = parse_name(name)
        gt = read_rgb(state_dir / "gt" / name)
        base = read_rgb(base_dir / "renders" / name)
        state = read_rgb(state_dir / "renders" / name)
        mask = read_mask(mask_path(args.data_root, seq, frame, view), gt.shape)
        band = boundary_band(mask, args.boundary_width)
        x0, y0, x1, y1 = max_error_crop_box(base, gt, args.crop_size)

        meta = {"seq": seq, "name": name, "frame": frame, "view": view}
        for method, pred in [("baseline", base), (args.method_name, state)]:
            full = {**meta, **image_metrics(pred, gt, loss_fn, device, with_lpips=False)}
            rows["full"][method].append(full)

            if frame in high_motion_frames:
                rows["high_motion"][method].append({**meta, **image_metrics(pred, gt, loss_fn, device, with_lpips=True)})

            rows["boundary"][method].append(
                {
                    **meta,
                    "L1": l1_np(pred, gt, band),
                    "PSNR": psnr_np(pred, gt, band),
                    "pixels": int(band.sum()),
                }
            )

            crop_pred = pred[y0:y1, x0:x1]
            crop_gt = gt[y0:y1, x0:x1]
            rows["high_error_crop"][method].append(
                {
                    **meta,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    **image_metrics(crop_pred, crop_gt, loss_fn, device, with_lpips=True),
                }
            )

    summary = {}
    for subset, by_method in rows.items():
        summary[subset] = {
            "baseline": summarize_method(by_method["baseline"]),
            args.method_name: summarize_method(by_method[args.method_name]),
            "delta_state_minus_baseline": add_delta(
                {
                    "baseline": summarize_method(by_method["baseline"]),
                    args.method_name: summarize_method(by_method[args.method_name]),
                },
                args.method_name,
            ),
            "count": len(by_method["baseline"]),
        }

    return {
        "seq": seq,
        "baseline_run": baseline_run,
        "baseline_dir": str(base_dir),
        "state_dir": str(state_dir),
        "num_common_images": len(names),
        "high_motion_frames": sorted(high_motion_frames),
        "motion_scores": motion_scores,
        "summary": summary,
        "rows": rows if args.save_rows else None,
    }


def average_sequence_summaries(results, method_name):
    out = {}
    subsets = results[0]["summary"].keys()
    for subset in subsets:
        out[subset] = {}
        for method in ["baseline", method_name, "delta_state_minus_baseline"]:
            keys = sorted({k for r in results for k in r["summary"][subset][method].keys()})
            out[subset][method] = {}
            for key in keys:
                vals = [r["summary"][subset][method].get(key) for r in results]
                vals = [v for v in vals if v is not None]
                out[subset][method][key] = float(np.mean(vals)) if vals else None
        out[subset]["count"] = int(sum(r["summary"][subset]["count"] for r in results))
    return out


def fmt_metric(summary, key, scale=1.0):
    val = summary.get(key)
    if val is None:
        return "NA"
    return f"{val * scale:.6f}"


def write_markdown(path, args, results, overall):
    lines = []
    lines.append(f"# {args.method_name} subset evaluation")
    lines.append("")
    lines.append(f"state_experiment: `{args.state_experiment}`")
    lines.append(f"state_run: `{args.state_run}`")
    lines.append(f"high_motion_frac: `{args.high_motion_frac}`")
    lines.append(f"boundary_width: `{args.boundary_width}`")
    lines.append(f"crop_size: `{args.crop_size}`")
    lines.append("")
    lines.append(f"Delta is `{args.method_name} - baseline`; negative L1 / LPIPS is better, positive PSNR / SSIM is better.")
    lines.append("")

    for subset in ["high_motion", "boundary", "high_error_crop"]:
        lines.append(f"## {subset} overall")
        lines.append("")
        keys = ["L1", "PSNR", "SSIM", "LPIPS"] if subset != "boundary" else ["L1", "PSNR"]
        header = "| Method | " + " | ".join(keys) + " |"
        lines.append(header)
        lines.append("|---|" + "|".join(["---:"] * len(keys)) + "|")
        for method in ["baseline", args.method_name, "delta_state_minus_baseline"]:
            label = "delta" if method == "delta_state_minus_baseline" else method
            vals = []
            for key in keys:
                vals.append(fmt_metric(overall[subset][method], key, 1000.0 if key == "LPIPS" else 1.0))
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## Per-sequence delta")
    lines.append("")
    lines.append("| Subset | Sequence | dL1 | dPSNR | dSSIM | dLPIPS x1000 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for subset in ["high_motion", "boundary", "high_error_crop"]:
        for r in results:
            d = r["summary"][subset]["delta_state_minus_baseline"]
            lines.append(
                f"| {subset} | {r['seq']} | {fmt_metric(d, 'L1')} | {fmt_metric(d, 'PSNR')} | "
                f"{fmt_metric(d, 'SSIM')} | {fmt_metric(d, 'LPIPS', 1000.0)} |"
            )
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", default=["0044_11", "0051_09", "0206_04", "0007_04", "0813_05", "0019_10"])
    parser.add_argument("--output-root", type=Path, default=Path("output/DNA-Rendering"))
    parser.add_argument("--data-root", type=Path, default=Path("DNA-Rendering"))
    parser.add_argument("--baseline-experiment", default="orginal")
    parser.add_argument("--baseline-run-default", default=None)
    parser.add_argument("--baseline-runs-json", default=None)
    parser.add_argument("--state-experiment", default="state_warm_a04")
    parser.add_argument("--state-run", default="20260710_220456")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("note/state_subset_eval_20260711"))
    parser.add_argument("--high-motion-frac", type=float, default=0.25)
    parser.add_argument("--boundary-width", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--save-rows", action="store_true")
    args = parser.parse_args()
    if args.method_name is None:
        args.method_name = args.state_experiment
    args.baseline_runs = dict(DEFAULT_BASELINE_RUNS)
    if args.baseline_runs_json:
        args.baseline_runs.update(json.loads(args.baseline_runs_json))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    results = []
    for seq in args.sequences:
        results.append(evaluate_sequence(seq, args, loss_fn, device))

    overall = average_sequence_summaries(results, args.method_name)
    payload = {
        "config": {
            "sequences": args.sequences,
            "baseline_experiment": args.baseline_experiment,
            "baseline_runs": args.baseline_runs,
            "baseline_run_default": args.baseline_run_default,
            "state_experiment": args.state_experiment,
            "state_run": args.state_run,
            "method_name": args.method_name,
            "high_motion_frac": args.high_motion_frac,
            "boundary_width": args.boundary_width,
            "crop_size": args.crop_size,
        },
        "overall": overall,
        "sequences": results,
    }
    json_path = args.out_dir / "subset_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(args.out_dir / "subset_metrics.md", args, results, overall)
    print(f"Wrote {json_path}")
    print(f"Wrote {args.out_dir / 'subset_metrics.md'}")


if __name__ == "__main__":
    main()
