#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from common import (
    PART_LOG_ROOT,
    default_part_label_dir,
    enable_part_stdout_logging,
    model_point_cloud_path,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def stable_file(path, stable_seconds):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False

    first = path.stat()
    time.sleep(stable_seconds)
    if not path.exists() or not path.is_file():
        return False
    second = path.stat()
    return (first.st_size, first.st_mtime_ns) == (second.st_size, second.st_mtime_ns)


def wait_for_training_artifacts(args):
    model_path = Path(args.model_path)
    point_cloud = model_point_cloud_path(model_path, args.iteration)
    mlp_ckpt = model_path / "mlp_ckpt" / f"iteration_{int(args.iteration)}" / "ckpt.pth"
    required = [point_cloud]
    if args.require_mlp_ckpt:
        required.append(mlp_ckpt)

    deadline = None
    if args.timeout_seconds > 0:
        deadline = time.time() + args.timeout_seconds

    print(f"[WATCH] Waiting for iteration {args.iteration} artifacts", flush=True)
    for path in required:
        print(f"[WATCH] required: {path}", flush=True)

    while True:
        ready = True
        for path in required:
            if not stable_file(path, args.stable_seconds):
                ready = False
                break
        if ready:
            print("[WATCH] Required artifacts are present and stable.", flush=True)
            return {
                "point_cloud": str(point_cloud),
                "mlp_ckpt": str(mlp_ckpt) if mlp_ckpt.exists() else None,
            }

        if deadline is not None and time.time() > deadline:
            missing = [str(path) for path in required if not path.exists()]
            raise TimeoutError(
                f"Timed out waiting for iteration {args.iteration} artifacts. "
                f"Missing: {missing if missing else 'none, but files did not become stable'}"
            )
        time.sleep(args.poll_seconds)


def run_command(cmd, env):
    print("[CMD] " + " ".join(str(v) for v in cmd), flush=True)
    subprocess.run([str(v) for v in cmd], cwd=str(REPO_ROOT), env=env, check=True)


def query_gpu_free_mb(gpu):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    text = result.stdout.strip().splitlines()
    if not text:
        return None
    return int(text[0].strip())


def wait_for_gpu_memory(args):
    if args.min_free_gpu_mb <= 0:
        return

    deadline = None
    if args.timeout_seconds > 0:
        deadline = time.time() + args.timeout_seconds

    last_report = 0.0
    while True:
        free_mb = query_gpu_free_mb(args.gpu)
        if free_mb is None:
            print("[WATCH] Could not query GPU memory; continuing without memory wait.", flush=True)
            return
        if free_mb >= args.min_free_gpu_mb:
            print(f"[WATCH] GPU {args.gpu} free memory {free_mb} MiB >= {args.min_free_gpu_mb} MiB.", flush=True)
            return

        now = time.time()
        if now - last_report >= 60.0:
            print(
                f"[WATCH] GPU {args.gpu} free memory {free_mb} MiB < "
                f"{args.min_free_gpu_mb} MiB; waiting before part-label build.",
                flush=True,
            )
            last_report = now

        if deadline is not None and now > deadline:
            raise TimeoutError(
                f"Timed out waiting for GPU {args.gpu} free memory >= {args.min_free_gpu_mb} MiB; "
                f"last free memory: {free_mb} MiB"
            )
        time.sleep(args.poll_seconds)


def build_part_labels(args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PATH"] = str(Path(args.python_bin).resolve().parent) + os.pathsep + env.get("PATH", "")

    out_dir = Path(args.out_dir) if args.out_dir else default_part_label_dir(args.model_path, args.iteration)
    label_path = out_dir / "gaussian_part_label.npy"
    if label_path.exists() and not args.overwrite:
        print(f"[WATCH] Existing labels found, skip: {label_path}", flush=True)
        return out_dir

    common_args = [
        "--source_path",
        args.source_path,
        "--model_path",
        args.model_path,
        "--iteration",
        args.iteration,
        "--gpu",
        args.gpu,
        "--out_dir",
        str(out_dir),
    ]
    if args.overwrite:
        common_args.append("--overwrite")

    prior_cmd = [args.python_bin, "part_label/build_smplx_nn_prior.py", *common_args]
    if args.smplx_vertex_seg:
        prior_cmd.extend(["--smplx_vertex_seg", args.smplx_vertex_seg])
    prior_cmd.extend(["--part_log_dir", args.part_log_dir])
    run_command(prior_cmd, env)

    final_cmd = [args.python_bin, "part_label/build_gaussian_part_labels.py", *common_args]
    if args.use_semantic_votes:
        semantic_root = args.semantic_root or str(Path(args.source_path) / "semantic_masks")
        final_cmd.extend(["--semantic_root", semantic_root])
    else:
        final_cmd.append("--prior_only")
    final_cmd.extend(["--part_log_dir", args.part_log_dir])
    run_command(final_cmd, env)

    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch a running SeqAvatar training output and build Gaussian part labels once a saved iteration appears."
    )
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=15000)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--smplx_vertex_seg", default=None)
    parser.add_argument("--use_semantic_votes", action="store_true")
    parser.add_argument("--semantic_root", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--poll_seconds", type=float, default=10.0)
    parser.add_argument("--stable_seconds", type=float, default=2.0)
    parser.add_argument("--timeout_seconds", type=float, default=21600.0)
    parser.add_argument(
        "--min_free_gpu_mb",
        type=int,
        default=4096,
        help="Wait until the target GPU has this much free memory before loading the part-label model. Use 0 to disable.",
    )
    parser.add_argument("--require_mlp_ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--part_log_dir", default=str(PART_LOG_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    enable_part_stdout_logging(args, "watch_and_build_part_labels", force=True)
    start_time = time.time()
    artifacts = wait_for_training_artifacts(args)
    wait_for_gpu_memory(args)
    out_dir = build_part_labels(args)
    write_json(
        Path(out_dir) / "watcher_meta.json",
        {
            "source_path": args.source_path,
            "model_path": args.model_path,
            "iteration": args.iteration,
            "gpu": args.gpu,
            "artifacts": artifacts,
            "out_dir": str(out_dir),
            "use_semantic_votes": args.use_semantic_votes,
            "elapsed_seconds": time.time() - start_time,
        },
    )
    print(f"[WATCH] Part labels ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
