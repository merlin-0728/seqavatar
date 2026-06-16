#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from queue import Empty, Queue
from pathlib import Path

from common import (
    PART_LOG_ROOT,
    enable_part_stdout_logging,
    resolve_part_log_dir,
    safe_log_component,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Edit these defaults when you want to run without typing GPU/sequence args.
DEFAULT_GPU_IDS = ["2", "3"]
DEFAULT_SEQUENCES = ["0007_04", "0019_10", "0044_11", "0051_09", "0206_04", "0813_05"]

OOM_PATTERNS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)

PRINT_LOCK = threading.Lock()
ERROR_TAIL_LINES = 40


class SequenceOOMError(RuntimeError):
    pass


def str_list(values):
    return [str(v) for v in values]


def is_oom_line(line):
    lowered = line.lower().replace(" ", "")
    return any(pattern.replace(" ", "") in lowered for pattern in OOM_PATTERNS)


def safe_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    with PRINT_LOCK:
        print(*args, **kwargs)


def run_command(cmd, log_path, env, dry_run=False, stream_output=False, label=None):
    cmd = str_list(cmd)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    label_text = f"[{label}] " if label else ""

    safe_print(f"\n{label_text}[CMD] {' '.join(cmd)}\n{label_text}[LOG] {log_path}")
    if dry_run:
        with log_path.open("w", encoding="utf-8") as f:
            f.write("[DRY-RUN] " + " ".join(cmd) + "\n")
        return

    saw_oom = False
    tail = deque(maxlen=ERROR_TAIL_LINES)
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n")
        f.write(f"[CWD] {REPO_ROOT}\n")
        f.write(f"[CUDA_VISIBLE_DEVICES] {env.get('CUDA_VISIBLE_DEVICES', '')}\n")
        f.write("=" * 80 + "\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if is_oom_line(line):
                saw_oom = True
            tail.append(line.rstrip("\n"))
            if stream_output:
                safe_print(f"{label_text}{line}", end="")
            f.write(line)
        ret = proc.wait()
        elapsed = time.time() - start_time
        f.write("=" * 80 + "\n")
        f.write(f"[EXIT_CODE] {ret}\n")
        f.write(f"[ELAPSED_SECONDS] {elapsed:.2f}\n")
        if ret != 0:
            safe_print(f"{label_text}[ERROR] Command failed with exit code {ret}. Log: {log_path}")
            if tail:
                safe_print(f"{label_text}[ERROR] Last {len(tail)} log lines:")
                for line in tail:
                    safe_print(f"{label_text}{line}")
            if saw_oom:
                raise SequenceOOMError(f"OOM while running: {' '.join(cmd)}. Log: {log_path}")
            raise subprocess.CalledProcessError(ret, cmd)
    safe_print(f"{label_text}[DONE] exit=0 elapsed={elapsed:.1f}s")


class BackgroundCommand:
    def __init__(self, cmd, log_path, log_file, proc, label=None):
        self.cmd = cmd
        self.log_path = Path(log_path)
        self.log_file = log_file
        self.proc = proc
        self.label = label


def log_tail(path, max_lines=ERROR_TAIL_LINES):
    tail = deque(maxlen=max_lines)
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                tail.append(line.rstrip("\n"))
    except FileNotFoundError:
        return []
    return list(tail)


def start_background_command(cmd, log_path, env, dry_run=False, label=None):
    cmd = str_list(cmd)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    label_text = f"[{label}] " if label else ""

    safe_print(f"\n{label_text}[START] {' '.join(cmd)}\n{label_text}[LOG] {log_path}")
    if dry_run:
        with log_path.open("w", encoding="utf-8") as f:
            f.write("[DRY-RUN] " + " ".join(cmd) + "\n")
        return None

    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("[CMD] " + " ".join(cmd) + "\n")
    log_file.write(f"[CWD] {REPO_ROOT}\n")
    log_file.write(f"[CUDA_VISIBLE_DEVICES] {env.get('CUDA_VISIBLE_DEVICES', '')}\n")
    log_file.write("=" * 80 + "\n")
    log_file.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return BackgroundCommand(cmd, log_path, log_file, proc, label=label)


def wait_background_command(bg):
    if bg is None:
        return 0
    label_text = f"[{bg.label}] " if bg.label else ""
    ret = bg.proc.wait()
    bg.log_file.write("=" * 80 + "\n")
    bg.log_file.write(f"[EXIT_CODE] {ret}\n")
    bg.log_file.close()
    if ret == 0:
        safe_print(f"{label_text}[DONE] exit=0")
    else:
        safe_print(f"{label_text}[ERROR] Background command failed with exit code {ret}. Log: {bg.log_path}")
        tail = log_tail(bg.log_path)
        if tail:
            safe_print(f"{label_text}[ERROR] Last {len(tail)} log lines:")
            for line in tail:
                safe_print(f"{label_text}{line}")
    return ret


def terminate_background_command(bg):
    if bg is None or bg.proc.poll() is not None:
        return
    label_text = f"[{bg.label}] " if bg.label else ""
    safe_print(f"{label_text}[STOP] Terminating background command.")
    bg.proc.terminate()
    try:
        bg.proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        bg.proc.kill()
        bg.proc.wait()
    bg.log_file.write("=" * 80 + "\n")
    bg.log_file.write("[TERMINATED]\n")
    bg.log_file.close()


def base_train_args(args, dataset_path, exp_name):
    return [
        args.python_bin,
        "train.py",
        "-s",
        dataset_path,
        "--eval",
        "--exp_name",
        exp_name,
        "--motion_offset_flag",
        "--smpl_type",
        "smplx",
        "--actor_gender",
        "neutral",
        "--densify_until_iter",
        args.densify_until_iter,
        "--seq_len",
        args.seq_len,
        "--seq_xyz_knn",
        args.seq_xyz_knn,
        "--time_step_num",
        args.time_step_num,
        "--max_time_step",
        args.max_time_step,
        "--minimal_time_step",
        args.minimal_time_step,
        "--l1_loss_w",
        args.l1_loss_w,
        "--ssim_loss_w",
        args.ssim_loss_w,
        "--lpips_loss_w",
        args.lpips_loss_w,
    ]


def watcher_args(args, dataset_path, model_path, gpu_id):
    cmd = [
        args.python_bin,
        "part_label/watch_and_build_part_labels.py",
        "--source_path",
        dataset_path,
        "--model_path",
        model_path,
        "--iteration",
        args.mid_iter,
        "--gpu",
        gpu_id,
        "--python_bin",
        args.python_bin,
        "--poll_seconds",
        args.watcher_poll_seconds,
        "--stable_seconds",
        args.watcher_stable_seconds,
        "--timeout_seconds",
        args.watcher_timeout_seconds,
        "--min_free_gpu_mb",
        args.watcher_min_free_gpu_mb,
    ]
    if args.smplx_vertex_seg:
        cmd.extend(["--smplx_vertex_seg", args.smplx_vertex_seg])
    if args.use_semantic_votes:
        cmd.append("--use_semantic_votes")
        if args.semantic_root:
            cmd.extend(["--semantic_root", args.semantic_root])
    if args.overwrite_part_labels:
        cmd.append("--overwrite")
    if not args.require_mlp_ckpt:
        cmd.append("--no-require_mlp_ckpt")
    cmd.extend(["--part_log_dir", args.part_log_dir])
    return cmd


def run_sequence(args, sequence, run_time, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PATH"] = str(Path(args.python_bin).resolve().parent) + os.pathsep + env.get("PATH", "")

    dataset_path = str((Path(args.data_path) / sequence).resolve())
    exp_name = f"DNA-Rendering/{sequence}/{args.experiment_name}/{run_time}/"
    model_path = str((REPO_ROOT / "output" / exp_name).resolve())
    log_dir = resolve_part_log_dir(args.part_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_prefix = "_".join(
        safe_log_component(v)
        for v in (sequence, args.experiment_name, run_time)
    )

    smc_path = Path(dataset_path) / f"{sequence}.smc"
    if not smc_path.exists():
        raise FileNotFoundError(f"Missing SMC file: {smc_path}")

    safe_print(
        "=================================================\n"
        f"[INFO] Sequence: {sequence}\n"
        f"[INFO] GPU: {gpu_id}\n"
        f"[INFO] Dataset: {dataset_path}\n"
        f"[INFO] Model: {model_path}\n"
        "================================================="
    )

    watcher_cmd = watcher_args(args, dataset_path, model_path, gpu_id)
    watcher = start_background_command(
        watcher_cmd,
        log_dir / f"watch_part_label_{log_prefix}_iter_{args.mid_iter}.log",
        env,
        args.dry_run,
        f"{sequence}|gpu{gpu_id}|watch_part_label",
    )

    train_cmd = [
        *base_train_args(args, dataset_path, exp_name),
        "--iterations",
        args.final_iter,
        "--test_iterations",
        "3000",
        args.mid_iter,
        args.final_iter,
        "--save_iterations",
        "3000",
        args.mid_iter,
        args.final_iter,
    ]
    try:
        run_command(
            train_cmd,
            log_dir / f"train_{log_prefix}_continuous_to_{args.final_iter}.log",
            env,
            args.dry_run,
            args.stream_subprocess_logs,
            f"{sequence}|gpu{gpu_id}|train",
        )
    except Exception:
        terminate_background_command(watcher)
        raise

    watcher_ret = wait_background_command(watcher)
    if watcher_ret != 0:
        safe_print(
            f"[{sequence}|gpu{gpu_id}|watch_part_label] [WARN] "
            "Watcher failed during training; retrying once after training finished."
        )
        run_command(
            watcher_cmd,
            log_dir / f"watch_part_label_{log_prefix}_iter_{args.mid_iter}_retry_after_train.log",
            env,
            args.dry_run,
            args.stream_subprocess_logs,
            f"{sequence}|gpu{gpu_id}|watch_part_label_retry",
        )

    render_cmd = [
        args.python_bin,
        "render.py",
        "-s",
        dataset_path,
        "-m",
        model_path,
        "--motion_offset_flag",
        "--smpl_type",
        "smplx",
        "--actor_gender",
        "neutral",
        "--iteration",
        args.final_iter,
        "--skip_train",
        "--seq_len",
        args.seq_len,
        "--seq_xyz_knn",
        args.seq_xyz_knn,
        "--time_step_num",
        args.time_step_num,
        "--max_time_step",
        args.max_time_step,
        "--minimal_time_step",
        args.minimal_time_step,
    ]
    run_command(
        render_cmd,
        log_dir / f"render_{log_prefix}_iter_{args.final_iter}.log",
        env,
        args.dry_run,
        args.stream_subprocess_logs,
        f"{sequence}|gpu{gpu_id}|render",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one continuous SeqAvatar training process, build Gaussian part labels "
            "from a watched middle-iteration point cloud, then render evaluation."
        )
    )
    parser.add_argument("--data_path", default="/media/image/mxz/human/SeqAvatar/DNA-Rendering")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--experiment_name", default="orginal")
    parser.add_argument("--run_time", default=None)
    parser.add_argument("--gpus", nargs="+", default=DEFAULT_GPU_IDS)
    parser.add_argument("--gpu", default=None, help="Backward-compatible single-GPU override.")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--mid_iter", type=int, default=15000)
    parser.add_argument("--final_iter", type=int, default=25000)
    parser.add_argument("--densify_until_iter", type=int, default=1800)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--seq_xyz_knn", type=int, default=8)
    parser.add_argument("--time_step_num", type=int, default=3)
    parser.add_argument("--max_time_step", type=int, default=3)
    parser.add_argument("--minimal_time_step", type=int, default=1)
    parser.add_argument("--l1_loss_w", type=float, default=1.0)
    parser.add_argument("--ssim_loss_w", type=float, default=0.01)
    parser.add_argument("--lpips_loss_w", type=float, default=0.01)
    parser.add_argument("--smplx_vertex_seg", default=None)
    parser.add_argument("--use_semantic_votes", action="store_true")
    parser.add_argument("--semantic_root", default=None)
    parser.add_argument("--overwrite_part_labels", action="store_true")
    parser.add_argument("--watcher_poll_seconds", type=float, default=10.0)
    parser.add_argument("--watcher_stable_seconds", type=float, default=2.0)
    parser.add_argument("--watcher_timeout_seconds", type=float, default=21600.0)
    parser.add_argument("--watcher_min_free_gpu_mb", type=int, default=4096)
    parser.add_argument("--require_mlp_ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--part_log_dir", default=str(PART_LOG_ROOT))
    parser.add_argument(
        "--stream_subprocess_logs",
        action="store_true",
        help="Also print train/render subprocess logs to the pipeline log. Disabled by default to keep parallel runs readable.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def worker_loop(args, run_time, gpu_id, work_queue, results, lock):
    while True:
        try:
            sequence = work_queue.get_nowait()
        except Empty:
            return

        try:
            run_sequence(args, sequence, run_time, gpu_id)
        except SequenceOOMError as exc:
            safe_print(f"[OOM][GPU {gpu_id}] Skip sequence {sequence}: {exc}")
            with lock:
                results.append((sequence, gpu_id, "oom"))
        except Exception:
            with lock:
                results.append((sequence, gpu_id, "failed"))
            raise
        else:
            with lock:
                results.append((sequence, gpu_id, "ok"))
        finally:
            work_queue.task_done()


def main():
    args = parse_args()
    run_time = args.run_time or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.exp_name = f"midpart_pipeline/{args.experiment_name}/{run_time}"
    enable_part_stdout_logging(args, "train_midpart_pipeline", force=True)
    gpu_ids = [args.gpu] if args.gpu is not None else args.gpus
    gpu_ids = [str(gpu) for gpu in gpu_ids]
    if not gpu_ids:
        raise SystemExit("No GPU ids configured. Edit DEFAULT_GPU_IDS or pass --gpus.")

    safe_print(
        "=================================================\n"
        "[INFO] Mid-part training pipeline\n"
        f"[INFO] Run time: {run_time}\n"
        f"[INFO] GPUs: {' '.join(gpu_ids)}\n"
        f"[INFO] Python: {args.python_bin}\n"
        f"[INFO] Sequences: {' '.join(args.sequences)}\n"
        f"[INFO] Mid iteration: {args.mid_iter}\n"
        f"[INFO] Final iteration: {args.final_iter}\n"
        f"[INFO] Part label mode: {'semantic_vote' if args.use_semantic_votes else 'prior_only'}\n"
        "[INFO] Training mode: continuous train.py process with external watcher\n"
        f"[INFO] Log dir: {resolve_part_log_dir(args.part_log_dir)}\n"
        "[INFO] Subprocess logs: separate per-sequence files in the log dir above\n"
        "================================================="
    )

    work_queue = Queue()
    for sequence in args.sequences:
        work_queue.put(sequence)

    results = []
    lock = threading.Lock()
    threads = []
    for gpu_id in gpu_ids:
        thread = threading.Thread(
            target=worker_loop,
            args=(args, run_time, gpu_id, work_queue, results, lock),
            name=f"gpu-{gpu_id}",
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    summary_lines = ["=================================================", "[INFO] Summary"]
    for sequence, gpu_id, status in sorted(results):
        summary_lines.append(f"[INFO] {sequence}: {status} on GPU {gpu_id}")
    summary_lines.append("=================================================")
    safe_print("\n".join(summary_lines))
    if any(status == "failed" for _sequence, _gpu_id, status in results):
        raise SystemExit("At least one sequence failed for a non-OOM reason. Check logs above.")
    safe_print("[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()
