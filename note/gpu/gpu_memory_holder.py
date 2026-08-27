#!/usr/bin/env python3
import argparse
import signal
import sys
import time


running = True


def stop(_signum, _frame):
    global running
    running = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Allocate CUDA memory and keep the process alive."
    )
    parser.add_argument("--target-mib", type=int, default=10000)
    parser.add_argument("--block-mib", type=int, default=256)
    parser.add_argument("--sleep-sec", type=int, default=60)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_mib <= 0 or args.block_mib <= 0:
        raise ValueError("target-mib and block-mib must be positive")

    try:
        import torch
    except Exception as exc:
        print(f"PyTorch is required for GPU allocation: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("CUDA is not available in this Python environment", file=sys.stderr)
        return 3

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    device = torch.device("cuda:0")
    tensors = []
    remaining_mib = args.target_mib

    while remaining_mib > 0:
        chunk_mib = min(args.block_mib, remaining_mib)
        try:
            tensors.append(
                torch.empty(chunk_mib * 1024 * 1024, dtype=torch.uint8, device=device)
            )
            torch.cuda.synchronize(device)
        except RuntimeError as exc:
            print(f"CUDA allocation failed after {args.target_mib - remaining_mib} MiB: {exc}", file=sys.stderr)
            return 4
        remaining_mib -= chunk_mib

    while running:
        time.sleep(args.sleep_sec)

    tensors.clear()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
