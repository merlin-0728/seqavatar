#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np

from common import LABELS, write_json


def parse_ints(values):
    if values is None:
        return None
    return [int(v) for v in values]


def list_available_ids(root, width):
    if not root.exists():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.isdigit():
            out.append(int(child.name))
    return out


def foreground_body_mask(mask_path):
    src = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if src is None:
        raise FileNotFoundError(mask_path)
    if src.ndim == 3:
        src = src[:, :, 0]
    return np.where(src != 0, 1, 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate SeqAvatar part semantic masks. A real human parser backend is not bundled; "
            "use --foreground_body_from_bkgd only for pipeline debugging."
        )
    )
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--semantic_root", default=None)
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--foreground_body_from_bkgd",
        action="store_true",
        help="Debug fallback: label every foreground pixel from bkgd_masks as body=1.",
    )
    args = parser.parse_args()

    source_path = Path(args.source_path).resolve()
    image_root = source_path / "images"
    bkgd_root = source_path / "bkgd_masks"
    semantic_root = Path(args.semantic_root).resolve() if args.semantic_root else source_path / "semantic_masks"

    if not args.foreground_body_from_bkgd:
        raise SystemExit(
            "No human parser backend is configured in this repository. "
            "Provide parser integration later, or pass --foreground_body_from_bkgd only for debugging."
        )

    views = parse_ints(args.views)
    if views is None:
        views = list_available_ids(image_root, 2)
    frames = parse_ints(args.frames)

    written = 0
    skipped = 0
    missing = 0
    for view in views:
        view_dir = image_root / f"{view:02d}"
        if frames is None:
            frame_ids = sorted(int(p.stem) for p in view_dir.glob("*.png") if p.stem.isdigit())
        else:
            frame_ids = frames

        for frame in frame_ids:
            img_path = view_dir / f"{frame:06d}.png"
            mask_path = bkgd_root / f"{view:02d}" / f"{frame:06d}.png"
            out_path = semantic_root / f"{view:02d}" / f"{frame:06d}.png"
            if not img_path.exists() or not mask_path.exists():
                missing += 1
                continue
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            label = foreground_body_mask(mask_path)
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                missing += 1
                continue
            if label.shape[:2] != image.shape[:2]:
                raise RuntimeError(f"Mask/image size mismatch: {mask_path} vs {img_path}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), label)
            written += 1

    write_json(
        semantic_root.parent / "semantic_masks_meta.json",
        {
            "source_path": str(source_path),
            "semantic_root": str(semantic_root),
            "labels": LABELS,
            "mode": "foreground_body_from_bkgd",
            "warning": "This is a debugging fallback, not a real human parsing result.",
            "views": views,
            "frames": frames if frames is not None else "all_available",
            "written": written,
            "skipped": skipped,
            "missing": missing,
        },
    )
    print(f"semantic masks written={written} skipped={skipped} missing={missing}")


if __name__ == "__main__":
    main()
