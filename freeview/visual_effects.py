from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


DEFAULT_BG_SIZE = (864, 1152)


def _to_uint8_rgb(value) -> np.ndarray:
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr[:3], (1, 2, 0))
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _to_uint8_alpha(value, size=None) -> np.ndarray:
    if value is None:
        if size is None:
            raise ValueError("size is required when alpha is None")
        return np.full((size[1], size[0]), 255, dtype=np.uint8)
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = np.max(arr, axis=0)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    return arr


def tensor_to_rgb_pil(value) -> Image.Image:
    return Image.fromarray(_to_uint8_rgb(value), "RGB")


def alpha_to_l_pil(value, size) -> Image.Image:
    return Image.fromarray(_to_uint8_alpha(value, size=size), "L")


def make_default_stage_background(size=DEFAULT_BG_SIZE) -> Image.Image:
    width, height = size
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    top = np.array([220, 232, 238], dtype=np.float32)
    bottom = np.array([194, 212, 222], dtype=np.float32)
    base = top * (1.0 - y[..., None]) + bottom * y[..., None]

    center_glow = np.exp(-(((x - 0.58) / 0.32) ** 2 + ((y - 0.38) / 0.45) ** 2))
    base = base + center_glow[..., None] * 20.0

    rng = np.random.default_rng(17)
    noise = rng.normal(0.0, 2.2, (height, width, 1)).astype(np.float32)
    base = np.clip(base + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    floor_y = int(height * 0.78)
    draw.rectangle((0, floor_y, width, height), fill=(226, 236, 241, 210))
    draw.line((0, floor_y, width, floor_y - int(height * 0.03)), fill=(130, 152, 165, 110), width=2)

    # Left translucent vertical rods, similar to the reference stage image.
    rod_area = int(width * 0.30)
    for idx, x0 in enumerate(range(-6, rod_area, max(8, width // 58))):
        alpha = 82 if idx % 3 else 130
        draw.line((x0, 0, x0 + int(width * 0.03), floor_y + int(height * 0.05)), fill=(255, 255, 255, alpha), width=2)
        draw.line((x0 + 4, 0, x0 + int(width * 0.03) + 4, floor_y), fill=(154, 179, 193, 55), width=1)

    # Back round plinth.
    back_bbox = (
        int(width * 0.58),
        int(height * 0.64),
        int(width * 1.04),
        int(height * 0.88),
    )
    draw.ellipse(back_bbox, fill=(232, 241, 245, 235), outline=(250, 252, 252, 165), width=3)
    draw.rectangle(
        (back_bbox[0], int(height * 0.73), back_bbox[2], int(height * 0.88)),
        fill=(199, 218, 229, 175),
    )
    draw.ellipse(back_bbox, outline=(250, 252, 252, 185), width=3)

    # Front white oval platform.
    front_bbox = (
        int(width * 0.04),
        int(height * 0.77),
        int(width * 0.98),
        int(height * 1.06),
    )
    draw.ellipse(front_bbox, fill=(246, 250, 251, 245), outline=(255, 255, 255, 220), width=4)
    draw.arc(front_bbox, 0, 180, fill=(255, 255, 255, 230), width=5)
    draw.arc(front_bbox, 180, 360, fill=(180, 202, 214, 105), width=3)

    return image.filter(ImageFilter.GaussianBlur(radius=0.25))


@lru_cache(maxsize=24)
def _background_cached(path: str | None, size: tuple[int, int], fit: str) -> Image.Image:
    if path:
        bg = Image.open(path).convert("RGB")
    else:
        bg = make_default_stage_background(DEFAULT_BG_SIZE)
    return fit_background(bg, size, fit=fit)


def fit_background(bg: Image.Image, size: tuple[int, int], fit="cover") -> Image.Image:
    width, height = size
    src_w, src_h = bg.size
    if fit == "contain":
        scale = min(width / src_w, height / src_h)
    else:
        scale = max(width / src_w, height / src_h)
    resized = bg.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (218, 230, 236))
    left = (width - resized.size[0]) // 2
    top = (height - resized.size[1]) // 2
    canvas.paste(resized, (left, top))
    if fit == "cover":
        return canvas.crop((0, 0, width, height))
    return canvas


def shifted_mask(mask: Image.Image, size: tuple[int, int], offset: Iterable[int]) -> Image.Image:
    dx, dy = [int(v) for v in offset]
    out = Image.new("L", size, 0)
    out.paste(mask, (dx, dy))
    return out


def compose_pil(
    foreground: Image.Image,
    alpha: Image.Image | None = None,
    *,
    background_path: str | Path | None = None,
    enable_background: bool = True,
    enable_shadow: bool = True,
    background_fit: str = "cover",
    shadow_offset: Iterable[int] = (46, 30),
    shadow_blur: float = 14.0,
    shadow_opacity: float = 0.33,
) -> Image.Image:
    foreground = foreground.convert("RGB")
    size = foreground.size
    if alpha is None:
        alpha = Image.new("L", size, 255)
    else:
        alpha = alpha.convert("L").resize(size, Image.Resampling.BILINEAR)

    if enable_background:
        bg = _background_cached(str(background_path) if background_path else None, size, background_fit).copy()
    else:
        bg = Image.new("RGB", size, (0, 0, 0))

    if enable_shadow:
        shadow_alpha = shifted_mask(alpha, size, shadow_offset)
        shadow_alpha = shadow_alpha.filter(ImageFilter.MaxFilter(9))
        shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(float(shadow_blur)))
        shadow_alpha = shadow_alpha.point(lambda p: int(p * float(shadow_opacity)))
        shadow_layer = Image.new("RGB", size, (0, 0, 0))
        bg.paste(shadow_layer, (0, 0), shadow_alpha)

    bg.paste(foreground, (0, 0), alpha)
    return bg


def compose_torch_render(
    image,
    alpha=None,
    *,
    background_path: str | Path | None = None,
    enable_background: bool = True,
    enable_shadow: bool = True,
    background_fit: str = "cover",
    shadow_offset: Iterable[int] = (46, 30),
    shadow_blur: float = 14.0,
    shadow_opacity: float = 0.33,
) -> np.ndarray:
    fg = tensor_to_rgb_pil(image)
    mask = alpha_to_l_pil(alpha, fg.size) if alpha is not None else None
    out = compose_pil(
        fg,
        mask,
        background_path=background_path,
        enable_background=enable_background,
        enable_shadow=enable_shadow,
        background_fit=background_fit,
        shadow_offset=shadow_offset,
        shadow_blur=shadow_blur,
        shadow_opacity=shadow_opacity,
    )
    return np.asarray(out, dtype=np.uint8)


def visual_metadata(args) -> dict:
    bg_path = getattr(args, "visual_background_path", None)
    return {
        "visual_background": bool(getattr(args, "visual_background", True)),
        "visual_shadow": bool(getattr(args, "visual_shadow", True)),
        "visual_background_path": str(bg_path) if bg_path else None,
        "visual_background_fit": getattr(args, "visual_background_fit", "cover"),
        "visual_shadow_offset": list(getattr(args, "visual_shadow_offset", [46, 30])),
        "visual_shadow_blur": float(getattr(args, "visual_shadow_blur", 14.0)),
        "visual_shadow_opacity": float(getattr(args, "visual_shadow_opacity", 0.33)),
    }
