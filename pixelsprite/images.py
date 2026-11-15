"""Palette-index images <-> RGBA, reading silhouettes from PNG, contact sheets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ALPHA_THRESHOLD = 128  # alpha >= this -> object pixel


def rgba_from_index(index: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Palette-index image (-1 = transparent) + palette (K,3) uint8 -> RGBA uint8 with alpha exactly 255 * mask."""
    mask = index >= 0
    out = np.zeros(index.shape + (4,), dtype=np.uint8)
    out[mask, :3] = palette[index[mask]]
    out[mask, 3] = 255
    return out


def mask_rgba(mask: np.ndarray, body: int = 160) -> np.ndarray:
    """Grey silhouette with a black 1-px outline (4-neighbourhood), transparent background."""
    p = np.pad(mask, 1)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    edge = mask & ~inner
    out = np.zeros(mask.shape + (4,), dtype=np.uint8)
    out[mask] = (body, body, body, 255)
    out[edge] = (0, 0, 0, 255)
    return out


def detect_scale(rgba: np.ndarray) -> int:
    """Largest integer k such that the image is made of k x k single-colour blocks (for upscaled PNGs)."""
    a = rgba.copy()
    a[a[..., 3] == 0] = 0  # RGB under alpha=0 is irrelevant
    h, w = a.shape[:2]
    for k in range(min(h, w) // 2, 1, -1):
        if h % k or w % k:
            continue
        b = a.reshape(h // k, k, w // k, k, a.shape[2])
        if (b == b[:, :1, :, :1]).all():
            return k
    return 1


def read_mask(path: str | Path) -> np.ndarray:
    """PNG (any integer upscale) -> bool silhouette from its alpha channel."""
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    k = detect_scale(rgba)
    return rgba[::k, ::k, 3] >= ALPHA_THRESHOLD


def checker_composite(rgba: np.ndarray, scale: int) -> np.ndarray:
    """RGBA sprite -> RGB, upscaled by `scale`, over a light checkerboard (shows transparency)."""
    up = rgba.repeat(scale, 0).repeat(scale, 1).astype(float)
    h, w = up.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.where(((yy // scale + xx // scale) % 2)[..., None] == 1, 236, 250).repeat(3, 2)
    a = up[..., 3:4] / 255.0
    return (up[..., :3] * a + bg * (1 - a)).astype(np.uint8)


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet(sprites: list[np.ndarray], title: str = "", scale: int = 8, cols: int = 4) -> Image.Image:
    """RGBA sprites -> one RGB contact sheet image."""
    s0 = sprites[0].shape[0]
    rows = int(np.ceil(len(sprites) / cols))
    cell, gap, top = s0 * scale, 10, 30 if title else 0
    img = Image.new("RGB", (cols * cell + (cols + 1) * gap, top + rows * (cell + gap) + gap), (252, 252, 251))
    if title:
        ImageDraw.Draw(img).text((gap, 7), title, fill=(11, 11, 11), font=_font(14))
    for i, rgba in enumerate(sprites):
        img.paste(Image.fromarray(checker_composite(rgba, scale)),
                  (gap + (i % cols) * (cell + gap), top + gap + (i // cols) * (cell + gap)))
    return img
