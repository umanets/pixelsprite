"""Colour maths used by the notebooks: sRGB <-> OKLab and mapping pixels to the nearest palette colour."""
from __future__ import annotations

import numpy as np
import torch


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB -> (..., 3) OKLab (L in 0..1). OKLab: Björn Ottosson, 2020."""
    lin = srgb_to_linear(rgb)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = lin @ m1.T
    return np.cbrt(lms) @ m2.T


def nearest_index(rgb: np.ndarray, palette: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    """(..., 3) uint8 colours -> (...) index of the nearest palette colour in OKLab."""
    flat = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    pal = torch.as_tensor(srgb_to_oklab(palette))
    out = np.empty(len(flat), dtype=np.int64)
    for s in range(0, len(flat), chunk):
        lab = torch.as_tensor(srgb_to_oklab(flat[s:s + chunk]))
        out[s:s + chunk] = torch.cdist(lab, pal).argmin(1).numpy()
    return out.reshape(np.shape(rgb)[:-1])


def to_index(rgba: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """RGBA sprites (..., S, S, 4) -> palette-index images (..., S, S) with -1 for transparent pixels."""
    idx = nearest_index(rgba[..., :3], palette)
    return np.where(rgba[..., 3] > 0, idx, -1)
