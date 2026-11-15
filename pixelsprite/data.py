"""Small data helpers for the learning notebooks: download one shard of the training data and decode it.

Needs the optional extras: pip install "pixelsprite[learn]" (pandas, pyarrow, matplotlib, scikit-learn).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .images import ALPHA_THRESHOLD, detect_scale

ATLAS = "Angais/pixel-art-atlas-1.4m"   # 1.4M sprites 16x16, CC0 (see its card for Krea's terms)
SPRITES_32 = "azyg0s/pixelart-172k"      # 172k sprites 32x32, MIT


@dataclass
class Sprites:
    rgba: np.ndarray          # (N, S, S, 4) uint8, alpha is exactly 0 or 255
    caption: list[str]
    subject: list[str]

    def __len__(self) -> int:
        return len(self.rgba)

    @property
    def mask(self) -> np.ndarray:
        return self.rgba[..., 3] > 0


def decode(png: bytes, size: int) -> np.ndarray:
    """PNG bytes -> RGBA (size, size, 4): undo integer upscaling, binarise alpha, zero RGB under transparency."""
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    if a.shape[0] != size:
        k = detect_scale(a)
        a = a[::k, ::k]
    a = a.copy()
    m = a[..., 3] >= ALPHA_THRESHOLD
    a[..., 3] = np.where(m, 255, 0)
    a[~m, :3] = 0
    return a


def load_shard(n: int | None = None, shard: str = "data/train-007.parquet", repo: str = ATLAS, size: int = 16,
               seed: int = 0) -> Sprites:
    """Download one parquet shard (about 29 MB, 94k sprites for the 16x16 atlas) and decode n random rows."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, shard, repo_type="dataset")
    df = pd.read_parquet(path)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    text_col = "caption" if "caption" in df else "text"
    subj_col = "subject" if "subject" in df else "text"
    rgba = np.stack([decode(img["bytes"], size) for img in df["image"]])
    return Sprites(rgba, df[text_col].astype(str).tolist(), df[subj_col].astype(str).tolist())
