"""SpritePipeline: load the model from a Hugging Face repo or a local folder and draw sprites."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .images import rgba_from_index, sheet
from .model import CanvasConfig, CanvasGPT2
from .sampling import sample
from .text import DEFAULT_TEXT_MODEL, TextEncoder

# Hugging Face repo with model.safetensors, config.json and palette.json (override with $PIXELSPRITE_MODEL).
DEFAULT_REPO = os.environ.get("PIXELSPRITE_MODEL", "ihor-umanets/pixelsprite-canvas-v2")
FILES = ("config.json", "model.safetensors", "palette.json")
SIZES = (16, 24, 32)


@dataclass
class Generation:
    """Result of one call: RGBA sprites (uint8, alpha = 0 or 255), palette-index images (-1 = transparent), stats."""
    sprites: list[np.ndarray]
    index: np.ndarray
    stats: dict = field(default_factory=dict)

    def save(self, folder: str | Path, stem: str = "sprite", sheet_scale: int = 8, title: str = "") -> Path:
        """Write each sprite as a PNG (real size, alpha = silhouette) plus an upscaled contact sheet.png."""
        from PIL import Image
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(self.sprites):
            Image.fromarray(s, "RGBA").save(folder / f"{stem}_{i:02d}.png")
        sheet(self.sprites, title, scale=sheet_scale).save(folder / "sheet.png")
        return folder


def _resolve(repo_or_path: str | Path, revision: str | None) -> dict[str, Path]:
    path = Path(repo_or_path)
    if path.is_dir():
        return {f: path / f for f in FILES}
    from huggingface_hub import hf_hub_download
    return {f: Path(hf_hub_download(str(repo_or_path), f, revision=revision)) for f in FILES}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpritePipeline:
    def __init__(self, model: CanvasGPT2, palette: np.ndarray, config: dict, device: str):
        self.model, self.palette, self.config, self.device = model, palette, config, device
        self.encoder = TextEncoder(config.get("text_model", DEFAULT_TEXT_MODEL), device=device)

    @classmethod
    def from_pretrained(cls, repo_or_path: str | Path = DEFAULT_REPO, device: str | None = None,
                        revision: str | None = None) -> "SpritePipeline":
        from safetensors.torch import load_file
        device = device or default_device()
        files = _resolve(repo_or_path, revision)
        config = json.loads(files["config.json"].read_text(encoding="utf-8"))
        model = CanvasGPT2(CanvasConfig.from_dict(config["model"]))
        model.load_state_dict(load_file(str(files["model.safetensors"])))
        model.to(device).eval()
        palette = np.asarray(json.loads(files["palette.json"].read_text(encoding="utf-8"))["colors"], dtype=np.uint8)
        return cls(model, palette, config, device)

    def embed(self, texts: list[str]) -> torch.Tensor:
        return self.encoder(texts)

    def default_chunk(self, canvas: int) -> int:
        """Sprites drawn in parallel. GPU values match the ones used to make the published examples (seeds
        reproduce only with the same device and chunk); CPU values keep memory modest."""
        if self.device == "cuda":
            return 128 if canvas == 16 else 48
        return 32 if canvas == 16 else 8

    def generate(self, text: str | None = None, n: int = 16, size: int = 16, seed: int = 0, *, g_shape: float = 1.0,
                 g_colour: float = 2.0, temperature: float = 1.0, top_p: float = 1.0, mask: np.ndarray | None = None,
                 use_filter: bool = True, chunk: int | None = None) -> Generation:
        """Draw n sprites of size 16, 24 or 32 from a caption (None = no text).

        g_shape / g_colour: guidance strength for the silhouette and for the colours (1 = plain conditional model;
        larger = follow the text harder). mask: optional bool silhouette of shape (size, size); then only colours
        are chosen. Size 24 is drawn as a 24x24 window in the middle of the 32x32 canvas."""
        if size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}")
        canvas, window = (32, 24) if size == 24 else (size, None)
        m = None
        if mask is not None:
            mask = np.asarray(mask, bool)
            if mask.shape != (size, size):
                raise ValueError(f"mask must be {size}x{size}, got {mask.shape}")
            m = mask
            if window:  # place the 24x24 silhouette in the middle of the 32x32 canvas
                m = np.zeros((canvas, canvas), bool)
                o = (canvas - window) // 2
                m[o:o + window, o:o + window] = mask
        emb = self.embed([text]) if text else None
        idx, stats = sample(self.model, n, emb, size=canvas, g_shape=g_shape, g_colour=g_colour,
                            temperature=temperature, top_p=top_p, seed=seed, mask=m, window=window,
                            use_filter=use_filter, chunk=chunk or self.default_chunk(canvas))
        return Generation([rgba_from_index(x, self.palette) for x in idx], idx, stats)
