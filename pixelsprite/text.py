"""Frozen multilingual sentence encoder: a caption (English, Russian, Ukrainian, ...) -> a 384-number vector."""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TextEncoder:
    """Loads the encoder lazily (first call) and returns L2-normalised float32 vectors, as used in training."""

    def __init__(self, name: str = DEFAULT_TEXT_MODEL, device: str | None = None):
        self.name, self.device, self._model = name, device, None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def __call__(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        v = self._load().encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        return torch.as_tensor(np.asarray(v, dtype=np.float32))
