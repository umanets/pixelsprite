"""CanvasGPT2: a decoder-only transformer that draws a pixel sprite one pixel at a time.

Sequence: [text, size, S mask-row tokens, S*S pixel tokens].
- text token: linear projection of a frozen sentence vector (a learned "null" vector when there is no text);
- size token: which canvas the model draws on (16 or 32; 24 is a 24x24 window on the 32 canvas);
- mask-row token r: the S bits of row r of a given silhouette (zeros + "unknown" kind when no silhouette is given);
- pixel token t: previous pixel's token + row / column of pixel t + edge flag of pixel t.
The output at pixel t is a distribution over K palette colours + EMPTY (transparent).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

KIND_PIXEL, KIND_TEXT, KIND_SIZE, KIND_ROW_KNOWN, KIND_ROW_UNKNOWN = 0, 1, 2, 3, 4
EDGE_INNER, EDGE_BOUNDARY, EDGE_OUTSIDE, EDGE_UNKNOWN = 0, 1, 2, 3


@dataclass
class CanvasConfig:
    n_colors: int = 1024
    text_dim: int = 384
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_size: int = 32
    sizes: tuple = field(default_factory=lambda: (16, 24, 32))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sizes"] = list(d["sizes"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CanvasConfig":
        keys = cls.__dataclass_fields__
        d = {k: v for k, v in d.items() if k in keys}
        d["sizes"] = tuple(d.get("sizes", (16, 24, 32)))
        return cls(**d)


def specials(n_colors: int) -> tuple[int, int, int]:
    """(EMPTY, FILLED, BOS) token ids for a palette of n_colors colours (FILLED is unused by this model)."""
    return n_colors, n_colors + 1, n_colors + 2


def boundary(mask: torch.Tensor) -> torch.Tensor:
    """(B,S,S) bool -> pixels of the mask that touch the outside (4-neighbourhood, canvas edge counts as outside)."""
    p = F.pad(mask.float(), (1, 1, 1, 1))
    inner = (p[:, :-2, 1:-1] > 0) & (p[:, 2:, 1:-1] > 0) & (p[:, 1:-1, :-2] > 0) & (p[:, 1:-1, 2:] > 0)
    return mask & ~inner


def edge_flags(mask: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """mask (B,S,S) bool, known (B,) bool -> (B,S*S) long edge flag per pixel."""
    b = mask.shape[0]
    bd = boundary(mask)
    e = torch.where(mask, torch.where(bd, EDGE_BOUNDARY, EDGE_INNER), torch.full_like(mask, EDGE_OUTSIDE, dtype=torch.long))
    e = e.long().view(b, -1)
    return torch.where(known[:, None], e, torch.full_like(e, EDGE_UNKNOWN))


class Block(nn.Module):
    """Pre-norm transformer block: causal self-attention + MLP, with an optional key/value cache."""

    def __init__(self, cfg: CanvasConfig):
        super().__init__()
        d = cfg.d_model
        self.h, self.p = cfg.n_heads, cfg.dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, cfg.d_ff), nn.Linear(cfg.d_ff, d)

    def forward(self, x, cache=None, pos: int = 0):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if cache is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        else:
            kc, vc = cache
            kc[:, :, pos:pos + t], vc[:, :, pos:pos + t] = k, v
            if t > 1:
                assert pos == 0, "multi-token cached call only for the prefix"
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                a = F.scaled_dot_product_attention(q, kc[:, :, :pos + 1], vc[:, :, :pos + 1])
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(b, t, d)), self.p, self.training)
        return x + F.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))), self.p, self.training)


def _init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)


class CanvasGPT2(nn.Module):
    def __init__(self, cfg: CanvasConfig):
        super().__init__()
        self.cfg = cfg
        d, Kc = cfg.d_model, cfg.n_colors
        self.EMPTY, _, self.BOS = specials(Kc)
        self.cond_proj = nn.Linear(cfg.text_dim, d)
        self.null_cond = nn.Parameter(torch.randn(d) * 0.02)
        self.size_emb = nn.Embedding(len(cfg.sizes), d)
        self.mask_proj = nn.Linear(cfg.max_size, d)
        self.tok = nn.Embedding(Kc + 3, d)
        self.row = nn.Embedding(cfg.max_size, d)
        self.col = nn.Embedding(cfg.max_size, d)
        self.edge = nn.Embedding(4, d)
        self.kind = nn.Embedding(5, d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, Kc + 1)
        self.apply(_init)
        for b in self.blocks:  # GPT-2 style scaling of the residual projections
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def cond(self, text: torch.Tensor | None, drop: torch.Tensor | None = None, batch: int | None = None):
        """Text vectors (B, text_dim) -> condition vectors (B, d); None -> the learned null condition."""
        if text is None:
            return self.null_cond.expand(batch, -1)
        c = self.cond_proj(text)
        if drop is not None:
            c = torch.where(drop[:, None], self.null_cond.expand_as(c), c)
        return c

    def prefix(self, cond, size: int, mask: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """cond (B,d); mask (B,S,S) bool (ignored where not known); known (B,) -> (B, 2 + S, d)."""
        b = cond.shape[0]
        k = self.kind.weight
        s_tok = (self.size_emb.weight[list(self.cfg.sizes).index(size)] + k[KIND_SIZE]).expand(b, -1)
        bits = torch.where(known[:, None, None], mask, torch.zeros_like(mask)).to(cond.dtype)
        bits = F.pad(bits, (0, self.cfg.max_size - size))
        rows = torch.arange(size, device=cond.device)
        kind = torch.where(known, KIND_ROW_KNOWN, KIND_ROW_UNKNOWN)
        rtok = self.mask_proj(bits) + self.row(rows)[None] + self.kind(kind)[:, None]
        return torch.cat([(cond + k[KIND_TEXT])[:, None], s_tok[:, None], rtok], 1)

    def tokens(self, prev, rows, cols, edge):
        return self.tok(prev) + self.row(rows) + self.col(cols) + self.edge(edge) + self.kind.weight[KIND_PIXEL]

    def forward(self, cond, size, mask, known, prev, rows, cols, edge):
        """Teacher forcing: logits (B, S*S, K+1) for every pixel at once."""
        x = torch.cat([self.prefix(cond, size, mask, known), self.tokens(prev, rows, cols, edge)], 1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x[:, 2 + size:]))

    def new_cache(self, b: int, length: int, device, dtype):
        hd = self.cfg.d_model // self.cfg.n_heads
        return [(torch.empty(b, self.cfg.n_heads, length, hd, device=device, dtype=dtype),
                 torch.empty(b, self.cfg.n_heads, length, hd, device=device, dtype=dtype)) for _ in self.blocks]

    def step(self, x, cache, pos: int) -> torch.Tensor:
        """Run new tokens x (B,t,d) through the network using the cache; logits of the last token (B, K+1)."""
        for blk, c in zip(self.blocks, cache):
            x = blk(x, c, pos)
        return self.head(self.ln(x[:, -1]))
