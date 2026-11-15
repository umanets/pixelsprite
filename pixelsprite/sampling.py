"""Sampling: one pixel at a time with a key/value cache, separate guidance for shape and colour, output filter."""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext

import numpy as np
import torch

from .model import CanvasGPT2, edge_flags


def pick(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator) -> torch.Tensor:
    """Sample one class per row from (B,V) logits with temperature and nucleus (top-p) truncation."""
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, -1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, -1, descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p          # always keeps the most likely class
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter_(1, si, sp)
    return torch.multinomial(probs, 1, generator=gen).squeeze(1)


def split_pick(lc, lu, g_shape, g_colour, temperature, top_p, gen, force_empty=None, force_fill=None, empty_id=None):
    """Classifier-free guidance applied separately to the EMPTY-vs-filled decision (shape) and to the colour choice.

    lc / lu: (B, K+1) logits with and without text (lu None = no guidance). Shape: the log-odds of EMPTY are mixed,
    s = s_u + g_shape * (s_c - s_u). Colour: log-probabilities over the K colours are mixed,
    col = lp_u + g_colour * (lp_c - lp_u). Temperature and top-p act on the colour choice only. Returns (B,) ids."""
    lc = lc.float()
    E = empty_id if empty_id is not None else lc.shape[1] - 1
    lpc = torch.log_softmax(lc, -1)
    s_c = lpc[:, E] - torch.log1p(-lpc[:, E].exp().clamp(max=1 - 1e-6))
    col_c = lc[:, :E]
    if lu is not None:
        lpu = torch.log_softmax(lu.float(), -1)
        s_u = lpu[:, E] - torch.log1p(-lpu[:, E].exp().clamp(max=1 - 1e-6))
        s = s_u + g_shape * (s_c - s_u)
        col = torch.log_softmax(lu.float()[:, :E], -1)
        col = col + g_colour * (torch.log_softmax(col_c, -1) - col)
    else:
        s, col = s_c, col_c
    p_empty = torch.sigmoid(s)
    if force_empty is not None:
        p_empty = torch.where(force_empty, torch.ones_like(p_empty), p_empty)
    if force_fill is not None:
        p_empty = torch.where(force_fill, torch.zeros_like(p_empty), p_empty)
    is_empty = torch.rand(p_empty.shape, generator=gen, device=p_empty.device) < p_empty
    colour = pick(col, temperature, top_p, gen)
    return torch.where(is_empty, torch.full_like(colour, E), colour)


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """Mixed precision used for sampling: bfloat16 on GPUs that support it, float16 on older GPUs, none on CPU."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def _chunk(model, cond_c, cond_u, size, mask, known, window, g_shape, g_colour, temperature, top_p, gen):
    dev = cond_c.device
    b = cond_c.shape[0]
    rep = 1 if cond_u is None else 2
    cond = cond_c if cond_u is None else torch.cat([cond_c, cond_u])
    n = size * size
    E = model.EMPTY
    ar = torch.arange(n, device=dev)
    rows0, cols0 = ar // size, ar % size
    edge = edge_flags(mask, known)                       # (b,n)
    m_flat = mask.view(b, n)
    force_empty = (~m_flat & known[:, None]) | (~window.view(1, n).expand(b, n))
    force_fill = m_flat & known[:, None]
    amp = amp_dtype(dev)
    out = torch.empty((b, n), dtype=torch.long, device=dev)
    with torch.autocast(dev.type, dtype=amp) if amp is not None else nullcontext():
        cache = model.new_cache(b * rep, 2 + size + n, dev, amp or torch.float32)
        pre = model.prefix(cond, size, mask.repeat(rep, 1, 1), known.repeat(rep))
        tok0 = model.tokens(torch.full((b * rep, 1), model.BOS, device=dev), rows0[:1].expand(b * rep, 1),
                            cols0[:1].expand(b * rep, 1), edge[:, :1].repeat(rep, 1))
        x = torch.cat([pre, tok0], 1)
        logits = model.step(x, cache, 0)
        pos = x.shape[1]
        for t in range(n):
            lc, lu = (logits, None) if cond_u is None else (logits[:b], logits[b:])
            tok = split_pick(lc, lu, g_shape, g_colour, temperature, top_p, gen, force_empty[:, t], force_fill[:, t], E)
            out[:, t] = tok
            if t + 1 == n:
                break
            x = model.tokens(tok.repeat(rep)[:, None], rows0[t + 1:t + 2].expand(b * rep, 1),
                             cols0[t + 1:t + 2].expand(b * rep, 1), edge[:, t + 1:t + 2].repeat(rep, 1))
            logits = model.step(x, cache, pos)
            pos += 1
    return torch.where(out == E, torch.full_like(out, -1), out).view(b, size, size)


# ---------------------------------------------------------------------------------------------- output filter
def components(mask: np.ndarray) -> np.ndarray:
    """Sizes of the 8-connected components of a boolean mask, largest first."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue, n = deque([(y0, x0)]), 0
        while queue:
            y, x = queue.popleft()
            n += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        queue.append((yy, xx))
        sizes.append(n)
    return np.sort(np.array(sizes, dtype=np.int64))[::-1]


def rejected(mask: np.ndarray, max_px: int = 2) -> str | None:
    """Reason to drop a generated sprite, or None: an empty canvas, or a detached piece of 1-2 pixels."""
    if not mask.any():
        return "empty"
    sizes = components(mask)
    if len(sizes) >= 2 and sizes[1:].min() <= max_px:
        return "detached_1_2px"
    return None


@torch.no_grad()
def sample(model: CanvasGPT2, n: int, text: torch.Tensor | None = None, *, size: int = 16, g_shape: float = 1.0,
           g_colour: float = 2.0, temperature: float = 1.0, top_p: float = 1.0, seed: int = 0,
           mask: np.ndarray | None = None, window: int | None = None, use_filter: bool = True, chunk: int = 128,
           max_rounds: int = 50) -> tuple[np.ndarray, dict]:
    """n sprites -> (palette-index images (n, out, out) with -1 = transparent, stats).

    size: canvas the model draws on (16 or 32). window: draw only inside the central window x window of the canvas
    and return it cropped (size=32, window=24 -> 24x24 sprites). mask: (S,S) or (n,S,S) bool silhouette on the
    canvas of `size` (then the model only chooses colours). With use_filter, sprites with an empty canvas or a
    detached 1-2 px piece are drawn again (the filter is off when a mask is given)."""
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    if text is not None:
        text = text.to(dev, torch.float32)
        text = text.expand(n, -1) if text.shape[0] == 1 else text
    win = torch.ones((size, size), dtype=torch.bool, device=dev)
    if window:
        o = (size - window) // 2
        win = torch.zeros_like(win)
        win[o:o + window, o:o + window] = True
    if mask is not None:
        m = np.asarray(mask, bool)
        m = np.repeat(m[None], n, 0) if m.ndim == 2 else m
        masks = torch.as_tensor(m, device=dev)
        known = torch.ones(n, dtype=torch.bool, device=dev)
        use_filter = False
    else:
        masks = torch.zeros((n, size, size), dtype=torch.bool, device=dev)
        known = torch.zeros(n, dtype=torch.bool, device=dev)
    out = np.full((n, size, size), -1, dtype=np.int64)
    pending = np.arange(n)
    stats = {"sampled": 0, "rejected": 0, "reasons": {}, "rounds": 0}
    while len(pending) and stats["rounds"] < max_rounds:
        stats["rounds"] += 1
        got = []
        for s in range(0, len(pending), chunk):
            sel = torch.as_tensor(pending[s:s + chunk], device=dev)
            if text is None:
                cond_c, cond_u = model.cond(None, batch=len(sel)), None
            else:
                cond_c = model.cond(text[sel])
                cond_u = model.cond(None, batch=len(sel)) if (g_shape != 1.0 or g_colour != 1.0) else None
            got.append(_chunk(model, cond_c, cond_u, size, masks[sel], known[sel], win, g_shape, g_colour,
                              temperature, top_p, gen).cpu().numpy())
        imgs = np.concatenate(got)
        stats["sampled"] += len(pending)
        keep = np.ones(len(pending), bool)
        if use_filter:
            for j, im in enumerate(imgs):
                why = rejected(im >= 0)
                if why:
                    keep[j] = False
                    stats["reasons"][why] = stats["reasons"].get(why, 0) + 1
        out[pending[keep]] = imgs[keep]
        stats["rejected"] += int((~keep).sum())
        pending = pending[~keep]
    if len(pending):
        raise RuntimeError(f"{len(pending)} sprites still rejected after {max_rounds} rounds")
    stats["reject_rate"] = stats["rejected"] / max(1, stats["sampled"])
    if window:
        o = (size - window) // 2
        out = out[:, o:o + window, o:o + window]
    return out, stats
