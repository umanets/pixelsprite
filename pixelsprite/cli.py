"""Command line: text -> pixel sprites.

    pixelsprite "frost sword"                         # 16 sprites 16x16 into outputs/frost_sword_16_s0/
    pixelsprite "golden crown" --size 32 --n 8 --seed 3
    pixelsprite "ледяной меч" --size 24               # Russian / Ukrainian captions work too
    pixelsprite "poison" --mask-file flask.png        # colour your own silhouette (16x16 or 32x32 PNG)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from .images import read_mask
from .pipeline import DEFAULT_REPO, SpritePipeline


def slug(text: str | None) -> str:
    s = re.sub(r"\W+", "_", text.lower()).strip("_")[:40] if text else ""
    return s or "unconditional"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pixelsprite", description="Text -> 16x16 / 24x24 / 32x32 pixel sprites.")
    ap.add_argument("text", nargs="?", default=None, help="caption (omit to draw without text)")
    ap.add_argument("--size", type=int, default=16, choices=(16, 24, 32))
    ap.add_argument("--n", type=int, default=16, help="number of sprites")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--g-shape", type=float, default=1.0, help="guidance for the silhouette (default 1)")
    ap.add_argument("--g-colour", type=float, default=2.0, help="guidance for the colours (default 2)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--mask-file", type=Path, default=None, help="PNG silhouette (alpha) to colour")
    ap.add_argument("--no-filter", action="store_true", help="keep sprites with stray 1-2 px pieces")
    ap.add_argument("--model", default=DEFAULT_REPO, help="Hugging Face repo id or local folder")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--scale", type=int, default=8, help="upscale factor of the contact sheet")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    pipe = SpritePipeline.from_pretrained(args.model, device=args.device)
    mask = read_mask(args.mask_file) if args.mask_file else None
    size = mask.shape[0] if mask is not None else args.size
    res = pipe.generate(args.text, args.n, size, args.seed, g_shape=args.g_shape, g_colour=args.g_colour,
                        temperature=args.temperature, top_p=args.top_p, mask=mask, use_filter=not args.no_filter)
    out = args.out or Path("outputs") / f"{slug(args.text)}_{size}_s{args.seed}"
    title = f'"{args.text}"' if args.text else "no text"
    res.save(out, stem=slug(args.text), sheet_scale=args.scale,
             title=f"{title}, {size}x{size}, shape g {args.g_shape}, colour g {args.g_colour}, seed {args.seed}")
    print(f"{len(res.sprites)} sprites -> {out} ({time.time() - t0:.1f}s on {pipe.device})")
    if mask is None and not args.no_filter:
        st = res.stats
        print(f"filter: redrew {st['rejected']} of {st['sampled']} ({100 * st['reject_rate']:.1f}%) {st['reasons']}")


if __name__ == "__main__":
    main()
