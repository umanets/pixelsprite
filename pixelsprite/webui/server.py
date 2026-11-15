"""Local web UI: type a caption (or draw a silhouette) and see the sprites.

    pixelsprite-ui                       # http://127.0.0.1:7860
    pixelsprite-ui --port 8000 --model ./my-local-model-folder

Standard library HTTP server, one model in memory, requests served one at a time on the model.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from ..images import detect_scale, mask_rgba
from ..pipeline import DEFAULT_REPO, SpritePipeline

HTML = Path(__file__).with_name("index.html")
MAX_N = 64
MAX_BODY = 2 << 20
LIMITS = {"g_shape": (0.0, 10.0), "g_colour": (0.0, 10.0), "temperature": (0.05, 3.0), "top_p": (0.05, 1.0)}


def png_b64(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class App:
    def __init__(self, model: str, device: str | None):
        self.pipe = SpritePipeline.from_pretrained(model, device=device)
        self.lock = threading.Lock()

    def info(self) -> dict:
        return {"device": self.pipe.device, "params": self.pipe.model.n_params(), "palette": len(self.pipe.palette),
                "sizes": [16, 24, 32], "text_model": self.pipe.config.get("text_model")}

    def generate(self, req: dict) -> dict:
        t0 = time.time()
        text = (req.get("text") or "").strip() or None
        size = int(req.get("size", 16))
        if size not in (16, 24, 32):
            raise ValueError("size must be 16, 24 or 32")
        n = int(np.clip(int(req.get("n", 16)), 1, MAX_N))
        seed = int(req.get("seed", 0))
        p = {k: float(np.clip(float(req.get(k, d)), *LIMITS[k]))
             for k, d in (("g_shape", 1.0), ("g_colour", 2.0), ("temperature", 1.0), ("top_p", 1.0))}
        use_filter = bool(req.get("filter", True))
        mode = req.get("mode", "new")
        mask = None
        if mode == "draw":
            bits = req.get("mask_bits")
            if not bits:
                raise ValueError("draw a silhouette first")
            mask = np.array([[c == "1" for c in row] for row in bits], bool)
            if mask.shape != (size, size):
                raise ValueError(f"the drawing is {mask.shape[1]}x{mask.shape[0]} but the canvas is {size}x{size}")
            if not mask.any():
                raise ValueError("the silhouette is empty")
        elif mode == "file":
            if not req.get("mask_png"):
                raise ValueError("upload a PNG silhouette or take one from a generated sprite")
            rgba = np.asarray(Image.open(io.BytesIO(base64.b64decode(req["mask_png"].split(",")[-1]))).convert("RGBA"))
            k = detect_scale(rgba)
            mask = rgba[::k, ::k, 3] >= 128
            if mask.shape != (size, size):
                raise ValueError(f"the silhouette is {mask.shape[1]}x{mask.shape[0]} but the canvas is {size}x{size}")
        stats = None
        with self.lock:
            if mode == "same":
                first = self.pipe.generate(text, 1, size, seed, use_filter=use_filter, **p)
                mask, stats, seed = first.index[0] >= 0, first.stats, seed + 1
            res = self.pipe.generate(text, n, size, seed, mask=mask, use_filter=use_filter, **p)
        stats = stats or (None if mask is not None else res.stats)
        return {"text": text, "size": size, "n": n, "seed": int(req.get("seed", 0)), "mode": mode, **p,
                "filter": use_filter, "sprites": [png_b64(s) for s in res.sprites],
                "masks": [png_b64(mask_rgba(x >= 0)) for x in res.index],
                "stats": None if stats is None else {k: stats[k] for k in ("sampled", "rejected", "reasons", "reject_rate")},
                "seconds": round(time.time() - t0, 2)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter console
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(200, app.info())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/api/generate":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            try:
                self._json(200, app.generate(json.loads(self.rfile.read(length) or b"{}")))
            except ValueError as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - report to the page instead of dropping the connection
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pixelsprite-ui", description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--model", default=DEFAULT_REPO, help="Hugging Face repo id or local folder")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t = time.time()
    app = App(args.model, args.device)
    app.generate({"text": "frost sword", "n": 2})  # warm-up: loads the text encoder
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"ready in {time.time() - t:.0f}s on {app.pipe.device} -> http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
