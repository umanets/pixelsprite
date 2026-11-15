"""Unit tests on a tiny random model (CPU, a few seconds) + an optional test with the real weights:
set PIXELSPRITE_TEST_MODEL to a local model folder or a Hugging Face repo id to run it."""
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from pixelsprite import CanvasConfig, CanvasGPT2, SpritePipeline, read_mask, rgba_from_index
from pixelsprite.model import EDGE_BOUNDARY, EDGE_INNER, EDGE_OUTSIDE, EDGE_UNKNOWN, edge_flags
from pixelsprite.sampling import components, rejected, sample, split_pick

K_ = 32


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return CanvasGPT2(CanvasConfig(n_colors=K_, d_model=32, n_layers=2, n_heads=4, d_ff=64)).eval()


def rand_index(b, s, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, K_, (b, s, s))
    idx[rng.random((b, s, s)) < 0.5] = -1
    return torch.as_tensor(idx)


def teacher_inputs(model, idx, known):
    """Inputs for teacher forcing: previous token (BOS first), row, column, edge flag of every pixel."""
    b, s = idx.shape[:2]
    flat = idx.view(b, -1)
    tok = torch.where(flat < 0, torch.full_like(flat, model.EMPTY), flat)
    prev = torch.cat([torch.full((b, 1), model.BOS), tok[:, :-1]], 1)
    ar = torch.arange(s * s)
    rows, cols = (ar // s).expand(b, -1), (ar % s).expand(b, -1)
    mask = idx >= 0
    return mask, prev, rows, cols, edge_flags(mask, known)


@pytest.mark.parametrize("size", [16, 32])
def test_kv_cache_matches_forward(model, size):
    idx = rand_index(3, size)
    known = torch.tensor([True, False, True])
    mask, prev, rows, cols, edge = teacher_inputs(model, idx, known)
    cond = model.cond(torch.randn(3, 384))
    with torch.no_grad():
        full = model(cond, size, mask, known, prev, rows, cols, edge)
        n = size * size
        cache = model.new_cache(3, 2 + size + n, "cpu", torch.float32)
        x = torch.cat([model.prefix(cond, size, mask, known), model.tokens(prev[:, :1], rows[:, :1], cols[:, :1], edge[:, :1])], 1)
        outs, pos = [model.step(x, cache, 0)], x.shape[1]
        for t in range(1, n):
            outs.append(model.step(model.tokens(prev[:, t:t + 1], rows[:, t:t + 1], cols[:, t:t + 1], edge[:, t:t + 1]), cache, pos))
            pos += 1
    assert torch.allclose(torch.stack(outs, 1), full, atol=1e-4)


def test_edge_flags():
    m = torch.zeros(2, 16, 16, dtype=torch.bool)
    m[:, 4:9, 4:9] = True
    e = edge_flags(m, torch.tensor([True, False])).view(2, 16, 16)
    assert e[0, 6, 6] == EDGE_INNER and e[0, 4, 6] == EDGE_BOUNDARY and e[0, 0, 0] == EDGE_OUTSIDE
    assert (e[1] == EDGE_UNKNOWN).all()


def test_split_pick_forcing():
    g = torch.Generator().manual_seed(0)
    lc = torch.randn(64, K_ + 1)
    force_e = torch.zeros(64, dtype=torch.bool)
    force_e[:32] = True
    tok = split_pick(lc, None, 1.0, 1.0, 1.0, 1.0, g, force_e, ~force_e, empty_id=K_)
    assert (tok[:32] == K_).all() and (tok[32:] < K_).all()


def test_split_pick_guidance_one_equals_conditional():
    """With g_shape = g_colour = 1 the guided distribution is the conditional one, whatever the unconditional is."""
    lc, lu = torch.randn(4000, K_ + 1), torch.randn(4000, K_ + 1)
    a = split_pick(lc, lu, 1.0, 1.0, 1.0, 1.0, torch.Generator().manual_seed(1), empty_id=K_)
    b = split_pick(lc, None, 1.0, 1.0, 1.0, 1.0, torch.Generator().manual_seed(1), empty_id=K_)
    assert (a == b).float().mean() > 0.99


def test_sample_mask_window_seed(model):
    mask = np.zeros((16, 16), bool)
    mask[3:12, 5:11] = True
    text = torch.randn(1, 384, generator=torch.Generator().manual_seed(1))
    a, _ = sample(model, 3, text, size=16, mask=mask, seed=3)
    b, _ = sample(model, 3, text, size=16, mask=mask, seed=3)
    assert np.array_equal(a, b)
    assert all(np.array_equal(x >= 0, mask) for x in a), "a given mask is followed exactly"
    w, _ = sample(model, 4, None, size=32, window=24, seed=0, use_filter=True, max_rounds=400, chunk=8)
    assert w.shape == (4, 24, 24) and all(rejected(x >= 0) is None for x in w)


def test_components_and_filter():
    m = np.zeros((16, 16), bool)
    m[2:6, 2:6] = True          # 16 px
    m[10, 10] = True            # 1 px, detached
    m[7, 7] = True              # two steps from the corner (5, 5): a separate piece
    assert list(components(m)) == [16, 1, 1]
    assert rejected(m) == "detached_1_2px"
    assert rejected(np.zeros((16, 16), bool)) == "empty"
    m2 = np.zeros((16, 16), bool)
    m2[2:6, 2:6] = True
    m2[6, 6] = True             # diagonal neighbour joins the square (8-connectivity)
    assert list(components(m2)) == [17] and rejected(m2) is None


def test_rgba_and_read_mask(tmp_path):
    pal = np.random.default_rng(0).integers(0, 256, (K_, 3), dtype=np.uint8)
    idx = rand_index(1, 16, seed=2)[0].numpy()
    rgba = rgba_from_index(idx, pal)
    assert np.array_equal(rgba[..., 3] == 255, idx >= 0) and np.array_equal(rgba[idx >= 0, :3], pal[idx[idx >= 0]])
    Image.fromarray(rgba.repeat(8, 0).repeat(8, 1), "RGBA").save(tmp_path / "m.png")
    assert np.array_equal(read_mask(tmp_path / "m.png"), idx >= 0)


def test_from_pretrained_roundtrip(model, tmp_path):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"model": model.cfg.to_dict(), "text_model": "none"}))
    pal = np.random.default_rng(0).integers(0, 256, (K_, 3)).tolist()
    (tmp_path / "palette.json").write_text(json.dumps({"colors": pal}))
    pipe = SpritePipeline.from_pretrained(tmp_path, device="cpu")
    r1 = pipe.generate(None, 3, 16, seed=5)
    r2 = sample(model, 3, None, size=16, seed=5, chunk=32)[0]
    assert np.array_equal(r1.index, r2) and len(r1.sprites) == 3 and r1.sprites[0].shape == (16, 16, 4)
    r24 = pipe.generate(None, 2, 24, seed=1)
    assert r24.index.shape == (2, 24, 24)


@pytest.mark.skipif(not os.environ.get("PIXELSPRITE_TEST_MODEL"), reason="set PIXELSPRITE_TEST_MODEL to run")
def test_real_model():
    pipe = SpritePipeline.from_pretrained(os.environ["PIXELSPRITE_TEST_MODEL"])
    r = pipe.generate("frost sword", 4, 16, seed=0)
    assert len(r.sprites) == 4 and all(s.shape == (16, 16, 4) for s in r.sprites)
    mask = r.index[0] >= 0
    r2 = pipe.generate("ледяной меч", 2, 16, seed=1, mask=mask)
    assert all(np.array_equal(s[..., 3] == 255, mask) for s in r2.sprites)
