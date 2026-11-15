"""pixelsprite: text -> 16x16 / 24x24 / 32x32 pixel-art sprites with a small autoregressive transformer."""
from .images import checker_composite, mask_rgba, read_mask, rgba_from_index, sheet
from .model import CanvasConfig, CanvasGPT2
from .pipeline import DEFAULT_REPO, Generation, SpritePipeline
from .sampling import sample, split_pick

__version__ = "0.1.0"
__all__ = ["SpritePipeline", "Generation", "CanvasGPT2", "CanvasConfig", "sample", "split_pick", "rgba_from_index",
           "read_mask", "mask_rgba", "checker_composite", "sheet", "DEFAULT_REPO", "__version__"]
