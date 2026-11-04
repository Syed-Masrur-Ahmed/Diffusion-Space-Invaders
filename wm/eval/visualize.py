"""Sample grids for single-step evaluation. Owned by CP3-A."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from torch import Tensor

SCALE = 4  # nearest-neighbour upscaling of each tile
GAP = 2  # pixels between tiles
GAP_VALUE = 64  # dark grey, so tiles stay apart on black game frames
DIFF_GAIN = 4


def _to_uint8(x: Tensor) -> Tensor:
    """[-1,1] float (..., C, H, W) -> uint8 (..., 3, H, W); greyscale is repeated to RGB."""
    x = ((x.detach().float().cpu() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return x.expand(*x.shape[:-3], 3, *x.shape[-2:]) if x.shape[-3] == 1 else x


def save_grid(path: str | Path, context: Tensor, target: Tensor, pred: Tensor) -> None:
    """Save a PNG with one row per sample: last context frame | target | prediction | |diff| x4.

    context (B,K,C,H,W), target and pred (B,C,H,W), all in [-1,1]; C is 3 or 1.
    """
    last, tgt, prd = _to_uint8(context[:, -1]), _to_uint8(target), _to_uint8(pred)
    diff = ((tgt.int() - prd.int()).abs() * DIFF_GAIN).clamp(max=255).to(torch.uint8)  # int: no uint8 wrap
    tiles = torch.stack([last, tgt, prd, diff], dim=1)  # (B, 4, 3, H, W)
    tiles = tiles.repeat_interleave(SCALE, -2).repeat_interleave(SCALE, -1).permute(0, 1, 3, 4, 2).numpy()

    B, cols, th, tw, _ = tiles.shape
    img = np.full((B * (th + GAP) - GAP, cols * (tw + GAP) - GAP, 3), GAP_VALUE, np.uint8)
    for r in range(B):
        for c in range(cols):
            y, x = r * (th + GAP), c * (tw + GAP)
            img[y:y + th, x:x + tw] = tiles[r, c]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, img)
