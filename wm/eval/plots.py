"""Metric-vs-horizon plots and rollout videos (ROADMAP 4.11). Owned by CP4-A."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from wm.eval.visualize import DIFF_GAIN, GAP, GAP_VALUE, _to_uint8

LABELS = {"psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑", "lpips": "LPIPS ↓"}


def plot_horizon(path: str | Path, curves: dict[str, dict[str, dict[str, list]]]) -> None:
    """One panel per metric, one line per model with a 95% CI band, log x axis.

    curves: {model: {metric: {"h": [...], "mean": [...], "ci95": [...]}}}.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless (Discovery)
    import matplotlib.pyplot as plt

    metrics = list(next(iter(curves.values())))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.6), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for name, per_metric in curves.items():
            c = per_metric[metric]
            h, mean, ci = np.asarray(c["h"]), np.asarray(c["mean"]), np.asarray(c["ci95"])
            ax.plot(h, mean, label=name, linewidth=1.5)
            ax.fill_between(h, mean - ci, mean + ci, alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("rollout horizon h (agent steps)")
        ax.set_ylabel(LABELS.get(metric, metric))
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_rollout_video(path: str | Path, real: Tensor, gen: Tensor, fps: int = 15, scale: int = 4,
                       widen: int = 1) -> None:
    """MP4 of real | generated | |diff| x4, from (T,C,H,W) frames in [-1,1].

    Tiles are upscaled `scale`x (nearest); `widen` stretches them horizontally for display only.
    """
    import imageio.v2 as iio2

    r, g = _to_uint8(real), _to_uint8(gen)
    diff = ((r.int() - g.int()).abs() * DIFF_GAIN).clamp(max=255).to(torch.uint8)  # int: no uint8 wrap
    tiles = torch.stack([r, g, diff], dim=1)  # (T, 3 tiles, 3, H, W)
    tiles = tiles.repeat_interleave(scale, -2).repeat_interleave(scale * widen, -1).permute(0, 1, 3, 4, 2).numpy()
    T, n, th, tw, _ = tiles.shape
    video = np.full((T, th, n * (tw + GAP) - GAP, 3), GAP_VALUE, np.uint8)
    for i in range(n):
        video[:, :, i * (tw + GAP):i * (tw + GAP) + tw] = tiles[:, i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 keeps the exact size (x264 only needs even sizes, which ours are for even scale).
    with iio2.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in video:
            writer.append_data(frame)
