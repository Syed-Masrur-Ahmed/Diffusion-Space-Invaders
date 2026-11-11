"""Autoregressive rollouts and per-horizon metrics (ROADMAP 4.11). Owned by CP4-A."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from wm.data.reader import ShardReader, list_shards
from wm.eval.metrics import METRICS
from wm.inference import WorldModelProtocol


def rollout(model: WorldModelProtocol, frames0: Tensor, actions: Tensor, T: int, **sample_kwargs) -> Tensor:
    """Generate T frames from real context frames0 (B,K,C,H,W) and actions (B,K+T-1) -> (B,T,C,H,W).

    Generated frame g_j (j >= K) is predicted from the last K frames (real at first, generated
    afterwards) and actions a_{j-K}..a_{j-1}; output index t holds g_{K+t}, horizon h = t + 1.
    `sample_kwargs` (num_steps, cfg_scale, ctx_level, generator) go to every predict_next call.
    """
    B, K = frames0.shape[:2]
    if K != model.K or frames0.shape[2:] != (model.C, model.H, model.W):
        raise ValueError(f"frames0 must be (B,{model.K},{model.C},{model.H},{model.W}), got {tuple(frames0.shape)}")
    if actions.shape != (B, K + T - 1):
        raise ValueError(f"actions must be (B, K+T-1) = ({B}, {K + T - 1}), got {tuple(actions.shape)}")
    context, out = frames0, []
    for t in range(T):
        frame = model.predict_next(context, actions[:, t:t + K], **sample_kwargs)
        out.append(frame)
        context = torch.cat([context[:, 1:], frame[:, None]], dim=1)
    return torch.stack(out, dim=1)


def horizon_metrics(pred: Tensor, target: Tensor, metric_names: list[str], batch_size: int = 1024) -> dict[str, Tensor]:
    """Per-frame metrics {name: (B,T)} between generated and real frames, both (B,T,C,H,W) in [-1,1]."""
    B, T = pred.shape[:2]
    p, g = (pred.flatten(0, 1) + 1) / 2, (target.flatten(0, 1) + 1) / 2
    return {name: torch.cat([METRICS[name](p[i:i + batch_size], g[i:i + batch_size]).cpu()
                             for i in range(0, B * T, batch_size)]).view(B, T)
            for name in metric_names}


@dataclass
class Trajectories:
    """Real eval trajectories: `length` consecutive rows from each of n episodes."""

    frames: np.ndarray  # uint8 (n, length, H, W, 3), as stored
    actions: np.ndarray  # int64 (n, length - 1); actions[:, i] produces frames[:, i + 1]
    episodes: list[tuple[str, int]]  # (shard path, shard-local episode index)


def load_trajectories(dataset_dirs: list[str | Path], n: int, length: int, start: int = 0) -> Trajectories:
    """Load rows start..start+length-1 of n episodes, spread evenly over the episodes long enough for them."""
    eligible = []
    for path in sorted(p for d in dataset_dirs for p in list_shards(d)):
        reader = ShardReader(path)
        eligible += [(reader, e) for e in np.flatnonzero(reader.episodes["length"] >= start + length)]
    if n > len(eligible):
        raise ValueError(f"asked for {n} trajectories of {length} rows, but only {len(eligible)} episodes have {start + length} rows")
    picks = [eligible[i] for i in np.round(np.linspace(0, len(eligible) - 1, n)).astype(int)]
    frames, actions = [], []
    for reader, e in picks:
        first = int(reader.episodes["start"][e]) + start
        rows = slice(first, first + length)
        frames.append(reader.read("frames", rows))
        actions.append(reader.read("actions", slice(rows.start, rows.stop - 1)).astype(np.int64))
    for reader, _ in eligible:
        reader.close()
    return Trajectories(np.stack(frames), np.stack(actions), [(str(r.path), int(e)) for r, e in picks])
