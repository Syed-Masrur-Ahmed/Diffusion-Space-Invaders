"""Shared test fixtures (ROADMAP 4.13). Workflows may append new fixtures at the end of this file."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wm.data.reader import list_shards
from wm.data.schema import END_TERMINATED, END_TRUNCATED, NUM_ACTIONS, RAM_SIZE, SOURCE_AGENT
from wm.data.writer import ShardWriter


def make_fake_shard(path: str | Path, n_episodes: int, lengths: list[int], H: int, W: int,
                    encode_rows: bool = True) -> Path:
    """Write a tiny synthetic shard with ShardWriter into directory `path`; returns the shard path.

    `lengths` are in rows (T + 1 for T steps), as in `episodes/length`. With `encode_rows`,
    each frame's pixels encode its identity, so alignment tests can decode exactly which row
    they got: R = row % 256, G = (row // 256) % 256, B = episode_idx % 256, where `row` is the
    global row index in the shard (see `decode_frame`). Other fields are deterministic
    functions of (row, episode, step): action = (5 * row + e) % 6, reward = 5 * (row % 4),
    lives = 3 - min(step // 10, 3), ram[j] = (row + j) % 256.
    """
    if len(lengths) != n_episodes or min(lengths, default=1) < 1:
        raise ValueError(f"need {n_episodes} episode lengths >= 1 (in rows), got {lengths}")
    rng = np.random.default_rng(0)
    attrs = {"env_id": "fake", "frameskip": 4, "maxpool": True, "resolution": H,
             "resize_method": "area", "collector_config": {"fixture": "make_fake_shard"}}

    def row_data(row: int, e: int, step: int) -> tuple[np.ndarray, np.ndarray, int]:
        frame = (encode_frame(row, e, H, W) if encode_rows
                 else rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
        ram = ((row + np.arange(RAM_SIZE)) % 256).astype(np.uint8)
        return frame, ram, 3 - min(step // 10, 3)

    with ShardWriter(path, worker=0, frame_shape=(H, W, 3), attrs=attrs) as w:
        row = 0
        for e, n_rows in enumerate(lengths):
            source = e % 3
            w.begin_episode(*row_data(row, e, 0), seed=1000 + e, noops=e % 31, source=source,
                            epsilon=0.2 if source == SOURCE_AGENT else -1.0)
            for step in range(1, n_rows):
                prev = row
                row += 1
                w.add_step((5 * prev + e) % NUM_ACTIONS, 5.0 * (prev % 4), *row_data(row, e, step))
            row += 1
            terminated = e % 2 == 0
            w.end_episode(END_TERMINATED if terminated else END_TRUNCATED, terminated)
    return list_shards(path)[0]


class TinyDenoiserNet(nn.Module):
    """Tiny stand-in for UNet with the same forward signature (4.7), for fast CPU tests."""

    def __init__(self, C: int = 3, K: int = 4, n_levels: int = 10, hidden: int = 16) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(C * (K + 1), hidden, 3, padding=1)
        self.conv_mid = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.noise = nn.Linear(1, hidden)
        self.action_emb = nn.Embedding(NUM_ACTIONS + 1, 8)
        self.action = nn.Linear(K * 8, hidden)
        self.level_emb = nn.Embedding(n_levels, 8)
        self.level = nn.Linear(K * 8, hidden)
        self.out = nn.Conv2d(hidden, C, 3, padding=1)
        # Zero output like the real U-Net, so F = 0 and D = c_skip * x at init.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, c_noise: Tensor, context: Tensor, actions: torch.LongTensor,
                ctx_level: torch.LongTensor) -> Tensor:
        """x (B,C,H,W), c_noise (B,), context (B,K*C,H,W), actions and ctx_level (B,K) -> (B,C,H,W)."""
        bias = (self.noise(c_noise[:, None])
                + self.action(self.action_emb(actions).flatten(1))
                + self.level(self.level_emb(ctx_level).flatten(1)))
        h = F.silu(self.conv_in(torch.cat([x, context], dim=1)) + bias[:, :, None, None])
        h = F.silu(self.conv_mid(h))
        return self.out(h)


def encode_frame(row: int, episode_idx: int, H: int, W: int) -> np.ndarray:
    """Frame whose pixels encode its row: R = row % 256, G = (row // 256) % 256, B = episode_idx % 256."""
    rgb = np.array([row % 256, (row // 256) % 256, episode_idx % 256], dtype=np.uint8)
    return np.broadcast_to(rgb, (H, W, 3)).copy()


def decode_frame(frame: np.ndarray) -> tuple[int, int]:
    """(row % 65536, episode_idx % 256) from a uint8 (H,W,3) frame written with `encode_frame`."""
    r, g, b = (int(v) for v in frame[0, 0])
    return r + 256 * g, b
