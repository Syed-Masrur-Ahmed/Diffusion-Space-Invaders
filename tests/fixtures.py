"""Shared test fixtures (ROADMAP 4.13). Workflows may append new fixtures at the end of this file."""

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wm.data.schema import NUM_ACTIONS


def make_fake_shard(path: str | Path, n_episodes: int, lengths: list[int], H: int, W: int,
                    encode_rows: bool = True) -> Path:
    """Write a tiny synthetic shard with ShardWriter. Implemented by CP1-B.

    With `encode_rows`, each frame's pixels encode its identity, so alignment tests can
    decode exactly which row they got: R = row % 256, G = (row // 256) % 256,
    B = episode_idx % 256, where `row` is the global row index in the shard.
    """
    raise NotImplementedError("CP1-B")


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
