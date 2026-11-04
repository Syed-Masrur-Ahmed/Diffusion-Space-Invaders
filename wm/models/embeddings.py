"""Noise, action, and context-level embeddings for the U-Net (ROADMAP 4.7). Owned by CP3-C."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wm.data.schema import NUM_ACTIONS


class FourierFeatures(nn.Module):
    """[cos, sin](2π·f·t) for `n_freqs` fixed random frequencies f ~ N(0, scale²), as in EDM."""

    def __init__(self, n_freqs: int = 16, scale: float = 16.0) -> None:
        super().__init__()
        # A buffer, so the frequencies are saved in checkpoints but never trained.
        self.register_buffer("freqs", torch.randn(n_freqs) * scale)

    def forward(self, t: Tensor) -> Tensor:
        """t (B,) -> (B, 2·n_freqs)."""
        angles = t.float()[:, None] * (2 * math.pi * self.freqs)
        return torch.cat([angles.cos(), angles.sin()], dim=1)


class ConditionEmbedding(nn.Module):
    """emb = SiLU(noise(c_noise) + actions(actions) + level(ctx_level)), shape (B, emb_dim)."""

    def __init__(self, K: int, n_levels: int, emb_dim: int = 256) -> None:
        super().__init__()
        self.fourier = FourierFeatures()
        self.noise = nn.Sequential(nn.Linear(2 * self.fourier.freqs.numel(), emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim))
        self.action_emb = nn.Embedding(NUM_ACTIONS + 1, 64)
        self.action_pos = nn.Parameter(torch.zeros(K, 64))
        self.action = nn.Linear(K * 64, emb_dim)
        self.level_emb = nn.Embedding(n_levels, 32)
        self.level = nn.Linear(K * 32, emb_dim)

    def forward(self, c_noise: Tensor, actions: torch.LongTensor, ctx_level: torch.LongTensor) -> Tensor:
        """c_noise (B,), actions (B,K) in 0..NUM_ACTIONS, ctx_level (B,K) -> (B, emb_dim)."""
        emb = (self.noise(self.fourier(c_noise))
               + self.action((self.action_emb(actions) + self.action_pos).flatten(1))
               + self.level(self.level_emb(ctx_level).flatten(1)))
        return F.silu(emb)
