"""Denoiser U-Net with AdaGN conditioning (ROADMAP 4.7, [FROZEN signature]). Implemented by CP3-C."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class UNetConfig:
    """U-Net hyperparameters (4.7). `ch` depends on H: [64,128,256] for 84, [64,128,256,256] for 96."""

    C: int
    K: int
    H: int
    ch: list[int]
    n_res: int = 2
    emb_dim: int = 256
    n_levels: int = 10
    attn_lowest: bool = False
    dropout: float = 0.0


class UNet(nn.Module):
    """Predicts the EDM network output F from the noisy target, context frames, and conditioning."""

    def __init__(self, cfg: UNetConfig) -> None:
        super().__init__()
        raise NotImplementedError("CP3-C")

    def forward(self, x: Tensor, c_noise: Tensor, context: Tensor, actions: torch.LongTensor,
                ctx_level: torch.LongTensor) -> Tensor:
        """x (B,C,H,W), c_noise (B,), context (B,K*C,H,W), actions (B,K), ctx_level (B,K) -> (B,C,H,W)."""
        raise NotImplementedError("CP3-C")
