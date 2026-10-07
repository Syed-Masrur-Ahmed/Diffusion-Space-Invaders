"""EDM preconditioning, noise sampling, and loss (ROADMAP 4.6). Implemented by CP2-C."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class EDMConfig:
    """EDM hyperparameters (4.6)."""

    sigma_data: float = 0.5
    p_mean: float = -0.4
    p_std: float = 1.2
    sigma_min: float = 2e-3  # training sigma clamp
    sigma_max: float = 20.0


class EDMDenoiser(nn.Module):
    """Wraps a network F with EDM preconditioning: D(x; sigma, cond) = c_skip x + c_out F(c_in x, ...)."""

    def __init__(self, net: nn.Module, cfg: EDMConfig) -> None:
        super().__init__()
        raise NotImplementedError("CP2-C")

    def denoise(self, x: Tensor, sigma: Tensor, context: Tensor, actions: torch.LongTensor,
                ctx_level: torch.LongTensor) -> Tensor:
        """Denoised estimate D, shape (B,C,H,W); `sigma` is (B,) on x's device."""
        raise NotImplementedError("CP2-C")

    def loss(self, target: Tensor, context: Tensor, actions: torch.LongTensor,
             ctx_level: torch.LongTensor) -> Tensor:
        """Scalar training loss; samples sigma and noise internally."""
        raise NotImplementedError("CP2-C")
