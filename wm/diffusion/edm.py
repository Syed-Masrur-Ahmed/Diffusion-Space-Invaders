"""EDM preconditioning, noise sampling, and loss (ROADMAP 4.6)."""

import math
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


def edm_coefficients(sigma: Tensor, sigma_data: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(c_skip, c_out, c_in, c_noise) for noise levels `sigma`, each the same shape as `sigma`."""
    norm = torch.sqrt(sigma**2 + sigma_data**2)
    return sigma_data**2 / norm**2, sigma * sigma_data / norm, 1 / norm, torch.log(sigma) / 4


class EDMDenoiser(nn.Module):
    """Wraps a network F with EDM preconditioning: D(x; sigma, cond) = c_skip x + c_out F(c_in x, ...)."""

    def __init__(self, net: nn.Module, cfg: EDMConfig) -> None:
        super().__init__()
        self.net = net
        self.cfg = cfg

    def _precondition(self, x: Tensor, sigma: Tensor, context: Tensor, actions: torch.LongTensor,
                      ctx_level: torch.LongTensor) -> tuple[Tensor, Tensor, Tensor]:
        """(F, c_skip, c_out), with the coefficients shaped (B,1,1,1)."""
        c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, self.cfg.sigma_data)
        c_skip, c_out, c_in = (c.view(-1, 1, 1, 1) for c in (c_skip, c_out, c_in))
        if context.dim() == 5:  # (B,K,C,H,W) from the dataset -> (B,K*C,H,W), oldest first
            context = context.flatten(1, 2)
        return self.net(c_in * x, c_noise, context, actions, ctx_level), c_skip, c_out

    def denoise(self, x: Tensor, sigma: Tensor, context: Tensor, actions: torch.LongTensor,
                ctx_level: torch.LongTensor) -> Tensor:
        """Denoised estimate D, shape (B,C,H,W); `sigma` is (B,) on x's device."""
        F, c_skip, c_out = self._precondition(x, sigma, context, actions, ctx_level)
        return c_skip * x + c_out * F

    def sample_sigma(self, n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        """Training noise levels: ln sigma ~ N(p_mean, p_std^2), clamped to [sigma_min, sigma_max]."""
        log_sigma = self.cfg.p_mean + self.cfg.p_std * torch.randn(n, device=device, dtype=dtype)
        return log_sigma.exp().clamp(self.cfg.sigma_min, self.cfg.sigma_max)

    def loss_terms(self, target: Tensor, context: Tensor, actions: torch.LongTensor,
                   ctx_level: torch.LongTensor) -> tuple[Tensor, Tensor]:
        """Per-sample losses (B,) and the sigmas (B,) they were drawn at, for loss-by-sigma logging."""
        sigma = self.sample_sigma(target.shape[0], target.device, target.dtype)
        x = target + sigma.view(-1, 1, 1, 1) * torch.randn_like(target)
        F, c_skip, c_out = self._precondition(x, sigma, context, actions, ctx_level)
        F_target = (target - c_skip * x) / c_out
        return ((F - F_target) ** 2).mean(dim=(1, 2, 3)), sigma

    def loss(self, target: Tensor, context: Tensor, actions: torch.LongTensor,
             ctx_level: torch.LongTensor) -> Tensor:
        """Scalar training loss; samples sigma and noise internally."""
        return self.loss_terms(target, context, actions, ctx_level)[0].mean()


def loss_by_sigma(losses: Tensor, sigma: Tensor, n_buckets: int = 8, sigma_min: float = 2e-3,
                  sigma_max: float = 20.0) -> dict[str, float]:
    """Mean loss per log-spaced sigma bucket, keyed "lo-hi"; empty buckets are left out."""
    edges = torch.logspace(math.log10(sigma_min), math.log10(sigma_max), n_buckets + 1,
                           dtype=torch.float64)
    # Interior edges only, so sigmas at or outside the range land in the first or last bucket.
    idx = torch.bucketize(sigma.detach().double().cpu(), edges[1:-1], right=True)
    losses = losses.detach().double().cpu()
    return {f"{edges[i]:.3g}-{edges[i + 1]:.3g}": losses[idx == i].mean().item()
            for i in range(n_buckets) if (idx == i).any()}
