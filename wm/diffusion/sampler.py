"""Karras-schedule Euler/Heun sampler with action CFG (ROADMAP 4.8, 4.9)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from wm.data.schema import NULL_ACTION
from wm.diffusion.edm import EDMDenoiser


@dataclass
class SamplerConfig:
    """Sampler settings (4.9)."""

    num_steps: int = 3
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: float = 7.0
    method: str = "euler"  # "euler" | "heun"


def karras_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: float) -> Tensor:
    """Decreasing Karras noise levels with a final 0 appended, length num_steps + 1."""
    if num_steps == 1:
        return torch.tensor([sigma_max, 0.0], dtype=torch.float64)
    ramp = torch.linspace(0, 1, num_steps, dtype=torch.float64)
    lo, hi = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    sigmas = (hi + ramp * (lo - hi)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _guided_denoise(denoiser: EDMDenoiser, x: Tensor, sigma: float, context: Tensor,
                    actions: torch.LongTensor, ctx_level: torch.LongTensor, cfg_scale: float) -> Tensor:
    """D(x; sigma), with action CFG D_u + w (D_c - D_u) from one doubled-batch call when w != 1."""
    B = x.shape[0]
    if cfg_scale == 1.0:
        return denoiser.denoise(x, x.new_full((B,), sigma), context, actions, ctx_level)
    D = denoiser.denoise(torch.cat([x, x]), x.new_full((2 * B,), sigma), torch.cat([context, context]),
                         torch.cat([actions, torch.full_like(actions, NULL_ACTION)]),
                         torch.cat([ctx_level, ctx_level]))
    D_c, D_u = D[:B], D[B:]
    return D_u + cfg_scale * (D_c - D_u)


@torch.no_grad()
def sample(denoiser: EDMDenoiser, context: Tensor, actions: torch.LongTensor,
           ctx_level: torch.LongTensor, cfg: SamplerConfig, cfg_scale: float = 1.0,
           generator: torch.Generator | None = None) -> Tensor:
    """Generate the next frame (B,C,H,W) in [-1,1]; cfg_scale=1 takes the single-pass path.

    `context` is (B,K,C,H,W) or flattened (B,K*C,H,W). The caller sets eval() and picks EMA weights.
    `generator` may live on any device; pass a CPU one for noise that is the same on every device.
    """
    if cfg.method not in ("euler", "heun"):
        raise ValueError(f"unknown sampler method {cfg.method!r}; use 'euler' or 'heun'")
    B, K = actions.shape
    C = context.shape[2] if context.dim() == 5 else context.shape[1] // K
    H, W = context.shape[-2:]
    sigmas = karras_sigmas(cfg.num_steps, cfg.sigma_min, cfg.sigma_max, cfg.rho).tolist()

    def D(x: Tensor, sigma: float) -> Tensor:
        return _guided_denoise(denoiser, x, sigma, context, actions, ctx_level, cfg_scale)

    # Noise is drawn on the generator's device, so a CPU generator gives the same x_0 on any device.
    noise_device = generator.device if generator is not None else context.device
    x = sigmas[0] * torch.randn((B, C, H, W), generator=generator, device=noise_device,
                                dtype=context.dtype).to(context.device)
    for s0, s1 in zip(sigmas[:-1], sigmas[1:]):
        d = (x - D(x, s0)) / s0
        x_next = x + (s1 - s0) * d
        if cfg.method == "heun" and s1 > 0:
            d_next = (x_next - D(x_next, s1)) / s1
            x_next = x + (s1 - s0) * (d + d_next) / 2
        x = x_next
    return x.clamp(-1, 1)
