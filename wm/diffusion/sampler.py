"""Karras-schedule Euler/Heun sampler with action CFG (ROADMAP 4.8, 4.9). Implemented by CP2-C."""

from dataclasses import dataclass

import torch
from torch import Tensor

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
    raise NotImplementedError("CP2-C")


def sample(denoiser: EDMDenoiser, context: Tensor, actions: torch.LongTensor,
           ctx_level: torch.LongTensor, cfg: SamplerConfig, cfg_scale: float = 1.0,
           generator: torch.Generator | None = None) -> Tensor:
    """Generate the next frame (B,C,H,W) in [-1,1]; cfg_scale=1 takes the single-pass path."""
    raise NotImplementedError("CP2-C")
