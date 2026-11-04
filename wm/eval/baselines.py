"""Baseline world models (ROADMAP 4.10, 4.11). CopyLastFrame is owned by CP3-A."""

import torch
from torch import Tensor

from wm.inference import WorldModelProtocol


class CopyLastFrame(WorldModelProtocol):
    """Predicts that the next frame equals the last context frame."""

    def __init__(self, K: int, C: int, H: int, W: int) -> None:
        self.K, self.C, self.H, self.W = K, C, H, W

    def predict_next(self, context: Tensor, actions: Tensor, num_steps: int | None = None,
                     cfg_scale: float | None = None, ctx_level: int = 0,
                     generator: torch.Generator | None = None) -> Tensor:
        """Return a copy of context[:, -1]; actions and sampler arguments are ignored."""
        return context[:, -1].clone()
