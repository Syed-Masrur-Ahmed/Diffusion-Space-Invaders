"""Baseline world models (ROADMAP 4.10, 4.11). CopyLastFrame is implemented by CP3-A."""

import torch
from torch import Tensor

from wm.inference import WorldModelProtocol


class CopyLastFrame(WorldModelProtocol):
    """Predicts that the next frame equals the last context frame."""

    def __init__(self, K: int, C: int, H: int, W: int) -> None:
        raise NotImplementedError("CP3-A")

    def predict_next(self, context: Tensor, actions: Tensor, num_steps: int | None = None,
                     cfg_scale: float | None = None, ctx_level: int = 0,
                     generator: torch.Generator | None = None) -> Tensor:
        raise NotImplementedError("CP3-A")
