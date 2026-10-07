"""Inference API (ROADMAP 4.10, [FROZEN]). WorldModel is implemented by CP4-C."""

from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor


class WorldModelProtocol(Protocol):
    """Anything that predicts the next frame from K context frames and actions."""

    K: int
    C: int
    H: int
    W: int

    def predict_next(self, context: Tensor, actions: Tensor, num_steps: int | None = None,
                     cfg_scale: float | None = None, ctx_level: int = 0,
                     generator: torch.Generator | None = None) -> Tensor:
        """context (B,K,C,H,W) and actions (B,K) -> next frame (B,C,H,W), all frames in [-1,1].

        actions[:, -1] is the action that produces the new frame.
        """
        ...


class WorldModel(WorldModelProtocol):
    """Trained diffusion world model; samples with EMA weights under eval() and inference_mode."""

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device | None = None,
                        use_ema: bool = True, compile: bool = False) -> "WorldModel":
        """Load a model from a training checkpoint."""
        raise NotImplementedError("CP4-C")

    def predict_next(self, context: Tensor, actions: Tensor, num_steps: int | None = None,
                     cfg_scale: float | None = None, ctx_level: int = 0,
                     generator: torch.Generator | None = None) -> Tensor:
        raise NotImplementedError("CP4-C")
