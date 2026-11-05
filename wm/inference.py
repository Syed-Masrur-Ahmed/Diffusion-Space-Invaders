"""Inference API (ROADMAP 4.10, [FROZEN]): load a training checkpoint and predict the next frame."""

import dataclasses
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from wm.config import from_dict
from wm.diffusion.edm import EDMDenoiser
from wm.diffusion.sampler import SamplerConfig, sample

FRAME_SIZE = 84  # model frames are 84x84 (D10)


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

    def __init__(self, denoiser: EDMDenoiser, sampler: SamplerConfig, K: int, C: int, H: int, W: int,
                 cfg_scale: float = 1.0) -> None:
        self.denoiser = denoiser.eval()
        self.sampler, self.cfg_scale = sampler, cfg_scale
        self.K, self.C, self.H, self.W = K, C, H, W

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device | None = None,
                        use_ema: bool = True, compile: bool = False) -> "WorldModel":
        """Load a model from a training checkpoint (4.12); `use_ema` picks `ema` over `model` weights."""
        from wm.train.trainer import RunConfig, build_net  # the trainer's config layout and net builder

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # config and RNG states are not tensors
        cfg = from_dict(RunConfig, ckpt["config"])
        if isinstance(cfg.model.ch, dict):  # configs/model/*.yaml style: ch keyed by H
            cfg.model.ch = cfg.model.ch[FRAME_SIZE]
        C = 1 if cfg.data.grayscale else 3
        net = build_net(cfg.model, C, cfg.data.K, FRAME_SIZE)
        net.load_state_dict(ckpt["ema" if use_ema else "model"], strict=True)
        net.requires_grad_(False).to(device)
        if compile:
            net = torch.compile(net)
        return cls(EDMDenoiser(net, cfg.edm), cfg.sampler, cfg.data.K, C, FRAME_SIZE, FRAME_SIZE)

    def predict_next(self, context: Tensor, actions: Tensor, num_steps: int | None = None,
                     cfg_scale: float | None = None, ctx_level: int = 0,
                     generator: torch.Generator | None = None) -> Tensor:
        """context (B,K,C,H,W) and actions (B,K) -> next frame (B,C,H,W) in [-1,1]; None means the checkpoint default."""
        sampler = self.sampler if num_steps is None else dataclasses.replace(self.sampler, num_steps=num_steps)
        level = torch.full_like(actions, ctx_level, dtype=torch.long)
        self.denoiser.eval()
        with torch.inference_mode():
            return sample(self.denoiser, context, actions.long(), level, sampler,
                          cfg_scale=self.cfg_scale if cfg_scale is None else cfg_scale, generator=generator)
