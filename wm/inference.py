"""Inference API (ROADMAP 4.10, [FROZEN]): load a training checkpoint and predict the next frame."""

import contextlib
import dataclasses
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn

from wm.config import from_dict
from wm.diffusion.edm import EDMDenoiser
from wm.diffusion.sampler import SamplerConfig, sample

FRAME_SIZE = 84  # model frames are 84x84 (D10)
_AMP_DTYPES = (torch.float16, torch.bfloat16)


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


def _channels_last_inputs(module: nn.Module, args: tuple) -> tuple:
    """Conv inputs as channels-last; actions and noise embeddings stay 1-D or 2-D."""
    x, c_noise, context, actions, ctx_level = args
    return (x.contiguous(memory_format=torch.channels_last), c_noise,
            context.contiguous(memory_format=torch.channels_last), actions, ctx_level)


def _check_amp(device: torch.device, amp_dtype: torch.dtype | None) -> torch.dtype | None:
    """fp16 or bf16 autocast, or None for fp32. bf16 on CUDA is refused when the GPU cannot run it."""
    if amp_dtype is None:
        return None
    if amp_dtype not in _AMP_DTYPES:
        raise ValueError(f"amp_dtype must be float16 or bfloat16, got {amp_dtype}")
    if device.type == "cuda" and amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 autocast was requested but this CUDA device does not support it")
    return amp_dtype


class WorldModel(WorldModelProtocol):
    """Trained diffusion world model; samples with EMA weights under eval() and inference_mode."""

    def __init__(self, denoiser: EDMDenoiser, sampler: SamplerConfig, K: int, C: int, H: int, W: int,
                 cfg_scale: float = 1.0, *, amp_dtype: torch.dtype | None = None, amp_device: str | None = None,
                 channels_last: bool = False) -> None:
        self.denoiser = denoiser.eval()
        self.sampler, self.cfg_scale = sampler, cfg_scale
        self.K, self.C, self.H, self.W = K, C, H, W
        self.amp_dtype, self.amp_device, self.channels_last = amp_dtype, amp_device, channels_last

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device | None = None,
                        use_ema: bool = True, compile: bool = False, *, channels_last: bool = True,
                        amp_dtype: torch.dtype | None = None) -> "WorldModel":
        """Load a model from a training checkpoint (4.12); `use_ema` picks `ema` over `model` weights.

        Positional arguments are the frozen 4.10 signature. `channels_last` (default) and `amp_dtype`
        (None, float16, or bfloat16) are keyword-only. Autocast is internal: `predict_next` still
        returns float32. `compile` wraps the net after the channels-last conversion.
        """
        from wm.train.trainer import RunConfig, build_net  # the trainer's config layout and net builder

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        amp_dtype = _check_amp(device, amp_dtype)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # config and RNG states are not tensors
        cfg = from_dict(RunConfig, ckpt["config"])
        if isinstance(cfg.model.ch, dict):  # configs/model/*.yaml style: ch keyed by H
            cfg.model.ch = cfg.model.ch[FRAME_SIZE]
        C = 1 if cfg.data.grayscale else 3
        net = build_net(cfg.model, C, cfg.data.K, FRAME_SIZE)
        net.load_state_dict(ckpt["ema" if use_ema else "model"], strict=True)
        net.requires_grad_(False).to(device)
        if channels_last:
            net.to(memory_format=torch.channels_last)
        if compile:
            net = torch.compile(net)
        if channels_last:  # on the outer module, so it still runs when `net` is a compiled wrapper
            net.register_forward_pre_hook(_channels_last_inputs)
        return cls(EDMDenoiser(net, cfg.edm), cfg.sampler, cfg.data.K, C, FRAME_SIZE, FRAME_SIZE,
                   amp_dtype=amp_dtype, amp_device=device.type, channels_last=channels_last)

    def predict_next(self, context: Tensor, actions: Tensor, num_steps: int | None = None,
                     cfg_scale: float | None = None, ctx_level: int = 0,
                     generator: torch.Generator | None = None) -> Tensor:
        """context (B,K,C,H,W) and actions (B,K) -> next frame (B,C,H,W) in [-1,1]; None means the checkpoint default."""
        sampler = self.sampler if num_steps is None else dataclasses.replace(self.sampler, num_steps=num_steps)
        level = torch.full_like(actions, ctx_level, dtype=torch.long)
        self.denoiser.eval()
        autocast = (torch.autocast(self.amp_device or "cpu", dtype=self.amp_dtype) if self.amp_dtype is not None
                    else contextlib.nullcontext())
        with torch.inference_mode(), autocast:
            out = sample(self.denoiser, context, actions.long(), level, sampler,
                         cfg_scale=self.cfg_scale if cfg_scale is None else cfg_scale, generator=generator)
        return out.to(dtype=context.dtype).contiguous()
