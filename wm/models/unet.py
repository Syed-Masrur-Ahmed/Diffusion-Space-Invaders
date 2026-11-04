"""Denoiser U-Net with AdaGN conditioning (ROADMAP 4.7, [FROZEN signature]). Implemented by CP3-C."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wm.config import from_dict
from wm.models.embeddings import ConditionEmbedding


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


def unet_config(model: dict[str, Any], C: int, K: int, H: int) -> UNetConfig:
    """UNetConfig from a `configs/model/*.yaml` dict plus the data's C, K, H; `ch` may map H -> list."""
    ch = model["ch"]
    if isinstance(ch, dict):
        if H not in ch:
            raise KeyError(f"model config has no ch for H={H} (has {sorted(ch)})")
        ch = ch[H]
    return from_dict(UNetConfig, {**model, "ch": ch, "C": C, "K": K, "H": H})


def _zero(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


def _group_norm(C: int, affine: bool = True) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, C), C, affine=affine)


class AdaGN(nn.Module):
    """GroupNorm without affine, then h·(1 + scale) + shift with (scale, shift) from the embedding."""

    def __init__(self, C: int, emb_dim: int) -> None:
        super().__init__()
        self.norm = _group_norm(C, affine=False)
        self.proj = _zero(nn.Linear(emb_dim, 2 * C))

    def forward(self, h: Tensor, emb: Tensor) -> Tensor:
        scale, shift = self.proj(emb)[:, :, None, None].chunk(2, dim=1)
        return self.norm(h) * (1 + scale) + shift


class ResBlock(nn.Module):
    """AdaGN → SiLU → Conv3×3 → AdaGN → SiLU → Dropout → Conv3×3 (zero-init), plus a skip."""

    def __init__(self, c_in: int, c_out: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = AdaGN(c_in, emb_dim)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = AdaGN(c_out, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _zero(nn.Conv2d(c_out, c_out, 3, padding=1))
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, h: Tensor, emb: Tensor) -> Tensor:
        r = self.conv1(F.silu(self.norm1(h, emb)))
        r = self.conv2(self.dropout(F.silu(self.norm2(r, emb))))
        return self.skip(h) + r


class SelfAttention(nn.Module):
    """Multi-head self-attention over all positions, residual with a zero-initialized output projection."""

    def __init__(self, C: int) -> None:
        super().__init__()
        self.heads = C // 64 if C % 64 == 0 else 1
        self.norm = _group_norm(C)
        self.qkv = nn.Conv2d(C, 3 * C, 1)
        self.out = _zero(nn.Conv2d(C, C, 1))

    def forward(self, h: Tensor) -> Tensor:
        B, C, H, W = h.shape
        qkv = self.qkv(self.norm(h)).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.transpose(-1, -2).unbind(1)  # each (B, heads, H·W, C/heads)
        a = F.scaled_dot_product_attention(q, k, v)
        return h + self.out(a.transpose(-1, -2).reshape(B, C, H, W))


class UNet(nn.Module):
    """Predicts the EDM network output F from the noisy target, context frames, and conditioning."""

    def __init__(self, cfg: UNetConfig) -> None:
        super().__init__()
        factor = 2 ** (len(cfg.ch) - 1)
        if cfg.H % factor:
            raise ValueError(f"H={cfg.H} must be divisible by 2**(len(ch) - 1) = {factor} for ch={cfg.ch}")
        self.cfg = cfg
        self.emb = ConditionEmbedding(cfg.K, cfg.n_levels, cfg.emb_dim)
        self.conv_in = nn.Conv2d(cfg.C * (cfg.K + 1), cfg.ch[0], 3, padding=1)

        def block(c_in: int, c_out: int) -> ResBlock:
            return ResBlock(c_in, c_out, cfg.emb_dim, cfg.dropout)

        # Encoder: n_res blocks per level, each output kept as a skip; stride-2 conv between levels.
        self.encoder, self.down = nn.ModuleList(), nn.ModuleList()
        c = cfg.ch[0]
        for i, c_level in enumerate(cfg.ch):
            if i > 0:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            self.encoder.append(nn.ModuleList(
                block(c if j == 0 else c_level, c_level) for j in range(cfg.n_res)))
            c = c_level

        self.mid = nn.ModuleList([block(c, c), block(c, c)])
        self.attn = SelfAttention(c) if cfg.attn_lowest else None

        # Decoder mirrors the encoder from the lowest level up, consuming one skip per block.
        self.decoder, self.up = nn.ModuleList(), nn.ModuleList()
        for i, c_level in enumerate(reversed(cfg.ch)):
            if i > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(c, c, 3, padding=1)))
            self.decoder.append(nn.ModuleList(
                block((c if j == 0 else c_level) + c_level, c_level) for j in range(cfg.n_res)))
            c = c_level

        self.out = nn.Sequential(_group_norm(c), nn.SiLU(), _zero(nn.Conv2d(c, cfg.C, 3, padding=1)))
        print(f"UNet: {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x: Tensor, c_noise: Tensor, context: Tensor, actions: torch.LongTensor,
                ctx_level: torch.LongTensor) -> Tensor:
        """x (B,C,H,W), c_noise (B,), context (B,K*C,H,W), actions (B,K), ctx_level (B,K) -> (B,C,H,W)."""
        emb = self.emb(c_noise, actions, ctx_level)
        h = self.conv_in(torch.cat([x, context], dim=1))
        skips = []
        for i, blocks in enumerate(self.encoder):
            if i > 0:
                h = self.down[i - 1](h)
            for blk in blocks:
                h = blk(h, emb)
                skips.append(h)
        h = self.mid[0](h, emb)
        if self.attn is not None:
            h = self.attn(h)
        h = self.mid[1](h, emb)
        for i, blocks in enumerate(self.decoder):
            if i > 0:
                h = self.up[i - 1](h)
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], dim=1), emb)
        return self.out(h)
