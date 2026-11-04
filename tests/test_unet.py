"""Tests for the denoiser U-Net (ROADMAP 4.7, CP3-C)."""

import time
from pathlib import Path

import pytest
import torch

from wm.config import load_config
from wm.data.schema import NUM_ACTIONS
from wm.diffusion.edm import EDMConfig, EDMDenoiser
from wm.models.unet import UNet, UNetConfig, unet_config

CONFIGS = Path(__file__).parents[1] / "configs" / "model"
CH = {84: [64, 128, 256], 96: [64, 128, 256, 256]}
PARAM_RANGE = {"small": (10e6, 15e6), "base": (30e6, 40e6)}
TINY = dict(C=3, K=4, H=16, ch=[32, 64], n_res=1)


def make_inputs(cfg: UNetConfig, B: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return dict(
        x=torch.randn(B, cfg.C, cfg.H, cfg.H, generator=g),
        c_noise=torch.randn(B, generator=g),
        context=torch.rand(B, cfg.K * cfg.C, cfg.H, cfg.H, generator=g) * 2 - 1,
        actions=torch.randint(0, NUM_ACTIONS + 1, (B, cfg.K), generator=g),
        ctx_level=torch.randint(0, cfg.n_levels, (B, cfg.K), generator=g),
    )


def perturb_zero_init(net: UNet) -> None:
    """Give every all-zero parameter small random values; 4.7's zero-inits otherwise block all
    conditioning (zero AdaGN Linears) and all gradients upstream of the zero output conv."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in net.parameters():
            if not p.any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


@pytest.mark.parametrize("C", [1, 3])
@pytest.mark.parametrize("K", [1, 4, 8])
@pytest.mark.parametrize("H", [84, 96])
def test_output_shape(H, K, C):
    cfg = UNetConfig(C=C, K=K, H=H, ch=CH[H])
    net = UNet(cfg)
    with torch.no_grad():
        out = net(**make_inputs(cfg))
    assert out.shape == (2, C, H, H)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("attn", [False, True])
def test_output_is_exactly_zero_at_init(attn):
    cfg = UNetConfig(**TINY, attn_lowest=attn)
    with torch.no_grad():
        out = UNet(cfg)(**make_inputs(cfg))
    assert torch.equal(out, torch.zeros_like(out))


def _changed(name: str, value: torch.Tensor, cfg: UNetConfig, k: int) -> torch.Tensor:
    """`value` with sample 0 changed (at frame k for actions and ctx_level)."""
    value = value.clone()
    if name == "actions":
        value[0, k] = (value[0, k] + 1) % (NUM_ACTIONS + 1)
    elif name == "ctx_level":
        value[0, k] = (value[0, k] + 1) % cfg.n_levels
    elif name == "context":
        value[0, k * cfg.C:(k + 1) * cfg.C] += 0.5
    else:
        value[0] += 0.5
    return value


@pytest.mark.parametrize("name, k", [("x", 0), ("c_noise", 0)] + [
    (name, k) for name in ("context", "actions", "ctx_level") for k in (0, 3)])  # oldest and newest frame
def test_each_input_changes_output(name, k):
    cfg = UNetConfig(**TINY)
    net = UNet(cfg)
    perturb_zero_init(net)
    inputs = make_inputs(cfg)
    with torch.no_grad():
        base = net(**inputs)
        out = net(**{**inputs, name: _changed(name, inputs[name], cfg, k)})
    assert (out[0] - base[0]).abs().max() > 1e-4
    torch.testing.assert_close(out[1], base[1], rtol=0, atol=1e-6)  # samples don't mix


@pytest.mark.parametrize("attn", [False, True])
def test_every_parameter_gets_a_gradient(attn):
    cfg = UNetConfig(**TINY, attn_lowest=attn)
    net = UNet(cfg)
    inputs = make_inputs(cfg)
    target = torch.randn(2, cfg.C, cfg.H, cfg.H)

    (net(**inputs) - target).square().mean().backward()
    assert all(p.grad is not None for p in net.parameters())

    perturb_zero_init(net)
    net.zero_grad(set_to_none=True)
    (net(**inputs) - target).square().mean().backward()
    unused = [n for n, p in net.named_parameters() if p.grad is None or not p.grad.any()]
    assert unused == []


@pytest.mark.parametrize("H, ch", [(84, [64, 128, 256, 256]), (90, [64, 128, 256]), (16, [32] * 6)])
def test_indivisible_H_raises(H, ch):
    with pytest.raises(ValueError, match=rf"H={H}.*{2 ** (len(ch) - 1)}"):
        UNet(UNetConfig(C=3, K=4, H=H, ch=ch))


def test_dropout_is_used_in_train_mode_only():
    cfg = UNetConfig(**TINY, dropout=0.5)
    net = UNet(cfg)
    perturb_zero_init(net)
    inputs = make_inputs(cfg)
    torch.manual_seed(0)
    assert not torch.equal(net(**inputs), net(**inputs))
    net.eval()
    assert torch.equal(net(**inputs), net(**inputs))


def test_state_dict_round_trip_reproduces_output():
    cfg = UNetConfig(**TINY)
    torch.manual_seed(0)
    a = UNet(cfg)
    perturb_zero_init(a)
    torch.manual_seed(1)
    b = UNet(cfg)
    b.load_state_dict(a.state_dict())
    inputs = make_inputs(cfg)
    with torch.no_grad():
        assert torch.equal(a(**inputs), b(**inputs))


def test_prints_parameter_count(capsys):
    net = UNet(UNetConfig(**TINY))
    n = sum(p.numel() for p in net.parameters())
    assert f"{n:,}" in capsys.readouterr().out


@pytest.mark.parametrize("H", [84, 96])
@pytest.mark.parametrize("name", ["small", "base"])
def test_model_configs(name, H):
    model = load_config(CONFIGS / f"{name}.yaml")
    cfg = unet_config(model, C=3, K=4, H=H)
    if name == "small":
        assert (cfg.ch, cfg.n_res) == (CH[H], 2)
    # 96 adds a 12×12 level at the lowest width, as 4.7 does for small.
    assert model["ch"][96] == model["ch"][84] + model["ch"][84][-1:]
    net = UNet(cfg)
    n = sum(p.numel() for p in net.parameters())
    if H == 84:  # the 4.7 parameter targets; at 96 the extra level adds ~60%
        lo, hi = PARAM_RANGE[name]
        assert lo <= n <= hi, f"{name}@{H}: {n:,} parameters"
    with torch.no_grad():
        assert net(**make_inputs(cfg, B=1)).shape == (1, 3, H, H)


def test_unet_config_resolution_and_overrides():
    model = load_config(CONFIGS / "small.yaml", ["ch=[32,64]"])
    assert unet_config(model, C=1, K=8, H=16).ch == [32, 64]
    with pytest.raises(KeyError, match="64"):
        unet_config(load_config(CONFIGS / "small.yaml"), C=3, K=4, H=64)


def test_trains_inside_edm_denoiser():
    cfg = UNetConfig(**TINY)
    edm = EDMDenoiser(UNet(cfg), EDMConfig())
    inputs = make_inputs(cfg)
    context = inputs["context"].view(2, cfg.K, cfg.C, cfg.H, cfg.H)  # 5-D, as WindowDataset gives it
    sigma = torch.full((2,), 0.7)
    with torch.no_grad():
        d = edm.denoise(inputs["x"], sigma, context, inputs["actions"], inputs["ctx_level"])
    torch.testing.assert_close(d, inputs["x"] * 0.25 / (0.49 + 0.25))  # D = c_skip·x at init
    loss = edm.loss(inputs["x"].clamp(-1, 1), context, inputs["actions"], inputs["ctx_level"])
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.gpu
def test_forward_time_batch64():
    cfg = unet_config(load_config(CONFIGS / "small.yaml"), C=3, K=4, H=84)
    net = UNet(cfg).cuda().eval()
    perturb_zero_init(net)
    inputs = {k: v.cuda() for k, v in make_inputs(cfg, B=64).items()}
    for dtype in (torch.float32, torch.bfloat16):
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            for _ in range(3):
                net(**inputs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                out = net(**inputs)
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1e3
        print(f"\nUNet small@84 forward, batch 64, {dtype}: {ms:.1f} ms ({torch.cuda.get_device_name()})")
        assert torch.isfinite(out).all()
        assert ms < 1000
