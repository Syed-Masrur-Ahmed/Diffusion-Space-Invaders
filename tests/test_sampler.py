from pathlib import Path

import pytest
import torch
from torch import nn

from fixtures import TinyDenoiserNet
from wm.config import from_dict, load_config
from wm.data.schema import NULL_ACTION
from wm.diffusion.edm import EDMConfig, EDMDenoiser
from wm.diffusion.sampler import SamplerConfig, karras_sigmas, sample

CONFIG = Path(__file__).resolve().parents[1] / "configs/train/edm_default.yaml"


class GaussianOracle:
    """Optimal denoiser for y ~ N(0, s^2) per pixel: D(x; sigma) = x s^2 / (sigma^2 + s^2)."""

    def __init__(self, s):
        self.s = s

    def denoise(self, x, sigma, context, actions, ctx_level):
        s2 = self.s**2
        return x * s2 / (sigma.view(-1, 1, 1, 1) ** 2 + s2)


class ActionDenoiser:
    """D = 0.1 * last action everywhere, so the CFG combination can be checked exactly."""

    def denoise(self, x, sigma, context, actions, ctx_level):
        return (0.1 * actions[:, -1].to(x.dtype)).view(-1, 1, 1, 1).expand_as(x)


class CountingNet(nn.Module):
    """Wraps a net and records the batch size of every forward call."""

    def __init__(self, net):
        super().__init__()
        self.net, self.batch_sizes = net, []

    def forward(self, x, c_noise, context, actions, ctx_level):
        self.batch_sizes.append(x.shape[0])
        return self.net(x, c_noise, context, actions, ctx_level)


def _cond(B=2, C=3, K=4, H=8, W=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        context=torch.rand(B, K, C, H, W, generator=g) * 2 - 1,
        actions=torch.randint(0, NULL_ACTION, (B, K), generator=g),
        ctx_level=torch.zeros(B, K, dtype=torch.long),
    )


def _tiny_edm(C=3, K=4, seed=0):
    torch.manual_seed(seed)
    net = TinyDenoiserNet(C=C, K=K)
    nn.init.normal_(net.out.weight, std=0.3)
    return EDMDenoiser(CountingNet(net), EDMConfig()).eval()


@pytest.mark.parametrize("N", [1, 2, 3, 10])
def test_karras_schedule(N):
    s = karras_sigmas(N, 2e-3, 5.0, 7.0)
    assert s.shape == (N + 1,)
    assert s[0].item() == pytest.approx(5.0, rel=1e-9)
    assert s[-1].item() == 0.0
    assert (s[1:] < s[:-1]).all()
    if N > 1:
        assert s[-2].item() == pytest.approx(2e-3, rel=1e-9)


def test_karras_single_step_is_sigma_max_then_zero():
    assert karras_sigmas(1, 2e-3, 5.0, 7.0).tolist() == [5.0, 0.0]


def test_karras_three_steps_values():
    s = karras_sigmas(3, 2e-3, 5.0, 7.0)
    mid = ((5.0 ** (1 / 7) + 2e-3 ** (1 / 7)) / 2) ** 7
    torch.testing.assert_close(s, torch.tensor([5.0, mid, 2e-3, 0.0], dtype=s.dtype))


def _oracle_samples(s, cfg, B=64, H=16, W=16):
    cond = _cond(B=B, C=1, H=H, W=W)
    return sample(GaussianOracle(s), **cond, cfg=cfg, generator=torch.Generator().manual_seed(0))


def test_gaussian_oracle_heun_50_steps():
    s = 0.3
    out = _oracle_samples(s, SamplerConfig(num_steps=50, method="heun"))
    assert out.std().item() == pytest.approx(s, rel=0.03)


def test_gaussian_oracle_euler_3_steps():
    """Few Euler steps shrink a Gaussian's std, so N=3 is checked against the closed form, not s.

    With this oracle each Euler step scales x by 1 + (s1 - s0) (1 - s^2 / (s0^2 + s^2)) / s0,
    starting from std sigma_0. On the default schedule that gives ~0.53 s; no s reaches within
    15% (the original ROADMAP tolerance), see progress/cp2-C.md.
    """
    s = 0.3
    cfg = SamplerConfig(num_steps=3, method="euler")
    sigmas = karras_sigmas(cfg.num_steps, cfg.sigma_min, cfg.sigma_max, cfg.rho).tolist()
    expected = sigmas[0]
    for s0, s1 in zip(sigmas[:-1], sigmas[1:]):
        expected *= 1 + (s1 - s0) * (1 - s**2 / (s0**2 + s**2)) / s0
    out = _oracle_samples(s, cfg)
    assert torch.isfinite(out).all()
    assert out.std().item() < s
    assert out.std().item() == pytest.approx(expected, rel=0.03)


@pytest.mark.parametrize("method,calls_per_sample", [("euler", 3), ("heun", 5)])
def test_cfg_scale_one_is_single_pass(method, calls_per_sample):
    edm = _tiny_edm()
    sample(edm, **_cond(B=2), cfg=SamplerConfig(num_steps=3, method=method), cfg_scale=1.0)
    # Heun skips the correction on the final step to sigma=0: 2N-1 calls.
    assert edm.net.batch_sizes == [2] * calls_per_sample


def test_cfg_doubles_batch_in_one_call():
    edm = _tiny_edm()
    sample(edm, **_cond(B=2), cfg=SamplerConfig(num_steps=3), cfg_scale=2.0)
    assert edm.net.batch_sizes == [4] * 3


def test_cfg_combination():
    cond = _cond(B=2, C=1)
    cond["actions"][:, -1] = torch.tensor([1, 3])
    out = sample(ActionDenoiser(), **cond, cfg=SamplerConfig(num_steps=1), cfg_scale=2.0)
    # One Euler step to sigma=0 returns D itself: D_u + w (D_c - D_u), with D_u = 0.1 * NULL_ACTION.
    d_u = 0.1 * NULL_ACTION
    expected = torch.tensor([d_u + 2 * (0.1 - d_u), d_u + 2 * (0.3 - d_u)]).clamp(-1, 1)
    torch.testing.assert_close(out[:, 0, 0, 0], expected)


def test_cfg_scale_zero_equals_unconditional():
    edm = _tiny_edm()
    cond = _cond(B=3)
    cfg = SamplerConfig(num_steps=3)
    guided = sample(edm, **cond, cfg=cfg, cfg_scale=0.0, generator=torch.Generator().manual_seed(5))
    uncond = sample(edm, **{**cond, "actions": torch.full_like(cond["actions"], NULL_ACTION)},
                    cfg=cfg, cfg_scale=1.0, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(guided, uncond)


def test_cfg_scale_one_equals_plain_conditional_denoiser_path():
    edm = _tiny_edm()
    cond = _cond(B=2)
    cfg = SamplerConfig(num_steps=2)
    out = sample(edm, **cond, cfg=cfg, cfg_scale=1.0, generator=torch.Generator().manual_seed(3))
    sigmas = karras_sigmas(2, cfg.sigma_min, cfg.sigma_max, cfg.rho).tolist()
    x = sigmas[0] * torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        for s0, s1 in zip(sigmas[:-1], sigmas[1:]):
            d = (x - edm.denoise(x, torch.full((2,), s0), **cond)) / s0
            x = x + (s1 - s0) * d
    torch.testing.assert_close(out, x.clamp(-1, 1))


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_seeded_sampling_is_deterministic(method):
    edm = _tiny_edm()
    cond = _cond(B=2)
    cfg = SamplerConfig(num_steps=3, method=method)
    a = sample(edm, **cond, cfg=cfg, cfg_scale=1.5, generator=torch.Generator().manual_seed(7))
    b = sample(edm, **cond, cfg=cfg, cfg_scale=1.5, generator=torch.Generator().manual_seed(7))
    c = sample(edm, **cond, cfg=cfg, cfg_scale=1.5, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("C,K", [(1, 1), (3, 4), (3, 8)])
def test_output_shape_and_range(C, K):
    edm = _tiny_edm(C=C, K=K)
    cond = _cond(B=2, C=C, K=K, H=12, W=10)
    out = sample(edm, **cond, cfg=SamplerConfig(num_steps=3))
    assert out.shape == (2, C, 12, 10)
    assert out.min() >= -1 and out.max() <= 1
    flat = {**cond, "context": cond["context"].flatten(1, 2)}
    out_flat = sample(edm, **flat, cfg=SamplerConfig(num_steps=3))
    assert out_flat.shape == (2, C, 12, 10)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        sample(_tiny_edm(), **_cond(), cfg=SamplerConfig(method="rk4"))


def test_default_config_matches_dataclass():
    cfg = load_config(CONFIG)
    assert from_dict(SamplerConfig, cfg["sampler"]) == SamplerConfig()
