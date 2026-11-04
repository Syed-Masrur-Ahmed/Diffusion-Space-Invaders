import math
from pathlib import Path

import pytest
import torch
from torch import nn

from fixtures import TinyDenoiserNet
from wm.config import from_dict, load_config
from wm.data.schema import NULL_ACTION
from wm.diffusion.edm import EDMConfig, EDMDenoiser, edm_coefficients, loss_by_sigma

CONFIG = Path(__file__).resolve().parents[1] / "configs/train/edm_default.yaml"


def _batch(B=4, C=3, K=4, H=8, W=8, dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    return dict(
        target=torch.rand(B, C, H, W, generator=g, dtype=dtype) * 2 - 1,
        context=torch.rand(B, K, C, H, W, generator=g, dtype=dtype) * 2 - 1,
        actions=torch.randint(0, NULL_ACTION + 1, (B, K), generator=g),
        ctx_level=torch.zeros(B, K, dtype=torch.long),
    )


class OracleNet(nn.Module):
    """Returns F_target for a known clean target, recovering x and sigma from its inputs."""

    def __init__(self, y, sigma_data=0.5):
        super().__init__()
        self.y, self.sd = y, sigma_data

    def forward(self, x_in, c_noise, context, actions, ctx_level):
        sigma = torch.exp(4 * c_noise).view(-1, 1, 1, 1)
        x = x_in * torch.sqrt(sigma**2 + self.sd**2)
        c_skip = self.sd**2 / (sigma**2 + self.sd**2)
        c_out = sigma * self.sd / torch.sqrt(sigma**2 + self.sd**2)
        return (self.y - c_skip * x) / c_out


class RecordingNet(nn.Module):
    """Records its inputs and returns a constant, so D's wiring can be checked by hand."""

    def __init__(self, value=0.25):
        super().__init__()
        self.value = value

    def forward(self, x_in, c_noise, context, actions, ctx_level):
        self.inputs = dict(x_in=x_in, c_noise=c_noise, context=context, actions=actions,
                           ctx_level=ctx_level)
        return torch.full_like(x_in, self.value)


def test_coefficient_limits():
    c_skip, c_out, c_in, c_noise = edm_coefficients(torch.tensor([1e-8]), 0.5)
    assert c_skip.item() == pytest.approx(1.0, abs=1e-6)
    assert c_out.item() == pytest.approx(0.0, abs=1e-6)
    assert c_in.item() == pytest.approx(2.0, rel=1e-6)
    assert c_noise.item() == pytest.approx(math.log(1e-8) / 4, rel=1e-6)


def test_coefficient_values():
    sigma = torch.tensor([0.5, 2.0])
    c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, 0.5)
    torch.testing.assert_close(c_skip, torch.tensor([0.5, 0.25 / 4.25]))
    torch.testing.assert_close(c_out, torch.tensor([0.25 / math.sqrt(0.5), 1.0 / math.sqrt(4.25)]))
    torch.testing.assert_close(c_in, torch.tensor([1 / math.sqrt(0.5), 1 / math.sqrt(4.25)]))
    torch.testing.assert_close(c_noise, torch.log(sigma) / 4)


def test_oracle_loss_is_zero():
    torch.manual_seed(0)
    b = _batch(B=64, dtype=torch.float64)
    edm = EDMDenoiser(OracleNet(b["target"]), EDMConfig())
    assert edm.loss(**b).item() == pytest.approx(0.0, abs=1e-12)


def test_denoise_at_init_is_c_skip_x():
    torch.manual_seed(0)
    b = _batch()
    edm = EDMDenoiser(TinyDenoiserNet(C=3, K=4), EDMConfig())
    x = torch.randn_like(b["target"])
    sigma = torch.tensor([0.01, 0.5, 2.0, 10.0])
    D = edm.denoise(x, sigma, b["context"], b["actions"], b["ctx_level"])
    c_skip = 0.25 / (sigma**2 + 0.25)
    torch.testing.assert_close(D, c_skip.view(-1, 1, 1, 1) * x)


def test_denoise_wiring():
    b = _batch()
    net = RecordingNet(value=0.25)
    edm = EDMDenoiser(net, EDMConfig())
    x = torch.randn_like(b["target"])
    sigma = torch.tensor([0.01, 0.5, 2.0, 10.0])
    D = edm.denoise(x, sigma, b["context"], b["actions"], b["ctx_level"])
    c_skip, c_out, c_in, c_noise = (c.view(-1, 1, 1, 1) for c in edm_coefficients(sigma, 0.5))
    torch.testing.assert_close(net.inputs["x_in"], c_in * x)
    torch.testing.assert_close(net.inputs["c_noise"], torch.log(sigma) / 4)
    # Context goes in unscaled, flattened to (B, K*C, H, W), oldest frame first.
    assert torch.equal(net.inputs["context"], b["context"].flatten(1, 2))
    assert torch.equal(net.inputs["actions"], b["actions"])
    assert torch.equal(net.inputs["ctx_level"], b["ctx_level"])
    torch.testing.assert_close(D, c_skip * x + c_out * 0.25)


def test_denoise_accepts_flat_context():
    b = _batch()
    edm = EDMDenoiser(RecordingNet(), EDMConfig())
    x, sigma = torch.randn_like(b["target"]), torch.full((4,), 0.7)
    D5 = edm.denoise(x, sigma, b["context"], b["actions"], b["ctx_level"])
    D4 = edm.denoise(x, sigma, b["context"].flatten(1, 2), b["actions"], b["ctx_level"])
    assert torch.equal(D5, D4)


def test_sigma_sampling_distribution():
    torch.manual_seed(0)
    cfg = EDMConfig()
    sigma = EDMDenoiser(TinyDenoiserNet(), cfg).sample_sigma(100_000, torch.device("cpu"))
    assert sigma.shape == (100_000,)
    assert sigma.min() >= cfg.sigma_min and sigma.max() <= cfg.sigma_max
    log_sigma = torch.log(sigma)
    assert log_sigma.mean().item() == pytest.approx(cfg.p_mean, abs=0.03)
    assert log_sigma.std().item() == pytest.approx(cfg.p_std, abs=0.03)


def test_sigma_is_clamped():
    cfg = EDMConfig(p_mean=0.0, p_std=10.0)
    torch.manual_seed(0)
    sigma = EDMDenoiser(TinyDenoiserNet(), cfg).sample_sigma(10_000, torch.device("cpu"))
    assert (sigma == cfg.sigma_min).any() and (sigma == cfg.sigma_max).any()


def test_loss_matches_per_sample_terms_and_backprops():
    b = _batch()
    net = TinyDenoiserNet(C=3, K=4)
    nn.init.normal_(net.out.weight, std=0.1)
    edm = EDMDenoiser(net, EDMConfig())
    torch.manual_seed(1)
    loss = edm.loss(**b)
    torch.manual_seed(1)
    per_sample, sigma = edm.loss_terms(**b)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert per_sample.shape == sigma.shape == (4,)
    torch.testing.assert_close(loss, per_sample.mean())
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())


def test_loss_equals_edm_weighted_loss():
    """mean((F - F_target)^2) equals lambda(sigma) * mean((D - y)^2), lambda = (s^2+sd^2)/(s*sd)^2."""
    b = _batch()
    net = TinyDenoiserNet(C=3, K=4)
    nn.init.normal_(net.out.weight, std=0.1)
    edm = EDMDenoiser(net, EDMConfig())
    torch.manual_seed(2)
    per_sample, sigma = edm.loss_terms(**b)
    torch.manual_seed(2)
    edm.sample_sigma(4, torch.device("cpu"))  # replay the same draws: sigma, then noise
    noise = torch.randn_like(b["target"])
    x = b["target"] + sigma.view(-1, 1, 1, 1) * noise
    D = edm.denoise(x, sigma, b["context"], b["actions"], b["ctx_level"])
    lam = (sigma**2 + 0.25) / (sigma * 0.5) ** 2
    torch.testing.assert_close(per_sample, lam * ((D - b["target"]) ** 2).mean(dim=(1, 2, 3)))


def test_loss_by_sigma_buckets():
    sigma = torch.tensor([0.003, 0.004, 0.1, 15.0, 19.0, 20.0])
    losses = torch.tensor([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
    out = loss_by_sigma(losses, sigma, n_buckets=3, sigma_min=2e-3, sigma_max=20.0)
    # Log-spaced edges: 2e-3, ~4.3e-2, ~0.93, 20. Empty buckets are left out.
    assert list(out.values()) == [pytest.approx(2.0), pytest.approx(5.0), pytest.approx(4.0)]
    assert list(out) == ["0.002-0.0431", "0.0431-0.928", "0.928-20"]


def test_loss_by_sigma_skips_empty_buckets():
    out = loss_by_sigma(torch.tensor([1.0, 2.0]), torch.tensor([5.0, 6.0]), n_buckets=4)
    assert len(out) == 1 and next(iter(out.values())) == pytest.approx(1.5)


def test_default_config_matches_dataclass():
    cfg = load_config(CONFIG)
    assert from_dict(EDMConfig, cfg["edm"]) == EDMConfig()
