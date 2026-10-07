import pytest
import torch

from fixtures import TinyDenoiserNet
from wm.data.schema import NULL_ACTION


def _inputs(B, C, K, H, W, n_levels=10):
    g = torch.Generator().manual_seed(0)
    return dict(
        x=torch.randn(B, C, H, W, generator=g),
        c_noise=torch.randn(B, generator=g),
        context=torch.randn(B, K * C, H, W, generator=g),
        actions=torch.randint(0, NULL_ACTION + 1, (B, K), generator=g),
        ctx_level=torch.randint(0, n_levels, (B, K), generator=g),
    )


@pytest.mark.parametrize("K", [1, 4])
@pytest.mark.parametrize("C", [1, 3])
def test_tiny_net_shape_and_zero_init(K, C):
    net = TinyDenoiserNet(C=C, K=K)
    out = net(**_inputs(2, C, K, 12, 10))
    assert out.shape == (2, C, 12, 10)
    assert torch.equal(out, torch.zeros_like(out))


def test_tiny_net_uses_conditioning_after_init():
    net = TinyDenoiserNet(C=3, K=4)
    torch.nn.init.normal_(net.out.weight)
    inputs = _inputs(2, 3, 4, 8, 8)
    base = net(**inputs)
    perturbed = {
        "x": inputs["x"] + 1,
        "c_noise": inputs["c_noise"] + 1,
        "context": inputs["context"] + 1,
        "actions": (inputs["actions"] + 1) % (NULL_ACTION + 1),
        "ctx_level": (inputs["ctx_level"] + 1) % 10,
    }
    for name, value in perturbed.items():
        assert not torch.allclose(net(**{**inputs, name: value}), base), name
