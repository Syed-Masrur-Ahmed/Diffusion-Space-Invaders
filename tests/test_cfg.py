"""Action CFG and the adagn+planes U-Net input (ROADMAP CP5-C)."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from scripts.action_sensitivity import (
    band_mae,
    main as sensitivity_main,
    player_band,
    propose_cfg_weight,
    with_last_action,
)
from tests.fixtures import TinyDenoiserNet, make_fake_shard
from wm.config import load_config
from wm.data.dataset import WindowDataset
from wm.data.schema import ACTION_MEANINGS, NULL_ACTION, NUM_ACTIONS
from wm.diffusion.edm import EDMConfig, EDMDenoiser
from wm.diffusion.sampler import SamplerConfig, sample
from wm.eval.rollout import rollout
from wm.inference import FRAME_SIZE, WorldModel
from wm.models.unet import UNet, UNetConfig, unet_config
from wm.train.trainer import DataSection, ModelSection, RunConfig, TrainSection

REPO = Path(__file__).resolve().parents[1]
K, C = 4, 3
H = W = FRAME_SIZE
TINY = dict(C=3, K=4, H=16, ch=[32, 64], n_res=1)
LEFT, RIGHT = ACTION_MEANINGS.index("LEFT"), ACTION_MEANINGS.index("RIGHT")


class _RecordingDenoiser:
    """Returns zeros and keeps every action batch `_guided_denoise` passes in."""

    def __init__(self) -> None:
        self.actions: list[torch.Tensor] = []

    def denoise(self, x, sigma, context, actions, ctx_level):
        self.actions.append(actions.detach().cpu().clone())
        return torch.zeros_like(x)


class _ActionDenoiser:
    """D = 0.1 * last action, so w=0 and w=1 have a closed form at one Euler step."""

    def denoise(self, x, sigma, context, actions, ctx_level):
        return (0.1 * actions[:, -1].to(x.dtype)).view(-1, 1, 1, 1).expand_as(x)


def _cond(B=2, seed=0, k=K):
    g = torch.Generator().manual_seed(seed)
    return dict(
        context=torch.rand(B, k, C, 8, 8, generator=g) * 2 - 1,
        actions=torch.randint(0, NUM_ACTIONS, (B, k), generator=g),
        ctx_level=torch.zeros(B, k, dtype=torch.long),
    )


def _perturb(net: UNet) -> None:
    """Fill all-zero parameters so AdaGN and the output conv actually depend on their inputs."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in net.parameters():
            if not p.any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def _inputs(cfg: UNetConfig, B: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return dict(
        x=torch.randn(B, cfg.C, cfg.H, cfg.H, generator=g),
        c_noise=torch.randn(B, generator=g),
        context=torch.rand(B, cfg.K * cfg.C, cfg.H, cfg.H, generator=g) * 2 - 1,
        actions=torch.randint(0, NUM_ACTIONS + 1, (B, cfg.K), generator=g),
        ctx_level=torch.randint(0, cfg.n_levels, (B, cfg.K), generator=g),
    )


def _run_config(arch: str = "tiny", **model) -> RunConfig:
    return RunConfig(
        name="test", seed=0, data=DataSection(K=K),
        model=ModelSection(arch=arch, tiny_hidden=8, **model),
        edm=EDMConfig(), sampler=SamplerConfig(num_steps=2), train=TrainSection(),
    )


def _save_ckpt(path: Path, cfg: RunConfig, net: torch.nn.Module, *, action_input: str | None = None) -> Path:
    saved = dataclasses.asdict(cfg)
    if action_input is not None:
        saved["model"]["action_input"] = action_input
    torch.save({"model": net.state_dict(), "ema": net.state_dict(), "optimizer": {}, "scheduler": {},
                "step": 1, "config": saved, "rng_states": {}}, path)
    return path


def _random_tiny(seed: int) -> TinyDenoiserNet:
    torch.manual_seed(seed)
    net = TinyDenoiserNet(C=C, K=K, hidden=8)
    torch.nn.init.normal_(net.out.weight, std=0.1)
    return net


# ---- sampler and WorldModel: w=1 is no guidance, w=0 is unconditional -------------------

def test_w1_matches_the_plain_conditional_path_and_does_not_double_the_batch():
    torch.manual_seed(0)
    net = TinyDenoiserNet(C=C, K=K, hidden=8)
    torch.nn.init.normal_(net.out.weight, std=0.3)
    edm = EDMDenoiser(net, EDMConfig()).eval()
    cond = _cond()
    cfg = SamplerConfig(num_steps=2)
    guided = sample(edm, **cond, cfg=cfg, cfg_scale=1.0, generator=torch.Generator().manual_seed(3))
    plain = sample(edm, **cond, cfg=cfg, generator=torch.Generator().manual_seed(3))
    assert torch.equal(guided, plain)

    seen = _RecordingDenoiser()
    sample(seen, **cond, cfg=SamplerConfig(num_steps=1), cfg_scale=1.0)
    assert len(seen.actions) == 1 and seen.actions[0].shape[0] == cond["actions"].shape[0]
    assert torch.equal(seen.actions[0], cond["actions"])


def test_w0_matches_all_null_actions_and_still_sends_them_in_the_doubled_batch():
    cond = _cond(B=2)
    cfg = SamplerConfig(num_steps=1)
    guided = sample(_ActionDenoiser(), **cond, cfg=cfg, cfg_scale=0.0,
                     generator=torch.Generator().manual_seed(5))
    uncond = sample(_ActionDenoiser(), **{**cond, "actions": torch.full_like(cond["actions"], NULL_ACTION)},
                    cfg=cfg, cfg_scale=1.0, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(guided, uncond)
    torch.testing.assert_close(guided[:, 0, 0, 0], torch.full((2,), 0.1 * NULL_ACTION).clamp(-1, 1))

    seen = _RecordingDenoiser()
    sample(seen, **cond, cfg=cfg, cfg_scale=0.0)
    actions = seen.actions[0]
    B = cond["actions"].shape[0]
    assert actions.shape[0] == 2 * B
    assert torch.equal(actions[:B], cond["actions"])
    assert torch.equal(actions[B:], torch.full_like(cond["actions"], NULL_ACTION))


def test_worldmodel_and_rollout_forward_cfg_scale(tmp_path: Path):
    path = _save_ckpt(tmp_path / "ckpt.pt", _run_config(), _random_tiny(0))
    wm = WorldModel.from_checkpoint(path, device="cpu", channels_last=False)
    g = torch.Generator().manual_seed(0)
    context = torch.rand(2, K, C, H, W, generator=g) * 2 - 1
    actions = torch.randint(0, NUM_ACTIONS, (2, K), generator=g)
    plain = wm.predict_next(context, actions, num_steps=1, generator=torch.Generator().manual_seed(1))
    one = wm.predict_next(context, actions, num_steps=1, cfg_scale=1.0, generator=torch.Generator().manual_seed(1))
    assert torch.equal(plain, one)

    zero = wm.predict_next(context, actions, num_steps=1, cfg_scale=0.0, generator=torch.Generator().manual_seed(2))
    null = wm.predict_next(context, torch.full_like(actions, NULL_ACTION), num_steps=1, cfg_scale=1.0,
                           generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(zero, null)

    sizes: list[int] = []
    wm.denoiser.net.register_forward_pre_hook(lambda m, args: sizes.append(args[0].shape[0]))
    wm.predict_next(context, actions, num_steps=1, cfg_scale=1.0)
    wm.predict_next(context, actions, num_steps=1, cfg_scale=2.0)
    assert sizes == [2, 4]

    rollout_actions = torch.randint(0, NUM_ACTIONS, (2, K + 1))
    frames0 = context
    guided = rollout(wm, frames0, rollout_actions, 2, num_steps=1, cfg_scale=0.0,
                     generator=torch.Generator().manual_seed(4))
    uncond = rollout(wm, frames0, torch.full_like(rollout_actions, NULL_ACTION), 2, num_steps=1, cfg_scale=1.0,
                     generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(guided, uncond)
    same = rollout(wm, frames0, rollout_actions, 2, num_steps=1, cfg_scale=1.0,
                   generator=torch.Generator().manual_seed(4))
    default = rollout(wm, frames0, rollout_actions, 2, num_steps=1, generator=torch.Generator().manual_seed(4))
    assert torch.equal(same, default)


def test_null_action_never_appears_in_dataset_windows(tmp_path: Path):
    lengths = [5, 1, 40, 3, 4, 12, 2, 60, 9, 8]
    make_fake_shard(tmp_path, len(lengths), lengths, 6, 5)
    seen = 0
    for k in (1, 4, 8):
        for split in ("train", "val"):
            ds = WindowDataset([tmp_path], k, split, val_fraction=0.3)
            try:
                if len(ds) == 0:
                    continue
                acts = torch.stack([ds[i]["actions"] for i in range(len(ds))])
            finally:
                ds.close()
            assert int(acts.min()) >= 0 and int(acts.max()) < NULL_ACTION
            assert int((acts == NULL_ACTION).sum()) == 0
            seen += acts.numel()
    assert seen > 0


# ---- adagn+planes ----------------------------------------------------------------------

def test_action_input_defaults_to_adagn_and_rejects_unknown_values():
    cfg = unet_config({"ch": [32, 64], "n_res": 1}, C=3, K=4, H=16)
    assert cfg.action_input == "adagn"
    planes = unet_config({"ch": [32, 64], "n_res": 1, "action_input": "adagn+planes"}, C=3, K=4, H=16)
    assert planes.action_input == "adagn+planes"
    with pytest.raises(ValueError, match="action_input"):
        UNet(UNetConfig(**TINY, action_input="cross_attn"))


@pytest.mark.parametrize("C_,K_,H_", [(3, 4, 16), (3, 4, 84), (1, 1, 16)])
def test_adagn_planes_output_shape_and_conv_in_width(C_, K_, H_):
    ch = [8, 16, 16] if H_ == 84 else [32, 64]
    cfg = UNetConfig(C=C_, K=K_, H=H_, ch=ch, n_res=1, emb_dim=32, action_input="adagn+planes")
    net = UNet(cfg)
    assert net.conv_in.in_channels == C_ * (K_ + 1) + NUM_ACTIONS + 1
    with torch.no_grad():
        out = net(**_inputs(cfg))
    assert out.shape == (2, C_, H_, H_)
    plain = UNet(UNetConfig(C=C_, K=K_, H=H_, ch=ch, n_res=1, emb_dim=32))
    assert plain.conv_in.in_channels == C_ * (K_ + 1)
    extra = (NUM_ACTIONS + 1) * ch[0] * 3 * 3  # conv_in kernel only; bias is per output channel
    assert sum(p.numel() for p in net.parameters()) - sum(p.numel() for p in plain.parameters()) == extra


def test_adagn_planes_is_zero_at_init_and_the_planes_are_the_latest_action():
    cfg = UNetConfig(**TINY, action_input="adagn+planes")
    net = UNet(cfg)
    inputs = _inputs(cfg)
    inputs["actions"] = torch.tensor([[0, 1, 2, LEFT], [4, 5, NULL_ACTION, RIGHT]])
    captured: dict[str, torch.Tensor] = {}
    net.conv_in.register_forward_pre_hook(lambda m, args: captured.setdefault("x", args[0].detach()))
    with torch.no_grad():
        out = net(**inputs)
    assert torch.equal(out, torch.zeros_like(out))
    planes = captured["x"][:, -(NUM_ACTIONS + 1):]
    expected = F.one_hot(inputs["actions"][:, -1], NUM_ACTIONS + 1).to(planes.dtype)
    assert torch.equal(planes[:, :, 0, 0], expected)
    assert torch.equal(planes, planes[:, :, :1, :1].expand_as(planes))

    older = inputs["actions"].clone()
    older[0, 0] = (older[0, 0] + 1) % NUM_ACTIONS
    captured.clear()
    with torch.no_grad():
        net(**{**inputs, "actions": older})
    assert torch.equal(captured["x"][:, -(NUM_ACTIONS + 1):], planes)
    latest = inputs["actions"].clone()
    latest[0, -1] = RIGHT
    captured.clear()
    with torch.no_grad():
        net(**{**inputs, "actions": latest})
    assert not torch.equal(captured["x"][:, -(NUM_ACTIONS + 1):, 0, 0], expected)


def test_adagn_planes_every_parameter_gets_a_gradient_and_actions_change_the_output():
    cfg = UNetConfig(**TINY, action_input="adagn+planes")
    net = UNet(cfg)
    inputs = _inputs(cfg)
    target = torch.randn(2, cfg.C, cfg.H, cfg.H)
    (net(**inputs) - target).square().mean().backward()
    assert all(p.grad is not None for p in net.parameters())

    _perturb(net)
    net.zero_grad(set_to_none=True)
    inputs["actions"] = with_last_action(inputs["actions"], RIGHT)
    (net(**inputs) - target).square().mean().backward()
    unused = [n for n, p in net.named_parameters() if p.grad is None or not p.grad.any()]
    assert unused == []
    extra = net.conv_in.weight.grad[:, -(NUM_ACTIONS + 1):]
    assert extra is not None and extra.any()

    with torch.no_grad():
        base = net(**inputs)
        changed_latest = net(**{**inputs, "actions": with_last_action(inputs["actions"], LEFT)})
        older = inputs["actions"].clone()
        older[:, 0] = (older[:, 0] + 1) % (NUM_ACTIONS + 1)
        changed_older = net(**{**inputs, "actions": older})
    assert not torch.allclose(base, changed_latest)
    assert not torch.allclose(base, changed_older)


def test_planes_checkpoint_loads_and_left_differs_from_right(tmp_path: Path):
    unet_cfg = UNetConfig(C=C, K=K, H=H, ch=[8, 16, 16], n_res=1, emb_dim=32, action_input="adagn+planes")
    torch.manual_seed(0)
    net = UNet(unet_cfg)
    _perturb(net)
    path = _save_ckpt(tmp_path / "ckpt.pt", _run_config("unet", ch=[8, 16, 16], n_res=1, emb_dim=32), net,
                      action_input="adagn+planes")
    wm = WorldModel.from_checkpoint(path, device="cpu", channels_last=False)
    assert isinstance(wm.denoiser.net, UNet)
    assert wm.denoiser.net.cfg.action_input == "adagn+planes"
    assert wm.denoiser.net.conv_in.in_channels == C * (K + 1) + NUM_ACTIONS + 1
    context = torch.zeros(2, K, C, H, W)
    left = wm.predict_next(context, with_last_action(torch.zeros(2, K, dtype=torch.long), LEFT), num_steps=1,
                           generator=torch.Generator().manual_seed(0))
    right = wm.predict_next(context, with_last_action(torch.zeros(2, K, dtype=torch.long), RIGHT), num_steps=1,
                            generator=torch.Generator().manual_seed(0))
    assert left.shape == (2, C, H, W)
    assert not torch.allclose(left, right)
    # from_checkpoint defaults to channels-last. The planes are concatenated after that conversion.
    default = WorldModel.from_checkpoint(path, device="cpu")
    kwargs = dict(num_steps=1, generator=torch.Generator().manual_seed(0))
    d_left = default.predict_next(context, with_last_action(torch.zeros(2, K, dtype=torch.long), LEFT), **kwargs)
    d_right = default.predict_next(context, with_last_action(torch.zeros(2, K, dtype=torch.long), RIGHT),
                                   num_steps=1, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(d_left, d_right)
    torch.testing.assert_close(d_left, left, atol=1e-4, rtol=1e-4)

    adagn = UNet(UNetConfig(C=C, K=K, H=H, ch=[8, 16, 16], n_res=1, emb_dim=32))
    adagn_path = _save_ckpt(tmp_path / "adagn.pt", _run_config("unet", ch=[8, 16, 16], n_res=1, emb_dim=32), adagn,
                            action_input="adagn")
    loaded = WorldModel.from_checkpoint(adagn_path, device="cpu", channels_last=False)
    assert loaded.denoiser.net.conv_in.in_channels == C * (K + 1)

    bad = _save_ckpt(tmp_path / "bad.pt", _run_config("unet", ch=[8, 16, 16], n_res=1, emb_dim=32), adagn,
                     action_input="cross_attn")
    with pytest.raises(ValueError, match="action_input"):
        WorldModel.from_checkpoint(bad, device="cpu")


# ---- action sensitivity ----------------------------------------------------------------

def test_player_band_is_the_scaled_raw_sprite_rows():
    assert player_band(210) == (185, 195)
    assert player_band(84) == (74, 78)  # 185/210 and 195/210 of 84
    assert player_band(16) == (14, 15)
    top, bottom = player_band(96)
    assert 0 <= top < bottom <= 96


def test_band_mae_is_in_eight_bit_units_and_ignores_rows_outside_the_band():
    left = torch.zeros(2, 3, 8, 8)
    right = torch.zeros(2, 3, 8, 8)
    right[:, :, 6:] = 2 / 127.5
    mae = band_mae(left, right, (6, 8))
    torch.testing.assert_close(mae, torch.full((2,), 2.0))
    outside = right.clone()
    outside[:, :, :6] = 1
    torch.testing.assert_close(band_mae(left, outside, (6, 8)), mae)


def test_with_last_action_replaces_only_the_action_that_produces_the_frame():
    actions = torch.arange(8).view(2, 4)
    out = with_last_action(actions, LEFT)
    assert torch.equal(out[:, :-1], actions[:, :-1])
    assert torch.equal(out[:, -1], torch.full((2,), LEFT))
    assert torch.equal(actions[:, -1], torch.tensor([3, 7]))  # the input is not modified


def test_propose_cfg_weight_picks_the_smallest_w_that_reaches_most_of_the_gain():
    rows = [
        {"w": 1.0, "sensitivity": 1.0},
        {"w": 1.5, "sensitivity": 1.85},
        {"w": 2.0, "sensitivity": 2.0},
        {"w": 3.0, "sensitivity": 1.9},
    ]
    got = propose_cfg_weight(rows, gain=0.8)
    assert got["w"] == 1.5 and got["baseline"] == 1.0 and got["best"] == 2.0
    assert got["target"] == pytest.approx(1.8)
    assert got["reason"] is None

    flat = propose_cfg_weight([{"w": 1.0, "sensitivity": 0.2}, {"w": 2.0, "sensitivity": 0.2}])
    assert flat["w"] == 1.0 and flat["reason"]
    assert propose_cfg_weight([])["w"] is None


def test_sensitivity_config_lists_the_roadmap_sweep():
    cfg = load_config(REPO / "configs/eval/action_sensitivity.yaml")
    assert cfg["weights"] == [1, 1.5, 2, 3]
    assert cfg["ctx_level"] == 0 and cfg["gain"] == 0.8
    assert cfg["band_raw"] == [185, 195] and cfg["data"]["dataset_dirs"] == ["data/debug"]


def test_action_sensitivity_script_on_a_fake_shard(tmp_path: Path):
    data = tmp_path / "data"
    make_fake_shard(data, 2, [8, 8], H, W)
    ckpt = _save_ckpt(tmp_path / "ckpt.pt", _run_config(), _random_tiny(0))
    cfg = {
        "model": str(ckpt), "out_dir": str(tmp_path / "out"), "n": 4, "weights": [1, 2],
        "num_steps": 1, "ctx_level": 0, "seed": 0, "gain": 0.8, "ignore_below": 0.0, "grid_n": 2,
        "include_copy_last": True, "use_ema": True, "channels_last": False,
        "data": {"dataset_dirs": [str(data)], "K": K, "grayscale": False, "val_fraction": 1.0, "batch_size": 2},
        "band_raw": [185, 195], "raw_height": 210,
    }
    path = tmp_path / "sens.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = sensitivity_main(["--config", str(path), "--set", "n=2"])
    result = json.loads((out / "sensitivity.json").read_text())
    assert result["n"] == 2 and result["band"] == [74, 78]
    by_key = {(r["model"], r["w"]): r for r in result["rows"]}
    assert by_key[("copy_last", None)]["sensitivity"] == 0.0
    assert by_key[(str(ckpt), 1.0)]["sensitivity"] > 0.0
    assert by_key[(str(ckpt), 2.0)]["sensitivity"] > 0.0
    assert result["proposal"]["ignores_actions"] is False
    assert (out / "sensitivity.csv").is_file()
    assert (out / "grid_w1.png").is_file()
    text = (out / "sensitivity.csv").read_text()
    assert text.startswith("model,w,sensitivity,std,n")
