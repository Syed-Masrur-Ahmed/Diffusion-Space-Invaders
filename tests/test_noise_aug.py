"""CP4-B context noise augmentation tests (ROADMAP 4.8): levels, noise scale, modes, trainer wiring, resume."""

import dataclasses
from pathlib import Path

import pytest
import torch

import wm.train.trainer as trainer_mod
from tests.fixtures import make_fake_shard
from tests.test_trainer import DEBUG_YAML, H, LENGTHS, W, K, _final_states, cfg_for
from wm.train.trainer import Trainer, context_noise, load_run_config, run

REPO = Path(__file__).resolve().parents[1]
N_LEVELS, AUG_MAX = 10, 0.7


@pytest.fixture
def fake_data(tmp_path: Path) -> Path:
    return make_fake_shard(tmp_path / "data/fake", len(LENGTHS), LENGTHS, H, W, encode_rows=True)


def _context(B: int = 2000, side: int = 8) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(B, K, 3, side, side) * 2 - 1


def test_aug_off_is_identity_and_draws_nothing():
    ctx = _context(16)
    state = torch.get_rng_state()
    out, level = context_noise(ctx, N_LEVELS, 0.0, "shared")
    assert torch.equal(torch.get_rng_state(), state)  # no random numbers drawn
    assert torch.equal(out, ctx)
    assert level.dtype == torch.long and level.shape == (16, K) and not level.any()


@pytest.mark.parametrize("mode", ["shared", "per_frame"])
def test_noise_std_matches_sigma_aug_per_level(mode):
    ctx = _context()
    out, level = context_noise(ctx, N_LEVELS, AUG_MAX, mode)
    diff = (out - ctx).flatten(2)  # (B, K, C*H*W)
    for lv in range(N_LEVELS):
        d = diff[level == lv]
        assert d.numel() > 10_000, (lv, d.numel())
        if lv == 0:
            assert not d.any()  # level 0 adds no noise, even with aug_max > 0
        else:
            target = lv / (N_LEVELS - 1) * AUG_MAX
            assert abs(d.std().item() / target - 1) < 0.05, (lv, d.std().item(), target)
            assert abs(d.mean().item()) < 0.05 * target


def test_noisy_context_is_not_clamped():
    ctx = torch.ones(500, K, 3, 4, 4)
    out, level = context_noise(ctx, N_LEVELS, AUG_MAX, "shared")
    assert out.max() > 1.5 and (level > 0).any()


def test_shared_mode_one_level_per_sample():
    _, level = context_noise(_context(2000, 2), N_LEVELS, AUG_MAX, "shared")
    assert torch.equal(level, level[:, :1].expand_as(level))


def test_per_frame_mode_levels_vary_within_samples():
    _, level = context_noise(_context(2000, 2), N_LEVELS, AUG_MAX, "per_frame")
    varies = (level != level[:, :1]).any(1).float().mean().item()
    assert varies > 0.95  # P(all K equal) = 10^-3


@pytest.mark.parametrize("mode", ["shared", "per_frame"])
def test_levels_uniform(mode):
    _, level = context_noise(_context(20_000, 1), N_LEVELS, AUG_MAX, mode)
    assert level.min() == 0 and level.max() == N_LEVELS - 1
    counts = torch.bincount(level.flatten(), minlength=N_LEVELS).float()
    expected = level.numel() / N_LEVELS
    assert ((counts / expected - 1).abs() < 0.1).all(), counts  # >= 5 binomial std per bin


def test_invalid_mode_and_aug_max_rejected(tmp_path):
    with pytest.raises(ValueError, match="ctx_noise_mode"):
        cfg_for(tmp_path, **{"train.ctx_noise_mode": "both"})
    with pytest.raises(ValueError, match="aug_max"):
        cfg_for(tmp_path, **{"train.aug_max": -0.1})


def _spy_net(trainer: Trainer) -> tuple[list, object]:
    seen = []
    hook = trainer.net.register_forward_pre_hook(lambda m, args: seen.append(args))
    return seen, hook


@pytest.mark.parametrize("mode", ["shared", "per_frame"])
def test_net_receives_the_levels_that_noised_the_context(tmp_path, fake_data, monkeypatch, mode):
    """The ctx_level the net sees is the one context_noise returned, and so is the noisy context."""
    cfg = cfg_for(tmp_path, **{"train.aug_max": AUG_MAX, "train.ctx_noise_mode": mode, "data.batch_size": 8,
                               "train.p_action_drop": 0.0})
    trainer = Trainer(cfg, run_dir=None)
    calls = []

    def recording(context, n_levels, aug_max, m):
        out = context_noise(context, n_levels, aug_max, m)
        calls.append((context, n_levels, aug_max, m, *out))
        return out

    monkeypatch.setattr(trainer_mod, "context_noise", recording)
    seen, hook = _spy_net(trainer)
    batch = next(iter(trainer.train_loader))
    torch.manual_seed(3)  # some levels > 0 in the 8 samples
    trainer.train_step([batch])
    hook.remove()
    ((clean, n_levels, aug_max, m, noisy, level),) = calls
    assert (n_levels, aug_max, m) == (N_LEVELS, AUG_MAX, mode)
    assert torch.equal(clean, batch["context"])  # noise is applied to the loader's clean context
    ((_, _, context, _, ctx_level),) = seen
    assert torch.equal(ctx_level, level) and ctx_level.dtype == torch.long and ctx_level.shape == (8, K)
    assert torch.equal(context, noisy.flatten(1, 2))
    assert ctx_level.any()
    per_frame_diff = (context - clean.flatten(1, 2)).view(8, K, -1).abs().amax(2)
    assert torch.equal(per_frame_diff == 0, ctx_level == 0)


def test_val_and_sample_grid_use_clean_context(tmp_path, fake_data):
    trainer = Trainer(cfg_for(tmp_path, **{"train.aug_max": AUG_MAX}), run_dir=None)
    seen, hook = _spy_net(trainer)
    trainer.val_loss()
    hook.remove()
    seen_ema = []
    hook = trainer.ema_net.register_forward_pre_hook(lambda m, args: seen_ema.append(args))
    trainer.sample_grid_predictions()
    hook.remove()
    assert seen and seen_ema
    for args in seen + seen_ema:
        assert not args[4].any()
    grid_ctx = trainer.grid_batch["context"].flatten(1, 2)
    assert all(torch.equal(args[2], grid_ctx) for args in seen_ema)


def test_level_histogram_logged(tmp_path, fake_data):
    cfg = cfg_for(tmp_path, **{"train.aug_max": AUG_MAX, "train.steps": 5, "train.log_every": 5})
    assert run(cfg) == 0
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    (run_dir,) = (tmp_path / "runs").iterdir()
    acc = EventAccumulator(str(run_dir / "tb"), size_guidance={"histograms": 0})
    acc.Reload()
    assert "train/ctx_level" in acc.Tags()["histograms"]
    assert any(t.startswith("loss_by_ctx_level/") for t in acc.Tags()["scalars"])


@pytest.mark.parametrize("mode", ["shared", "per_frame"])
def test_resume_equals_straight_run_with_aug(tmp_path, fake_data, mode):
    """aug_max = 0.7: 10 steps, save, resume to 20 equals 20 straight steps; and the noise changes training."""
    common = {"data.batch_size": 3, "train.sample_every": 1000, "train.save_every": 1000,
              "train.aug_max": AUG_MAX, "train.ctx_noise_mode": mode}
    assert run(cfg_for(tmp_path, **common, **{"train.runs_dir": str(tmp_path / "straight")})) == 0
    split = {**common, "train.runs_dir": str(tmp_path / "split")}
    assert run(cfg_for(tmp_path, **split, **{"train.steps": 10})) == 0
    assert run(cfg_for(tmp_path, **split), resume="auto") == 0
    a, b = _final_states(tmp_path / "straight"), _final_states(tmp_path / "split")
    assert a["step"] == b["step"] == 20
    for key in ("model", "ema"):
        for name, v in a[key].items():
            torch.testing.assert_close(b[key][name], v, rtol=1e-6, atol=1e-7, msg=f"{key}.{name}")
    off = {**common, "train.aug_max": 0.0, "train.runs_dir": str(tmp_path / "off")}
    assert run(cfg_for(tmp_path, **off)) == 0
    c = _final_states(tmp_path / "off")
    assert any(not torch.equal(c["model"][n], v) for n, v in a["model"].items())


def _flat(cfg) -> dict:
    def walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from walk(v, f"{prefix}{k}.")
            else:
                yield f"{prefix}{k}", v
    return dict(walk(dataclasses.asdict(cfg)))


@pytest.mark.parametrize("name, aug_max", [("debug_noaug", 0.0), ("debug_aug", 0.7)])
def test_run_configs_differ_from_debug_only_in_name_and_aug(name, aug_max):
    base = _flat(load_run_config(DEBUG_YAML))
    cfg = _flat(load_run_config(REPO / f"configs/train/{name}.yaml"))
    assert set(cfg) == set(base)
    assert {k for k in base if base[k] != cfg[k]} <= {"name", "train.aug_max"}
    assert (cfg["name"], cfg["train.aug_max"], cfg["train.ctx_noise_mode"]) == (name, aug_max, "shared")
