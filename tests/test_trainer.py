"""CP3-B trainer tests: overfitting, exact resume, action dropout, checkpoints, dry run, signals."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.fixtures import decode_frame, make_fake_shard
from wm.data.reader import ShardReader
from wm.data.schema import NULL_ACTION
from wm.train.trainer import (CKPT_KEYS, EpochBatchSampler, Trainer, drop_actions, find_auto_resume,
                              load_run_config, run)

REPO = Path(__file__).resolve().parents[1]
DEBUG_YAML = REPO / "configs/train/debug.yaml"
H, W, K = 6, 8, 4  # non-square, so a swapped H/W shows up
LENGTHS = [12, 3, 20, 9, 15, 2, 11]  # 40 windows at K=4 -> 10 batches of 4 (train split ~ all)


def overrides(tmp_path: Path, **extra) -> list[str]:
    """--set overrides that point debug.yaml at a fake shard, the tiny net, and CPU-friendly sizes."""
    base = {
        "data.root": str(tmp_path / "data"), "data.dirs": "[fake]", "data.K": K,
        "data.batch_size": 4, "data.num_workers": 0, "data.val_fraction": 0.2,
        "model.arch": "tiny", "train.runs_dir": str(tmp_path / "runs"), "train.steps": 20,
        "train.lr": 3e-3, "train.warmup_steps": 5, "train.log_every": 5, "train.val_every": 10,
        "train.val_windows": 8, "train.sample_every": 10, "train.n_samples": 2,
        "train.save_every": 10,
    }
    base.update(extra)
    return [f"{k}={v}" for k, v in base.items()]


@pytest.fixture
def fake_data(tmp_path: Path) -> Path:
    return make_fake_shard(tmp_path / "data/fake", len(LENGTHS), LENGTHS, H, W, encode_rows=True)


def cfg_for(tmp_path: Path, **extra):
    return load_run_config(DEBUG_YAML, overrides(tmp_path, **extra))


def test_debug_yaml_defaults():
    cfg = load_run_config(DEBUG_YAML)
    assert (cfg.name, cfg.data.K, cfg.data.batch_size, cfg.train.steps) == ("debug", 4, 64, 50_000)
    assert cfg.model.arch == "unet" and cfg.model.ch == [64, 128, 256]
    assert cfg.data.dirs == ["debug"] and not cfg.data.in_memory and cfg.data.num_workers >= 4
    assert (cfg.train.p_action_drop, cfg.train.aug_max, cfg.train.ctx_noise_mode) == (0.1, 0.0, "shared")
    assert (cfg.train.lr, cfg.train.weight_decay, cfg.train.ema_decay) == (1e-4, 1e-2, 0.999)
    assert cfg.train.keep_last == 3 and cfg.sampler.num_steps == 3 and cfg.sampler.method == "euler"


def test_action_drop_rate_and_whole_sequences():
    torch.manual_seed(0)
    actions = torch.randint(0, 6, (20_000, K))
    dropped = drop_actions(actions, 0.1)
    is_null = dropped == NULL_ACTION
    row_dropped = is_null.all(1)
    assert torch.equal(is_null.any(1), row_dropped)  # all K actions or none
    assert torch.equal(dropped[~row_dropped], actions[~row_dropped])
    assert abs(row_dropped.float().mean().item() - 0.1) < 0.03
    assert torch.equal(drop_actions(actions, 0.0), actions)


def test_epoch_batch_sampler_order_and_skip():
    s = EpochBatchSampler(n=10, batch_size=3, seed=7)
    assert len(s) == 3
    s.set_position(0, 0)
    e0 = list(s)
    assert len(e0) == 3 and all(len(b) == 3 for b in e0)
    assert len({i for b in e0 for i in b}) == 9  # drop_last, no repeats
    s.set_position(0, 0)
    assert list(s) == e0  # deterministic per epoch
    s.set_position(1, 0)
    e1 = list(s)
    assert e1 != e0
    s.set_position(1, 2)
    assert list(s) == e1[2:]  # skipping resumes the same permutation


def test_train_batches_aligned_and_conditioning(tmp_path, fake_data):
    """Trainer batches keep the dataset alignment, and the net sees ctx_level zeros and actions."""
    trainer = Trainer(cfg_for(tmp_path, **{"train.p_action_drop": 0.0}), run_dir=None)
    reader = ShardReader(fake_data)
    actions_all = reader.read("actions", slice(None))
    batch = next(iter(trainer.train_loader))
    for b in range(batch["target"].shape[0]):
        i = int(batch["row"][b])
        to_u8 = lambda x: ((x.permute(1, 2, 0) + 1) * 127.5).round().byte().numpy()  # noqa: E731
        assert decode_frame(to_u8(batch["target"][b]))[0] == i
        assert [decode_frame(to_u8(c))[0] for c in batch["context"][b]] == list(range(i - K, i))
        assert batch["actions"][b].tolist() == actions_all[i - K:i].tolist()

    seen = {}
    hook = trainer.net.register_forward_pre_hook(lambda m, args: seen.update(args=args))
    trainer.train_step([batch])
    hook.remove()
    _, _, context, actions, ctx_level = seen["args"]
    assert context.shape == (4, K * 3, H, W)
    assert ctx_level.dtype == torch.long and ctx_level.shape == (4, K) and not ctx_level.any()
    assert torch.equal(actions, batch["actions"])  # p_action_drop = 0


def test_overfit_one_batch(tmp_path, fake_data):
    cfg = cfg_for(tmp_path, **{"train.p_action_drop": 0.0, "train.warmup_steps": 0,
                               "data.batch_size": 8, "model.tiny_hidden": 32})
    trainer = Trainer(cfg, run_dir=None)
    batch = next(iter(trainer.train_loader))
    before = trainer.eval_loss(batch)
    for _ in range(300):
        trainer.train_step([batch])
    after = trainer.eval_loss(batch)
    assert after * 5 <= before, (before, after)


def test_ema_tracks_weights_during_warmup(tmp_path, fake_data):
    trainer = Trainer(cfg_for(tmp_path, **{"train.warmup_steps": 3, "train.ema_decay": 0.5}), None)
    batches = iter(trainer.train_loader)
    for _ in range(3):
        trainer.train_step([next(batches)])
    for p, e in zip(trainer.net.parameters(), trainer.ema_net.parameters()):
        assert torch.equal(p, e)
    trainer.train_step([next(batches)])
    assert any(not torch.equal(p, e) for p, e in zip(trainer.net.parameters(), trainer.ema_net.parameters()))


def test_checkpoint_keys_run_dir_and_pruning(tmp_path, fake_data):
    cfg = cfg_for(tmp_path, **{"train.steps": 7, "train.save_every": 1, "train.sample_every": 7})
    assert run(cfg) == 0
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("_debug")
    for name in ("config.yaml", "git_commit.txt", "tb", "checkpoints", "samples"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "samples/step_0000007.png").exists()
    ckpts = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
    assert ckpts == ["ckpt_0000005.pt", "ckpt_0000006.pt", "ckpt_0000007.pt", "ckpt_last.pt"]
    ckpt = torch.load(run_dir / "checkpoints/ckpt_last.pt", weights_only=False)
    assert set(ckpt) == set(CKPT_KEYS) == {"model", "ema", "optimizer", "scheduler", "step", "config",
                                           "rng_states"}
    assert ckpt["step"] == 7 and ckpt["config"]["name"] == "debug"
    assert {"python", "numpy", "torch"} <= set(ckpt["rng_states"])


def _final_states(runs_dir: Path) -> dict:
    (run_dir,) = runs_dir.iterdir()
    return torch.load(run_dir / "checkpoints/ckpt_last.pt", weights_only=False)


# workers=2 is slow only because each persistent worker with an open HDF5 file takes ~5 s to
# shut down (torch's join timeout, then terminate); see progress/cp3-B.md.
@pytest.mark.parametrize("workers", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_resume_equals_straight_run(tmp_path, fake_data, workers):
    """10 steps, save, resume to 20 equals 20 straight steps (crosses epochs; resume is mid-epoch)."""
    common = {"data.num_workers": workers, "data.batch_size": 3, "train.sample_every": 1000,
              "train.save_every": 1000}
    per_epoch = len(Trainer(cfg_for(tmp_path, **common), None).sampler)
    assert 10 % per_epoch and 20 > per_epoch  # resume lands mid-epoch, and the run crosses epochs
    assert run(cfg_for(tmp_path, **common, **{"train.runs_dir": str(tmp_path / "straight")})) == 0
    split = {**common, "train.runs_dir": str(tmp_path / "split")}
    assert run(cfg_for(tmp_path, **split, **{"train.steps": 10})) == 0
    assert run(cfg_for(tmp_path, **split), resume="auto") == 0
    a, b = _final_states(tmp_path / "straight"), _final_states(tmp_path / "split")
    assert a["step"] == b["step"] == 20
    for key in ("model", "ema"):
        for name, v in a[key].items():
            torch.testing.assert_close(b[key][name], v, rtol=1e-6, atol=1e-7, msg=f"{key}.{name}")


def test_find_auto_resume(tmp_path):
    runs = tmp_path / "runs"
    for name, has_ckpt in [("20261101-0900_debug", True), ("20261102-0900_debug", True),
                           ("20261103-0900_debug", False), ("20261104-0900_debug_aug", True),
                           ("20261104-0901_x_debug", True)]:
        (runs / name / "checkpoints").mkdir(parents=True)
        if has_ckpt:
            (runs / name / "checkpoints/ckpt_last.pt").write_bytes(b"")
    assert find_auto_resume(runs, "debug") == runs / "20261102-0900_debug"
    assert find_auto_resume(runs, "debug_aug") == runs / "20261104-0900_debug_aug"
    assert find_auto_resume(runs, "other") is None
    assert find_auto_resume(tmp_path / "missing", "debug") is None


def _train_cmd(tmp_path: Path, *args: str, **extra) -> list[str]:
    cmd = [sys.executable, str(REPO / "scripts/train.py"), "--config", str(DEBUG_YAML), *args]
    for item in overrides(tmp_path, **extra):
        cmd += ["--set", item]
    return cmd


def test_dry_run(tmp_path, fake_data):
    out = subprocess.run(_train_cmd(tmp_path, "--dry-run"), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "dry run: 5 steps" in out.stdout
    assert not (tmp_path / "runs").exists()  # a dry run writes nothing


def test_sigusr1_saves_and_auto_resume_continues(tmp_path, fake_data):
    cmd = _train_cmd(tmp_path, "--resume", "auto", **{"train.steps": 1_000_000, "train.log_every": 1,
                                                      "train.save_every": 1_000_000,
                                                      "train.val_every": 1_000_000,
                                                      "train.sample_every": 1_000_000})
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 60
    for line in proc.stdout:
        if line.startswith("step ") and int(line.split()[1]) >= 3:
            break
        assert time.monotonic() < deadline, "no training progress"
    proc.send_signal(signal.SIGUSR1)
    rest = proc.stdout.read()
    assert proc.wait(timeout=60) == 0, rest
    (run_dir,) = (tmp_path / "runs").iterdir()
    step = torch.load(run_dir / "checkpoints/ckpt_last.pt", weights_only=False)["step"]
    assert step >= 3 and "SIGUSR1" in rest

    out = subprocess.run(_train_cmd(tmp_path, "--resume", "auto", **{"train.steps": step + 3}),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert f"resumed {run_dir} at step {step}" in out.stdout
    assert torch.load(run_dir / "checkpoints/ckpt_last.pt", weights_only=False)["step"] == step + 3
    assert len(list((tmp_path / "runs").iterdir())) == 1  # continued in place
