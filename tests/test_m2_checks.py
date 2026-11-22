"""Milestone 2 checks: the initial-loss formula, the shard cross-check, and the test-set config."""

import importlib.util
from pathlib import Path

import torch

from tests.fixtures import make_fake_shard
from tests.test_collect import _load_collect
from wm.config import load_config
from wm.data.dataset import WindowDataset
from wm.data.schema import EPISODE_UID_STRIDE
from wm.diffusion.edm import EDMConfig, EDMDenoiser

REPO = Path(__file__).resolve().parents[1]


def _m2():
    spec = importlib.util.spec_from_file_location("m2_checks", REPO / "scripts/m2_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ZeroNet(torch.nn.Module):
    def forward(self, x, c_noise, context, actions, ctx_level):
        return torch.zeros_like(x)


def test_expected_initial_loss_matches_a_zero_output_net():
    m2 = _m2()
    torch.manual_seed(0)
    target = torch.where(torch.rand(512, 3, 8, 8) < 0.9, -1.0, 0.4)  # mostly "black", like the game
    den = EDMDenoiser(_ZeroNet(), EDMConfig())
    g = torch.Generator().manual_seed(0)
    per_sample_m2 = target.pow(2).flatten(1).mean(1)
    for s in (0.01, 0.3, 1.0, 5.0):
        got = m2.loss_at_sigma(den, target, target[:, None].expand(-1, 4, -1, -1, -1),
                               torch.zeros(512, 4, dtype=torch.long), s, g).mean()
        want = m2.expected_initial_loss(per_sample_m2, torch.tensor([s]), 0.5).mean()
        assert abs(got - want) / want < 0.03, (s, got, want)


def test_expected_initial_loss_is_one_when_data_matches_sigma_data():
    m2 = _m2()
    sigma = torch.logspace(-3, 1.3, 20)
    assert torch.allclose(m2.expected_initial_loss(torch.tensor([0.25]), sigma, 0.5), torch.ones(1, 20))


def test_window_matches_shard_and_catches_a_shifted_action(tmp_path):
    m2 = _m2()
    make_fake_shard(tmp_path, 3, [30, 25, 40], 8, 8)
    ds = WindowDataset([tmp_path], 4, "train", val_fraction=0.0)
    for i in (0, 10, len(ds) - 1):
        item = ds[i]
        assert m2.window_matches_shard(ds, item)
        bad = {**item, "actions": torch.roll(item["actions"], 1)}
        assert not m2.window_matches_shard(ds, bad)


def test_test_set_seeds_are_eval_range_and_disjoint_from_eval():
    test, ev = load_config(REPO / "configs/data/test.yaml"), load_config(REPO / "configs/data/eval.yaml")
    collect = _load_collect()
    for workers in (1, 2, 8):
        parts = collect.plan_parts(test, workers)  # raises if seeds leave the eval range
        assert sum(p["budget"] for p in parts) == test["total_episodes"]
    eval_hi = ev["base_seed"] + len(ev["jobs"]) * 8 * EPISODE_UID_STRIDE
    assert test["base_seed"] >= eval_hi
    keys = ("resolution", "resize_method", "max_episode_steps", "total_episodes", "jobs")
    assert {k: test[k] for k in keys} == {k: ev[k] for k in keys}


def _stats():
    spec = importlib.util.spec_from_file_location("m2_data_stats", REPO / "scripts/m2_data_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_stats_counts_and_detects_a_copied_episode(tmp_path):
    import json
    import shutil

    st = _stats()
    make_fake_shard(tmp_path / "train", 4, [30, 25, 40, 22], 8, 8)  # encode_rows: every frame is unique
    shutil.copytree(tmp_path / "train", tmp_path / "copy")  # the same shard again: every window leaks
    cfg = REPO / "configs/train/debug.yaml"
    st.main(["--config", str(cfg), "--set", f"data.root={tmp_path}", "--set", "data.dirs=[train]",
             "--set", "data.val_fraction=0.0", "--holdout", "copy", "--out", str(tmp_path / "out")])
    stats = json.loads((tmp_path / "out/stats.json").read_text())
    train = stats["splits"]["train"]
    assert (train["episodes"], train["rows"], train["windows"]) == (4, 117, 117 - 4 * 4)
    assert sum(train["actions"].values()) == 117 - 4  # one LAST_ROW_ACTION per episode is not counted
    copy = stats["leakage"]["copy"]
    assert copy["windows_identical_to_a_train_window"] == copy["windows"] == 101
    assert copy["shared_seeds"] == 4 and copy["frames_identical_to_a_train_frame"] == 117
