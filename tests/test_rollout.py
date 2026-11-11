"""Rollout tests (ROADMAP 4.11, CP4-A): alignment fixtures on encoded fake shards, plots, videos, the script."""

import csv
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.fixtures import TinyDenoiserNet, decode_frame, make_fake_shard
from wm.data.reader import ShardReader, list_shards
from wm.diffusion.edm import EDMConfig
from wm.diffusion.sampler import SamplerConfig
from wm.eval.baselines import CopyLastFrame
from wm.eval.rollout import horizon_metrics, load_trajectories, rollout
from wm.inference import FRAME_SIZE
from wm.train.trainer import DataSection, ModelSection, RunConfig, TrainSection

REPO = Path(__file__).resolve().parents[1]
K, T, H, W = 4, 12, 8, 8
LENGTHS = [30, 10, 25, 40]  # rows; episode 1 is too short for K + T = 16 rows


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_rollouts", REPO / "scripts" / "eval_rollouts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1,1] (..., C, H, W) -> uint8 (..., H, W, C)."""
    return ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).movedim(-3, -1).numpy()


def to_float(u8: np.ndarray) -> torch.Tensor:
    """uint8 (..., H, W, C) -> [-1,1] (..., C, H, W)."""
    return torch.from_numpy(u8).movedim(-1, -3).float() / 127.5 - 1


def decode(x: torch.Tensor) -> int:
    """Row index encoded in one [-1,1] (C,H,W) frame."""
    return decode_frame(to_uint8(x))[0]


def encoded(row: int, like: torch.Tensor) -> torch.Tensor:
    """Frame (C,H,W) in [-1,1] encoding `row` with R and G as `encode_frame` does (B channel 0)."""
    rgb = torch.tensor([row % 256, (row // 256) % 256, 0], dtype=torch.float32) / 127.5 - 1
    return rgb[:, None, None].expand_as(like).clone()


class FakeModel(CopyLastFrame):
    """WorldModelProtocol stub whose prediction is computed from the last context frame and actions."""

    def __init__(self, fn, C: int = 3) -> None:
        super().__init__(K, C, H, W)
        self.fn, self.calls = fn, []

    def predict_next(self, context, actions, num_steps=None, cfg_scale=None, ctx_level=0, generator=None):
        assert context.shape[1:] == (K, self.C, H, W) and actions.shape == (len(context), K)
        self.calls.append({"num_steps": num_steps, "cfg_scale": cfg_scale, "ctx_level": ctx_level})
        return torch.stack([self.fn(c[-1], a) for c, a in zip(context, actions)])


@pytest.fixture
def shard_dir(tmp_path: Path) -> Path:
    make_fake_shard(tmp_path / "eval", len(LENGTHS), LENGTHS, H, W, encode_rows=True)
    return tmp_path / "eval"


@pytest.fixture
def traj(shard_dir: Path):
    return load_trajectories([shard_dir], n=3, length=K + T)


def test_load_trajectories_alignment(shard_dir: Path, traj):
    reader = ShardReader(list_shards(shard_dir)[0])
    starts = reader.episodes["start"]
    assert traj.frames.shape == (3, K + T, H, W, 3) and traj.frames.dtype == np.uint8
    assert traj.actions.shape == (3, K + T - 1) and traj.actions.dtype == np.int64
    assert [e for _, e in traj.episodes] == [0, 2, 3]  # episode 1 is too short
    for b, (_, e) in enumerate(traj.episodes):
        for j in range(K + T):
            assert decode_frame(traj.frames[b, j]) == (starts[e] + j, e)
        np.testing.assert_array_equal(traj.actions[b], reader.read("actions", slice(starts[e], starts[e] + K + T - 1)))
    assert (traj.actions >= 0).all()


def test_load_trajectories_start_offset(shard_dir: Path):
    reader = ShardReader(list_shards(shard_dir)[0])
    starts = reader.episodes["start"]
    traj = load_trajectories([shard_dir], n=2, length=K + T, start=10)
    assert [e for _, e in traj.episodes] == [0, 3]  # episode 2 has 25 rows < 10 + 16
    for b, (_, e) in enumerate(traj.episodes):
        assert [decode_frame(f)[0] for f in traj.frames[b]] == list(range(starts[e] + 10, starts[e] + 10 + K + T))
        np.testing.assert_array_equal(traj.actions[b], reader.read("actions", slice(starts[e] + 10, starts[e] + 10 + K + T - 1)))


def test_load_trajectories_spreads_and_rejects(shard_dir: Path):
    assert [e for _, e in load_trajectories([shard_dir], n=2, length=K + T).episodes] == [0, 3]
    with pytest.raises(ValueError, match="only 3 episodes"):
        load_trajectories([shard_dir], n=4, length=K + T)


def test_copy_last_repeats_the_last_context_frame(traj):
    frames = to_float(traj.frames)
    gen = rollout(CopyLastFrame(K, 3, H, W), frames[:, :K], torch.from_numpy(traj.actions), T)
    assert gen.shape == (3, T, 3, H, W)
    assert torch.equal(gen, frames[:, K - 1:K].expand_as(gen))


def test_index_model_generates_frame_j(traj):
    """A model that outputs 'last frame's index + 1' must produce g_j encoding row start + j."""
    frames = to_float(traj.frames)
    model = FakeModel(lambda last, a: encoded(decode(last) + 1, last))
    gen = rollout(model, frames[:, :K], torch.from_numpy(traj.actions), T)
    for b in range(len(gen)):
        start = decode(frames[b, 0])
        assert [decode(gen[b, t]) for t in range(T)] == [start + K + t for t in range(T)]


def test_action_model_frame_j_encodes_a_j_minus_1(traj):
    """A model that writes actions[:, -1] into the pixels shows that g_j was produced by a_{j-1}."""
    frames = to_float(traj.frames)
    model = FakeModel(lambda last, a: encoded(int(a[-1]), last))
    gen = rollout(model, frames[:, :K], torch.from_numpy(traj.actions), T)
    for b in range(len(gen)):
        assert [decode(gen[b, t]) for t in range(T)] == [traj.actions[b, K + t - 1] for t in range(T)]


def test_oracle_model_gets_perfect_metrics(traj):
    """An oracle that returns the true f_j (looked up from the encoded index) scores perfectly at every horizon."""
    frames = to_float(traj.frames)
    by_row = {decode(f): f for f in frames.flatten(0, 1)}
    model = FakeModel(lambda last, a: by_row[decode(last) + 1])
    gen = rollout(model, frames[:, :K], torch.from_numpy(traj.actions), T)
    assert torch.equal(gen, frames[:, K:])
    values = horizon_metrics(gen, frames[:, K:], ["psnr", "ssim"])
    assert values["psnr"].shape == (3, T) and values["ssim"].shape == (3, T)
    assert torch.all(values["psnr"] == 100) and torch.allclose(values["ssim"], torch.ones(3, T), atol=1e-3)


def test_horizon_metrics_are_per_frame():
    target = torch.zeros(2, 3, 3, H, W)
    pred = target.clone()
    pred[1, 2] += 0.2  # [-1,1] units: 0.1 in [0,1], so 20 dB
    psnr = horizon_metrics(pred, target, ["psnr"], batch_size=4)["psnr"]
    expected = torch.full((2, 3), 100.0)
    expected[1, 2] = 20.0
    assert torch.allclose(psnr, expected, atol=1e-4)


def test_rollout_passes_sampler_kwargs_and_checks_shapes(traj):
    frames = to_float(traj.frames)
    model = FakeModel(lambda last, a: last)
    actions = torch.from_numpy(traj.actions)[:, :K + 2]
    rollout(model, frames[:, :K], actions, 3, num_steps=5, cfg_scale=2.0, ctx_level=2)
    assert model.calls == [{"num_steps": 5, "cfg_scale": 2.0, "ctx_level": 2}] * 3
    with pytest.raises(ValueError, match="actions"):
        rollout(model, frames[:, :K], actions[:, :K + 1], 3)
    with pytest.raises(ValueError, match="frames0"):
        rollout(model, frames[:, :K - 1], actions, 3)


def test_rollout_uses_generated_frames_as_context(traj):
    """After the first step the context holds only generated frames: here every step adds 0.1."""
    frames = to_float(traj.frames)
    gen = rollout(FakeModel(lambda last, a: last + 0.1), frames[:, :K], torch.from_numpy(traj.actions)[:, :K + 4], 5)
    for t in range(5):
        assert torch.allclose(gen[:, t], frames[:, K - 1] + 0.1 * (t + 1), atol=1e-5)


def test_plot_and_video(tmp_path: Path):
    import imageio.v3 as iio

    from wm.eval.plots import plot_horizon, save_rollout_video

    h = list(range(1, 11))
    curves = {name: {m: {"h": h, "mean": [1.0 + i] * 10, "ci95": [0.1] * 10} for m in ["psnr", "ssim", "lpips"]}
              for i, name in enumerate(["copy_last", "model"])}
    plot_horizon(tmp_path / "horizon.png", curves)
    assert iio.imread(tmp_path / "horizon.png").ndim == 3

    real = torch.rand(6, 3, H, W) * 2 - 1
    save_rollout_video(tmp_path / "v.mp4", real, real.flip(0), fps=15, scale=4)
    video = _read_video(tmp_path / "v.mp4")
    assert video.shape == (6, H * 4, 3 * W * 4 + 2 * 2, 3)


def _read_video(path: Path) -> np.ndarray:
    import imageio.v2 as iio2

    with iio2.get_reader(path) as reader:
        return np.stack(list(reader.iter_data()))  # len(reader) is inf for ffmpeg


def _tiny_ckpt(path: Path) -> Path:
    torch.manual_seed(0)
    net = TinyDenoiserNet(C=3, K=K, hidden=8)
    torch.nn.init.normal_(net.out.weight, std=0.1)
    cfg = RunConfig(name="test", seed=0, data=DataSection(K=K), model=ModelSection(arch="tiny", tiny_hidden=8),
                    edm=EDMConfig(), sampler=SamplerConfig(num_steps=2), train=TrainSection())
    torch.save({"model": net.state_dict(), "ema": net.state_dict(), "optimizer": {}, "scheduler": {}, "step": 1,
                "config": dataclasses.asdict(cfg), "rng_states": {}}, path)
    return path


def test_script_end_to_end(tmp_path: Path):
    """copy_last plus a tiny checkpoint swept over two context levels: CSV, JSON, figure and videos."""
    data = tmp_path / "eval"
    make_fake_shard(data, 3, [24, 22, 26], FRAME_SIZE, FRAME_SIZE, encode_rows=False)
    (tmp_path / "run" / "checkpoints").mkdir(parents=True)
    ckpt = _tiny_ckpt(tmp_path / "run" / "checkpoints" / "ckpt_last.pt")
    out = _load_script().main([
        "--config", str(REPO / "configs" / "eval" / "rollouts.yaml"), "--models", "copy_last", str(ckpt),
        "--T", "10", "--n", "3", "--set", f"dataset_dirs=[{data}]", "--set", "start=0", "--set", f"out_dir={tmp_path / 'out'}",
        "--set", "metrics=[psnr,ssim]", "--set", "ctx_levels=[0,2]", "--set", "batch_size=2",
        "--set", "n_videos=1", "--set", "report_horizons=[1,2,5,10]"])

    result = json.loads((out / "metrics.json").read_text())
    assert list(result["models"]) == ["copy_last", "run_ckpt_last@ctx0", "run_ckpt_last@ctx2"]
    copy_last = result["models"]["copy_last"]
    assert copy_last["n"] == 3 and len(copy_last["metrics"]["psnr"]["mean"]) == 10
    assert copy_last["metrics"]["psnr"]["h"] == list(range(1, 11))
    assert set(copy_last["report"]["psnr"]) == {"1", "2", "5", "10"}
    assert result["models"]["run_ckpt_last@ctx2"]["ctx_level"] == 2
    assert result["models"]["run_ckpt_last@ctx0"]["metrics"] != result["models"]["run_ckpt_last@ctx2"]["metrics"]

    with open(out / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 10 and set(rows[0]) >= {"model", "h", "psnr_mean", "psnr_ci95", "ssim_mean"}
    assert (out / "horizon.png").exists()
    videos = sorted(p.name for p in (out / "videos").glob("*.mp4"))
    assert videos == ["copy_last_traj00.mp4", "run_ckpt_last@ctx0_traj00.mp4", "run_ckpt_last@ctx2_traj00.mp4"] and all(p.stat().st_size > 0 for p in (out / "videos").glob("*.mp4"))


def test_script_is_deterministic(tmp_path: Path):
    data = tmp_path / "eval"
    make_fake_shard(data, 2, [20, 20], FRAME_SIZE, FRAME_SIZE, encode_rows=False)
    (tmp_path / "run" / "checkpoints").mkdir(parents=True)
    ckpt = _tiny_ckpt(tmp_path / "run" / "checkpoints" / "ckpt_last.pt")
    results = []
    for i in range(2):
        out = _load_script().main([
            "--config", str(REPO / "configs" / "eval" / "rollouts.yaml"), "--models", str(ckpt), "--T", "6",
            "--n", "2", "--set", f"dataset_dirs=[{data}]", "--set", "start=0", "--set", f"out_dir={tmp_path / f'out{i}'}",
            "--set", "metrics=[psnr]", "--set", "n_videos=0"])
        results.append(json.loads((out / "metrics.json").read_text())["models"]["run_ckpt_last@ctx0"]["metrics"])
    assert results[0] == results[1]
