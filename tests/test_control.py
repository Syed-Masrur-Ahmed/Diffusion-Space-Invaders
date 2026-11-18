"""CP5-A: pixel detectors, counterfactual metrics, inverse dynamics model, and the eval scripts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.fixtures import decode_frame, make_fake_shard
from tests.test_rollout import _tiny_ckpt
from wm.controls import ACTION_TO_MOVE_FIRE
from wm.data.reader import ShardReader
from wm.data.schema import ACTION_MEANINGS, END_TRUNCATED, EVAL_SEED_BASE, NUM_ACTIONS, SOURCE_RANDOM
from wm.data.writer import ShardWriter
from wm.env import make_env, to_model_frame
from wm.eval import counterfactual as cf
from wm.eval import detectors
from wm.eval.baselines import CopyLastFrame
from wm.models import idm

REPO = Path(__file__).resolve().parents[1]
H = W = 84
GROUND_RGB = (80, 89, 22)


def scene(player: float | None = None, laser_col: int | None = None, laser_rows: tuple[int, int] = (64, 72),
          bullet: tuple[int, int] | None = None, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Synthetic 84x84 frame: ground, a 5-px player centred on column `player`, optional laser / bullet."""
    f = np.zeros((H, W, 3), np.uint8)
    f[78:] = GROUND_RGB
    f[62:70, 10:16] = (181, 83, 40)  # a shield, which must not count as player or laser
    if player is not None:
        x = int(round(player))
        f[74:78, x - 2:x + 3] = detectors.PLAYER_RGB
    if laser_col is not None:
        f[laser_rows[0]:laser_rows[1], laser_col:laser_col + 2] = detectors.LASER_RGB
    if bullet is not None:  # an alien bullet: one grey column, 4 rows
        f[bullet[0]:bullet[0] + 4, bullet[1]] = detectors.LASER_RGB
    if noise:  # generated frames are close to, not exactly, the palette
        f = np.clip(f + np.random.default_rng(seed).normal(0, noise, f.shape), 0, 255).astype(np.uint8)
    return f


# --- Detectors ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [8, 19, 42, 75])
def test_player_x_is_the_sprite_centroid(x):
    np.testing.assert_allclose(detectors.player_x(scene(player=x)[None]), [x])


def test_player_x_tolerates_generated_frame_noise_and_ignores_other_colours():
    frames = np.stack([scene(player=x, laser_col=x, bullet=(70, 60), noise=8.0, seed=x) for x in (12, 30, 55)])
    np.testing.assert_allclose(detectors.player_x(frames), [12, 30, 55], atol=0.5)


def test_player_x_is_nan_without_a_player():
    frames = np.stack([scene(), scene(player=40)])
    frames[0, 40:44, 30:35] = detectors.PLAYER_RGB  # player colour above the bottom band
    x = detectors.player_x(frames)
    assert np.isnan(x[0]) and x[1] == 40


def test_laser_present_only_for_a_laser_above_the_player():
    frames = np.stack([
        scene(player=30, laser_col=30),                       # just fired
        scene(player=30, laser_col=30, noise=8.0),            # same, generated-looking
        scene(player=30),                                     # nothing
        scene(player=30, laser_col=60),                       # a streak in another column
        scene(player=30, laser_col=30, laser_rows=(10, 18)),  # high up: an old shot, not a new one
        scene(laser_col=30),                                  # no player to fire from
    ])
    np.testing.assert_array_equal(detectors.laser_present(frames), [True, True, False, False, False, False])


def test_calibration_recovers_a_linear_ram_map():
    ram_x = np.arange(35, 118, 3)
    frames = np.stack([scene(player=0.5 * r + 1.0) for r in ram_x] + [scene()])  # last frame: no player
    ram = np.zeros((len(frames), 128), np.uint8)
    ram[:-1, 28] = ram_x
    cal = detectors.calibrate_player_x(detectors.player_x(frames), ram)
    assert cal["n"] == len(ram_x)
    assert cal["mae_px"] < 0.5 and abs(cal["slope"] - 0.5) < 0.02


def test_frames_from_model_output_round_trip():
    f = scene(player=20, laser_col=20)
    x = torch.from_numpy(f).permute(2, 0, 1).float()[None] / 127.5 - 1
    np.testing.assert_array_equal(detectors.to_uint8_frames(x), f[None])


# --- Action mapping ------------------------------------------------------------------------


def test_action_to_move_fire_covers_every_action_consistently():
    assert sorted(ACTION_TO_MOVE_FIRE) == list(range(NUM_ACTIONS))
    assert len(set(ACTION_TO_MOVE_FIRE.values())) == NUM_ACTIONS  # one action per (move, fire)
    for a, (move, fire) in ACTION_TO_MOVE_FIRE.items():
        name = ACTION_MEANINGS[a]
        assert fire == ("FIRE" in name)
        assert move == (1 if "RIGHT" in name else 2 if "LEFT" in name else 0)


# --- Inverse dynamics model ----------------------------------------------------------------


def test_action_targets_follow_the_mapping_table():
    move, fire = idm.action_targets(torch.arange(NUM_ACTIONS))
    assert list(zip(move.tolist(), fire.tolist())) == [ACTION_TO_MOVE_FIRE[a] for a in range(NUM_ACTIONS)]


@pytest.mark.parametrize("size", [84, 96])
def test_idm_shapes_and_gradients(size):
    model = idm.IDM(C=3, H=size, W=size)
    f0, f1 = torch.rand(2, 2, 3, size, size) * 2 - 1
    move, fire = model(f0, f1)
    assert move.shape == (2, 3) and fire.shape == (2, 2)
    (move.sum() + fire.sum()).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_idm_overfits_256_fake_pairs():
    torch.manual_seed(0)
    f0, f1 = torch.rand(2, 256, 3, 12, 12) * 2 - 1
    actions = torch.randint(0, NUM_ACTIONS, (256,))  # random labels: memorized, not learned
    model = idm.IDM(C=3, H=12, W=12, ch=(8, 16), hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(100):  # full batch; reaches 100% at ~50 steps
        loss, _ = idm.loss_and_accuracy(model, f0, f1, actions)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, acc = idm.loss_and_accuracy(model, f0, f1, actions)
    assert acc["move"] == 1.0 and acc["fire"] == 1.0


def test_idm_checkpoint_round_trip(tmp_path):
    model = idm.IDM(C=3, H=16, W=16, ch=(8, 16), hidden=32)
    idm.save_idm(tmp_path / "idm.pt", model, step=7)
    loaded = idm.load_idm(tmp_path / "idm.pt")
    f0, f1 = torch.rand(2, 4, 3, 16, 16)
    for a, b in zip(model.eval()(f0, f1), loaded(f0, f1)):
        torch.testing.assert_close(a, b)


# --- Counterfactual branches and metrics ---------------------------------------------------

K = 4
PROGRAMS = [ACTION_MEANINGS.index(p) for p in cf.PROGRAMS]


@pytest.fixture
def encoded_dir(tmp_path):
    make_fake_shard(tmp_path / "enc", 3, [60, 20, 45], 8, 8)  # episode 1 is too short for min_step + horizon
    return tmp_path / "enc"


def test_pick_branches_alignment(encoded_dir):
    reader_rows = {}
    for b in cf.pick_branches([encoded_dir], n=2, K=K, horizon=10, min_step=12, seed=0, require_ready=False):
        r = ShardReader(b.shard)
        start, length = int(r.episodes["start"][b.episode]), int(r.episodes["length"][b.episode])
        assert 12 <= b.t <= length - 1 - 10
        assert [decode_frame(f)[0] for f in b.frames] == list(range(start + b.t - K + 1, start + b.t + 1))
        np.testing.assert_array_equal(b.prefix_actions, r.read("actions", slice(start, start + b.t)))
        np.testing.assert_array_equal(b.context_actions, b.prefix_actions[-(K - 1):])
        assert b.seed == r.episodes["seed"][b.episode]
        assert b.laser_idle == (r.read("ram", slice(start + b.t, start + b.t + 1))[0, cf.PLAYER_LASER_Y] == cf.LASER_IDLE)
        reader_rows[b.episode] = b.t
    assert sorted(reader_rows) == [0, 2]  # spread over the two long-enough episodes


def test_pick_branches_is_seeded_and_rejects(encoded_dir):
    pick = lambda seed: [b.t for b in cf.pick_branches([encoded_dir], 2, K, 10, 12, seed, require_ready=False)]  # noqa: E731
    assert pick(0) == pick(0)
    with pytest.raises(ValueError):
        cf.pick_branches([encoded_dir], 3, K, 10, 12, 0, require_ready=False)
    with pytest.raises(ValueError):  # encoded frames have no player (and their RAM bytes are never 246 at 85)
        cf.pick_branches([encoded_dir], 1, K, 10, 12, 0)


class ActionEcho(CopyLastFrame):
    """Writes the oldest action of its window into R and the newest into G of every pixel."""

    def predict_next(self, context, actions, num_steps=None, cfg_scale=None, ctx_level=0, generator=None):
        out = torch.zeros(len(context), 3, self.H, self.W)
        out[:, 0] = actions[:, 0, None, None].float() * 20
        out[:, 1] = actions[:, -1, None, None].float() * 20
        return out / 127.5 - 1


def test_model_branches_feed_context_actions_then_the_program(encoded_dir):
    branches = cf.pick_branches([encoded_dir], 2, K, 10, 12, 0, require_ready=False)
    model = ActionEcho(K, 3, 8, 8)
    out = cf.model_branches(model, branches, PROGRAMS, horizon=5, batch_size=5)
    assert out.shape == (2, len(PROGRAMS), 6, 8, 8, 3) and out.dtype == np.uint8
    for b, br in enumerate(branches):
        for p, prog in enumerate(PROGRAMS):
            np.testing.assert_array_equal(out[b, p, 0], br.frames[-1])  # step 0 is the real branch frame
            assert out[b, p, 1, 0, 0, 1] == 20 * prog  # g_{t+1} comes from the program's first action
            assert out[b, p, 1, 0, 0, 0] == 20 * br.context_actions[0]  # window starts at a_{t-K+1}
            assert out[b, p, K, 0, 0, 0] == 20 * prog  # by step K the whole window is the program


def test_copy_last_branches_repeat_the_branch_frame(encoded_dir):
    branches = cf.pick_branches([encoded_dir], 2, K, 10, 12, 0, require_ready=False)
    out = cf.model_branches(CopyLastFrame(K, 3, 8, 8), branches, PROGRAMS, horizon=4, batch_size=7)
    for b, br in enumerate(branches):
        assert (out[b] == br.frames[-1]).all()


def test_ordering_accuracy_with_margin():
    x = np.full((4, len(PROGRAMS), 3), 40.0)
    L, N, R = (cf.PROGRAMS.index(p) for p in ("LEFT", "NOOP", "RIGHT"))
    x[0, L, -1], x[0, R, -1] = 30, 50           # ordered
    x[1, L, -1], x[1, R, -1] = 38.5, 50         # LEFT within the 2-px margin
    x[2, L, -1], x[2, N, -1] = 30, np.nan       # player not found under NOOP
    x[3, L, -1], x[3, R, -1] = 30, 50
    x[3, L, -2] = 0                              # only the last step counts
    assert cf.ordering_accuracy(x, margin=2.0) == 0.5
    assert cf.ordering_accuracy(np.full((3, len(PROGRAMS), 2), 40.0), 2.0) == 0.0  # copy-last


def test_mean_dx_ignores_missing_steps():
    x = np.zeros((2, len(PROGRAMS), 4))
    R = cf.PROGRAMS.index("RIGHT")
    x[0, R] = [10, 12, 14, 16]
    x[1, R] = [20, np.nan, 26, 30]  # only the 26 -> 30 step is usable
    assert cf.mean_dx(x)["RIGHT"] == pytest.approx((2 + 2 + 2 + 4) / 4)
    assert cf.mean_dx(x)["NOOP"] == 0


def test_fire_response_conditions_on_idle_laser_and_window():
    laser = np.zeros((4, len(PROGRAMS), 6), bool)
    F = cf.PROGRAMS.index("FIRE")
    laser[0, F, 2] = True   # within 3 steps
    laser[1, F, 5] = True   # too late
    laser[2, F, 1] = True   # branch 2 already had a laser in flight: excluded
    laser[3, F, 0] = True   # a laser at step 0 does not count as a response
    laser[0, cf.PROGRAMS.index("RIGHTFIRE"), 3] = True
    idle = np.array([True, True, False, True])
    assert cf.fire_response(laser, idle, ["FIRE"], window=3) == (pytest.approx(1 / 3), 3)
    assert cf.fire_response(laser, idle, ["FIRE", "RIGHTFIRE"], window=3) == (pytest.approx(2 / 6), 6)


# --- Real emulator reference (needs ALE) ---------------------------------------------------

REAL_STEPS = 70  # past the ~32-step start-up phase, so the player responds to input


def collect_real(out_dir, n_episodes: int = 2, steps: int = REAL_STEPS):
    """A tiny real shard: seeded episodes of random actions, written like scripts/collect.py does."""
    rng = np.random.default_rng(0)
    attrs = {"env_id": "ALE/SpaceInvaders-v5", "frameskip": 4, "maxpool": True, "resolution": 84,
             "resize_method": "max", "collector_config": {"fixture": "test_control"}}
    with ShardWriter(out_dir, worker=0, frame_shape=(84, 84, 3), attrs=attrs) as w:
        for e in range(n_episodes):
            env = make_env(EVAL_SEED_BASE + e)
            obs, info = env.reset()
            ale = env.unwrapped.ale
            w.begin_episode(to_model_frame(obs, 84, "max"), ale.getRAM(), info["lives"], info["seed"],
                            info["noops"], SOURCE_RANDOM, -1.0)
            for _ in range(steps):
                a = int(rng.integers(NUM_ACTIONS))
                obs, reward, _, _, info = env.step(a)
                w.add_step(a, reward, to_model_frame(obs, 84, "max"), ale.getRAM(), info["lives"])
            w.end_episode(END_TRUNCATED, terminated=False)
            env.close()
    return out_dir


@pytest.fixture(scope="module")
def real_dir(tmp_path_factory):
    return collect_real(tmp_path_factory.mktemp("real"))


@pytest.mark.env
def test_real_branches_match_a_fresh_replay(real_dir):
    (branch,) = cf.pick_branches([real_dir], 1, K, horizon=10, min_step=40, seed=0)
    frames, ended = cf.real_branches(branch, PROGRAMS, horizon=10)
    assert frames.shape == (len(PROGRAMS), 11, 84, 84, 3) and not ended.any()
    for p in (cf.PROGRAMS.index("LEFT"), cf.PROGRAMS.index("FIRE")):  # each branch starts from the same state
        env = make_env(branch.seed)
        env.reset()
        for a in branch.prefix_actions:
            env.step(int(a))
        for s in range(1, 11):
            obs, *_ = env.step(PROGRAMS[p])
            np.testing.assert_array_equal(frames[p, s], to_model_frame(obs, 84, "max"), err_msg=f"program {p} step {s}")
    assert branch.laser_idle and not np.isnan(detectors.player_x(branch.frames[-1:]))[0]
    x = detectors.player_x(frames[:, -1])
    assert x[cf.PROGRAMS.index("LEFT")] < x[cf.PROGRAMS.index("NOOP")] < x[cf.PROGRAMS.index("RIGHT")]


@pytest.mark.env
def test_real_branches_refuse_a_branch_the_replay_does_not_reach(real_dir):
    (branch,) = cf.pick_branches([real_dir], 1, K, horizon=10, min_step=40, seed=0)
    branch.prefix_actions = branch.prefix_actions[:-1]
    with pytest.raises(RuntimeError, match="does not reach"):
        cf.real_branches(branch, PROGRAMS, horizon=2)


# --- Scripts -----------------------------------------------------------------------------


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_idm_end_to_end(tmp_path):
    data = tmp_path / "data"
    make_fake_shard(data, 6, [20] * 6, 12, 12, encode_rows=False)
    out = _load("train_idm").main([
        "--config", str(REPO / "configs" / "train" / "idm.yaml"), "--set", f"dataset_dirs=[{data}]",
        "--set", "val_fraction=0.5", "--set", "steps=6", "--set", "batch_size=8", "--set", "num_workers=0",
        "--set", "log_every=3", "--set", "eval_every=3", "--set", "val_batches=1", "--set", "model.ch=[8,16]",
        "--set", "model.hidden=16", "--set", f"out_dir={tmp_path / 'idm'}"])
    metrics = json.loads((out / "metrics.json").read_text())
    val = metrics["val"]
    assert metrics["steps"] == 6 and val["n"] > 0
    assert np.sum(val["move"]["confusion"]) == np.sum(val["fire"]["confusion"]) == val["n"]
    assert 0 <= val["move"]["accuracy"] <= 1 and 0 < val["fire"]["majority"] <= 1
    model = idm.load_idm(out / "idm.pt")
    assert model.config["H"] == 12 and torch.load(out / "idm.pt", weights_only=False)["step"] == 6


@pytest.mark.env
def test_eval_counterfactual_end_to_end(real_dir, tmp_path):
    """copy_last and a tiny checkpoint against the real emulator, with an (untrained) IDM."""
    (tmp_path / "run" / "checkpoints").mkdir(parents=True)
    ckpt = _tiny_ckpt(tmp_path / "run" / "checkpoints" / "ckpt_last.pt")
    idm.save_idm(tmp_path / "idm.pt", idm.IDM(C=3, H=84, W=84, ch=(8,), hidden=8))
    out = _load("eval_counterfactual").main([
        "--config", str(REPO / "configs" / "eval" / "counterfactual.yaml"), "--models", "copy_last", str(ckpt),
        "--n", "2", "--set", f"dataset_dirs=[{real_dir}]", "--set", "horizon=6", "--set", "min_step=40",
        "--set", "batch_size=5", "--set", f"idm={tmp_path / 'idm.pt'}", "--set", "strip_branches=1",
        "--set", f"out_dir={tmp_path / 'out'}"])
    result = json.loads((out / "metrics.json").read_text())
    assert list(result["results"]) == ["real", "copy_last", "run_ckpt_last@ctx0"]
    assert len(result["branches"]) == 2 and all(b["laser_idle"] and b["t"] >= 40 for b in result["branches"])
    copy_last, real = result["results"]["copy_last"], result["results"]["real"]
    assert copy_last["ordering_accuracy"] == 0.0  # the roadmap's expected copy-last result
    assert all(v == 0 for v in copy_last["dx_per_step"].values())
    assert real["dx_per_step"]["LEFT"] < 0 < real["dx_per_step"]["RIGHT"]
    assert real["idm"]["n"] == copy_last["idm"]["n"] == 2 * len(cf.PROGRAMS) * 6
    assert set(copy_last["final_x_error_px"]) == set(cf.PROGRAMS)
    assert result["calibration"]["n"] > 0 and result["calibration"]["mae_px"] < 1.0
    assert (out / "counterfactual.png").stat().st_size > 0
