import importlib.util
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from wm.env import ENV_ID, AgentObsStack, MaxPoolSkip, make_env, to_agent_obs, to_model_frame

INSPECT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_frames.py"


def _load_inspect_frames():
    spec = importlib.util.spec_from_file_location("inspect_frames", INSPECT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_env() -> gym.Env:
    return gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, full_action_space=False)


class _ScriptedEnv(gym.Env):
    """Base step t returns frame [[10t, 100-10t]], reward t, lives 3-t, and terminates at step `end`."""

    observation_space = gym.spaces.Box(0, 255, (1, 2, 3), np.uint8)
    action_space = gym.spaces.Discrete(6)

    def __init__(self, end: int | None = None) -> None:
        self.end, self.t = end, 0

    @staticmethod
    def frame(t: int) -> np.ndarray:
        return np.array([[[10 * t] * 3, [100 - 10 * t] * 3]], np.uint8)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self.frame(0), {"lives": 3}

    def step(self, action):
        self.t += 1
        return self.frame(self.t), float(self.t), self.t == self.end, False, {"lives": 3 - self.t}


# --- MaxPoolSkip -------------------------------------------------------------------------


def test_maxpoolskip_full_skip_maxes_last_two_frames_and_sums_rewards():
    env = MaxPoolSkip(_ScriptedEnv())
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    np.testing.assert_array_equal(obs, np.maximum(_ScriptedEnv.frame(3), _ScriptedEnv.frame(4)))
    assert reward == 1 + 2 + 3 + 4
    assert not terminated and not truncated
    assert info["lives"] == 3 - 4  # from the last base step
    assert env.unwrapped.t == 4


@pytest.mark.parametrize("end", [1, 2, 3])
def test_maxpoolskip_early_termination_returns_valid_obs(end):
    env = MaxPoolSkip(_ScriptedEnv(end=end))
    env.reset()
    obs, reward, terminated, _, info = env.step(0)
    frames = [_ScriptedEnv.frame(t) for t in range(1, end + 1)]
    expected = frames[0] if end == 1 else np.maximum(frames[-2], frames[-1])
    np.testing.assert_array_equal(obs, expected)
    assert obs.dtype == np.uint8 and obs.shape == (1, 2, 3)
    assert reward == sum(range(1, end + 1))
    assert terminated and info["lives"] == 3 - end
    assert env.unwrapped.t == end  # stopped stepping the base env


@pytest.mark.env
def test_maxpoolskip_matches_manual_base_steps():
    env, ref = MaxPoolSkip(_base_env()), _base_env()
    env.reset(seed=0)
    ref.reset(seed=0)
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(300):
        action = int(rng.integers(0, 6))
        obs, reward, terminated, truncated, info = env.step(action)
        steps = [ref.step(action) for _ in range(4)]
        assert reward == sum(float(s[1]) for s in steps)
        np.testing.assert_array_equal(obs, np.maximum(steps[-2][0], steps[-1][0]))
        assert info["lives"] == steps[-1][4]["lives"]
        total += reward
        if terminated or truncated:
            break
    assert total > 0  # the comparison covered steps that actually scored
    env.close()
    ref.close()


# --- make_env ------------------------------------------------------------------------------


@pytest.mark.env
def test_make_env_reset_and_step():
    env = make_env(0)
    obs, info = env.reset()
    assert obs.shape == (210, 160, 3) and obs.dtype == np.uint8
    assert 0 <= info["noops"] <= 30
    assert info["seed"] == 0

    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (210, 160, 3) and obs.dtype == np.uint8
    assert isinstance(float(reward), float)
    assert info["lives"] == 3
    assert not terminated and not truncated
    env.close()


@pytest.mark.env
def test_later_resets_advance_episode_seed():
    env = make_env(5)
    seeds = [env.reset()[1]["seed"] for _ in range(3)]
    assert seeds == [5, 6, 7]
    env.close()


@pytest.mark.env
def test_noops_derived_from_seed():
    expected = int(np.random.default_rng(3).integers(0, 31))
    env = make_env(3)
    assert env.reset()[1]["noops"] == expected
    env.close()


@pytest.mark.env
def test_noops_reproducible_in_range_and_actually_taken():
    env = make_env(0)
    for seed in range(20):
        _, info = env.reset(seed=seed)
        n = int(np.random.default_rng(seed).integers(0, 31))
        assert info["noops"] == n and 0 <= n <= 30
        assert info["seed"] == seed
        assert env.unwrapped.ale.getEpisodeFrameNumber() == n
    env.close()


@pytest.mark.env
def test_determinism_same_seed_same_actions():
    actions = np.random.default_rng(123).integers(0, 6, size=200)
    runs = []
    for _ in range(2):
        env = make_env(7)
        obs, info = env.reset()
        trace = [(obs, 0.0, info["lives"])]
        for a in actions:
            obs, reward, terminated, truncated, info = env.step(int(a))
            trace.append((obs, reward, info["lives"]))
            if terminated or truncated:
                break
        runs.append(trace)
        env.close()
    assert len(runs[0]) == len(runs[1])
    for (o1, r1, l1), (o2, r2, l2) in zip(*runs):
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2 and l1 == l2


@pytest.mark.env
def test_truncates_after_max_episode_steps():
    env = make_env(0, max_episode_steps=5)
    env.reset()
    for _ in range(4):
        assert not env.step(0)[3]
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    env.close()


@pytest.mark.env
def test_game_over_terminates_with_valid_obs():
    env = make_env(0)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(5_000):
        obs, _, terminated, truncated, info = env.step(int(rng.integers(0, 6)))
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert obs.shape == (210, 160, 3) and obs.dtype == np.uint8
    assert info["lives"] == 0
    env.close()


# --- Frame transforms ----------------------------------------------------------------------


def _raw(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (210, 160, 3), dtype=np.uint8)


@pytest.mark.parametrize("size", [84, 96])
@pytest.mark.parametrize("method", ["area", "max"])
def test_to_model_frame_shape_and_dtype(size, method):
    frame = to_model_frame(_raw(), size, method)
    assert frame.shape == (size, size, 3) and frame.dtype == np.uint8


def test_to_agent_obs_shape_and_dtype():
    agent = to_agent_obs(_raw())
    assert agent.shape == (84, 84) and agent.dtype == np.uint8


@pytest.mark.parametrize("size", [84, 96])
def test_max_method_matches_numpy_adaptive_max_pool(size):
    raw = _raw(1)
    out = to_model_frame(raw, size, "max")
    H, W = raw.shape[:2]
    for i in range(size):
        r0, r1 = (i * H) // size, math.ceil((i + 1) * H / size)
        for j in range(size):
            c0, c1 = (j * W) // size, math.ceil((j + 1) * W / size)
            np.testing.assert_array_equal(out[i, j], raw[r0:r1, c0:c1].max(axis=(0, 1)))


@pytest.mark.parametrize("size", [84, 96])
def test_max_keeps_one_pixel_object_that_area_dims(size):
    raw = np.zeros((210, 160, 3), np.uint8)
    raw[100, 81] = 142  # a one-pixel laser segment on the black background
    assert to_model_frame(raw, size, "max").max() == 142
    assert to_model_frame(raw, size, "area").max() < 142


def test_to_model_frame_unknown_method_raises():
    with pytest.raises(ValueError):
        to_model_frame(_raw(), 84, "bilinear")


# --- AgentObsStack -------------------------------------------------------------------------


def test_agent_obs_stack_shifts_and_zero_fills():
    stack = AgentObsStack(n=4)
    a = np.full((84, 84), 1, np.uint8)
    b = np.full((84, 84), 2, np.uint8)
    s = stack.reset(a)
    assert s.shape == (84, 84, 4) and s.dtype == np.uint8
    assert s[0, 0].tolist() == [0, 0, 0, 1]
    s = stack.push(b)
    assert s[0, 0].tolist() == [0, 0, 1, 2]


def test_agent_obs_stack_drops_oldest_and_reset_clears():
    stack = AgentObsStack(n=4)
    stack.reset(np.full((84, 84), 1, np.uint8))
    for v in range(2, 7):
        s = stack.push(np.full((84, 84), v, np.uint8))
    assert s[5, 5].tolist() == [3, 4, 5, 6]
    s = stack.reset(np.full((84, 84), 9, np.uint8))
    assert s[5, 5].tolist() == [0, 0, 0, 9]


# --- Laser flicker (uses scripts/inspect_frames.py) ----------------------------------------


@pytest.mark.env
def test_maxpooled_frames_show_laser_at_least_as_often_as_raw_frames():
    inspect_frames = _load_inspect_frames()
    # Seeds 0 and 2 have an even and an odd no-op count, so they cover both flicker phases.
    counts = inspect_frames.laser_stats(seeds=[0, 2], steps=100)
    visible_max, total = counts["raw max-pooled"]
    visible_last, _ = counts["raw last frame"]
    assert total >= 20  # the rollout produced enough laser frames to compare
    assert visible_max >= visible_last
    assert visible_max == total  # the laser is defined from the max-pooled frame, so it must show there
