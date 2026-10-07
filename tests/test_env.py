import numpy as np
import pytest

from wm.env import AgentObsStack, make_env, to_agent_obs, to_model_frame


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


def test_frame_transforms_shapes():
    raw = np.random.default_rng(0).integers(0, 256, (210, 160, 3), dtype=np.uint8)
    frame = to_model_frame(raw, 84)
    assert frame.shape == (84, 84, 3) and frame.dtype == np.uint8
    agent = to_agent_obs(raw)
    assert agent.shape == (84, 84) and agent.dtype == np.uint8


def test_agent_obs_stack_shifts_and_zero_fills():
    stack = AgentObsStack(n=4)
    a = np.full((84, 84), 1, np.uint8)
    b = np.full((84, 84), 2, np.uint8)
    s = stack.reset(a)
    assert s.shape == (84, 84, 4) and s.dtype == np.uint8
    assert s[0, 0].tolist() == [0, 0, 0, 1]
    s = stack.push(b)
    assert s[0, 0].tolist() == [0, 0, 1, 2]
