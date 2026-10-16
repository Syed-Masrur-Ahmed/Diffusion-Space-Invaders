"""Policies: random holds, epsilon mixing, and the pretrained zoo agent (CP1-C)."""

import numpy as np
import pytest

from wm.data.schema import NUM_ACTIONS
from wm.env import to_agent_obs

# Imported after the constants so a missing policy class fails the tests that need it.
from wm.agents.policies import EpsilonMixPolicy, RandomPolicy
from wm.agents.pretrained import ZooAgentPolicy


def _raw(value: int = 0) -> np.ndarray:
    return np.full((210, 160, 3), value, dtype=np.uint8)


class _StepPolicy:
    """Deterministic stand-in: action is the step index mod NUM_ACTIONS."""

    def __init__(self) -> None:
        self.i = 0
        self.seen: list[np.ndarray] = []

    def reset(self, seed: int, raw_obs: np.ndarray) -> None:
        self.i = 0
        self.seed = seed

    def act(self, raw_obs: np.ndarray) -> int:
        self.seen.append(raw_obs)
        action = self.i % NUM_ACTIONS
        self.i += 1
        return action


class _ConstPolicy:
    """Stand-in that always returns one action and records the frames it sees."""

    def __init__(self, action: int) -> None:
        self.action = action
        self.seen: list[np.ndarray] = []

    def reset(self, seed: int, raw_obs: np.ndarray) -> None:
        self.seed = seed

    def act(self, raw_obs: np.ndarray) -> int:
        self.seen.append(raw_obs)
        return self.action


class _FakeRng:
    """Always starts a hold of `hold` steps on action `action`."""

    def __init__(self, action: int, hold: int) -> None:
        self.action = action
        self.hold = hold
        self.n_random = 0
        self.n_integers = 0
        self.n_geometric = 0

    def random(self) -> float:
        self.n_random += 1
        return 0.0

    def integers(self, low: int, high: int) -> int:
        self.n_integers += 1
        assert (low, high) == (0, NUM_ACTIONS)
        return self.action

    def geometric(self, p: float) -> int:
        self.n_geometric += 1
        return self.hold


def test_random_policy_hold_mean_and_uniform_actions():
    obs = _raw()
    policy = RandomPolicy(mean_hold=4)
    policy.reset(0, obs)
    actions = []
    holds: list[int] = []
    run = 0
    for _ in range(10_000):
        actions.append(policy.act(obs))
        run += 1
        # _left is how many further steps this hold still has after the one just taken.
        if policy._left == 0:
            holds.append(run)
            run = 0
    assert holds
    assert abs(float(np.mean(holds)) - 4) <= 0.5
    assert float(np.std(holds)) > 1.0  # a geometric hold, not a constant
    assert all(type(a) is int and 0 <= a < NUM_ACTIONS for a in actions)
    freq = np.bincount(actions, minlength=NUM_ACTIONS) / len(actions)
    assert np.all(np.abs(freq - 1 / NUM_ACTIONS) < 0.05)


def test_random_policy_repeats_for_the_same_seed():
    obs = _raw()

    def rollout(seed: int) -> list[int]:
        policy = RandomPolicy()
        policy.reset(seed, obs)
        return [policy.act(obs) for _ in range(200)]

    assert rollout(0) == rollout(0)
    assert rollout(0) != rollout(1)


def test_epsilon_zero_matches_base_and_one_is_never_the_base():
    frames = [_raw(i) for i in range(12)]

    base0 = _StepPolicy()
    mix0 = EpsilonMixPolicy(base0, epsilon=0)
    twin = _StepPolicy()
    mix0.reset(1, frames[0])
    mix0.rng = _FakeRng(action=4, hold=3)  # ignored at epsilon 0
    twin.reset(1, frames[0])
    got0 = [mix0.act(frame) for frame in frames]
    expect = [twin.act(frame) for frame in frames]
    assert got0 == expect
    assert base0.seed == 1
    assert mix0.rng.n_random == 0

    base = _ConstPolicy(0)
    mix1 = EpsilonMixPolicy(base, epsilon=1, mean_hold=2)
    mix1.reset(2, frames[0])
    fake = _FakeRng(action=4, hold=3)
    mix1.rng = fake
    got1 = [mix1.act(frame) for frame in frames]
    assert got1 == [4] * len(frames)
    assert all(action != base.action for action in got1)
    assert base.seen == frames  # the base still observes every frame
    assert fake.n_integers == len(frames) // 3
    assert fake.n_random == fake.n_integers == fake.n_geometric


class _FakeSpace:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeModel:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.observation_space = _FakeSpace(shape)
        self.seen: list[np.ndarray] = []

    def predict(self, obs: np.ndarray, deterministic: bool = False):
        assert deterministic is False
        self.seen.append(np.asarray(obs).copy())
        return np.int64(2), None


def _planes(obs: np.ndarray, channels_first: bool) -> list[np.ndarray]:
    if channels_first:
        return [obs[c] for c in range(obs.shape[0])]
    return [obs[..., c] for c in range(obs.shape[-1])]


@pytest.mark.parametrize("shape", [(4, 84, 84), (84, 84, 4)])
def test_zoo_agent_frame_stack_alignment(shape):
    model = _FakeModel(shape)
    policy = ZooAgentPolicy(model=model)
    first, second = _raw(0), _raw(255)
    policy.reset(0, first)
    first_action = policy.act(first)
    second_action = policy.act(second)
    assert type(first_action) is int and type(second_action) is int
    assert first_action == 2 and second_action == 2

    channels_first = shape[0] == 4
    newest = -1
    planes0 = _planes(model.seen[0], channels_first)
    planes1 = _planes(model.seen[1], channels_first)
    grey0 = to_agent_obs(first)
    grey1 = to_agent_obs(second)
    assert model.seen[0].shape == shape
    # reset already stacked `first`; the first act must not push it again.
    assert np.array_equal(planes0[newest], grey0)
    assert all(np.all(plane == 0) for plane in planes0[:newest])
    assert np.array_equal(planes1[newest], grey1)
    assert np.array_equal(planes1[newest - 1], grey0)


@pytest.mark.network
@pytest.mark.env
def test_zoo_agent_loads_and_acts():
    from wm.env import make_env

    env = make_env(0)
    try:
        obs, _info = env.reset()
        policy = ZooAgentPolicy()
        policy.reset(0, obs)
        action = policy.act(obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
        assert not terminated and not truncated
        action2 = policy.act(obs)
    finally:
        env.close()
    assert type(action) is int and type(action2) is int
    assert 0 <= action < NUM_ACTIONS
    assert 0 <= action2 < NUM_ACTIONS


def _episode_return(policy, seed: int) -> float:
    from wm.env import make_env

    env = make_env(seed)
    try:
        obs, _info = env.reset()
        policy.reset(seed, obs)
        total = 0.0
        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            done = terminated or truncated
        return total
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.env
def test_zoo_agent_mean_score_at_least_twice_random():
    agent = ZooAgentPolicy()
    agent_scores = [_episode_return(agent, seed) for seed in range(3)]
    random_scores = [_episode_return(RandomPolicy(), seed) for seed in range(3)]
    assert float(np.mean(agent_scores)) >= 2 * float(np.mean(random_scores))
