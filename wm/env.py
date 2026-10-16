"""Space Invaders environment and frame transforms (ROADMAP 4.3). Owned by CP1-A after Phase 0."""

from typing import Any

import ale_py
import cv2
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

from wm.data.schema import ACTION_MEANINGS

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"
NOOP = ACTION_MEANINGS.index("NOOP")


def _max_last_two(frames: list[np.ndarray]) -> np.ndarray:
    # Space Invaders draws some sprites (lasers especially) on alternate frames only.
    return frames[-1] if len(frames) == 1 else np.maximum(frames[-2], frames[-1])


class SeededNoopReset(gym.Wrapper):
    """Resets with seed + episode counter, then takes a seed-derived number of base NOOP steps."""

    def __init__(self, env: gym.Env, seed: int, noop_max: int = 30) -> None:
        super().__init__(env)
        self.seed = seed
        self.noop_max = noop_max
        self._episode = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:  # an explicit seed restarts the counter from that seed
            self.seed, self._episode = seed, 0
        episode_seed = self.seed + self._episode
        self._episode += 1

        obs, info = self.env.reset(seed=episode_seed, options=options)
        n_noop = int(np.random.default_rng(episode_seed).integers(0, self.noop_max + 1))
        frames = [obs]
        for _ in range(n_noop):
            obs, _, _, _, info = self.env.step(NOOP)
            frames = [frames[-1], obs]
        obs = _max_last_two(frames) if n_noop >= 2 else frames[0]
        return obs, {**info, "noops": n_noop, "seed": episode_seed}


class MaxPoolSkip(gym.Wrapper):
    """Repeats an action `skip` times, sums rewards, and max-pools the last two raw frames."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self.skip = skip

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        frames: list[np.ndarray] = []
        total = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            frames = [*frames[-1:], obs]
            if terminated or truncated:
                break
        return _max_last_two(frames), total, terminated, truncated, info


def make_env(seed: int, noop_max: int = 30, max_episode_steps: int = 27_000) -> gym.Env:
    """Create the project's Space Invaders env: seeded no-op reset, 4-frame max-pool skip, time limit."""
    base = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, full_action_space=False)
    assert base.unwrapped.get_action_meanings() == ACTION_MEANINGS
    env = SeededNoopReset(base, seed, noop_max)
    env = MaxPoolSkip(env, skip=4)
    return TimeLimit(env, max_episode_steps=max_episode_steps)


def to_model_frame(raw: np.ndarray, size: int, method: str = "area") -> np.ndarray:
    """Resize a raw (210,160,3) uint8 frame to the model's (size,size,3) uint8 frame."""
    if method == "area":
        return cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)  # cv2 takes (W, H)
    if method == "max":
        import torch  # lazy: only "max" needs torch, and the play scripts shouldn't pay for importing it

        # Max-pooling keeps one-pixel bright objects (lasers) that area averaging dims. Float
        # because the CPU max-pool has no uint8 kernel; uint8 -> float32 -> uint8 is exact.
        chw = torch.from_numpy(raw).permute(2, 0, 1).float()
        pooled = torch.nn.functional.adaptive_max_pool2d(chw, (size, size))
        return pooled.permute(1, 2, 0).to(torch.uint8).contiguous().numpy()
    raise ValueError(f"unknown resize method {method!r}")


def to_agent_obs(raw: np.ndarray) -> np.ndarray:
    """Greyscale 84x84 uint8 observation matching SB3's WarpFrame."""
    grey = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    return cv2.resize(grey, (84, 84), interpolation=cv2.INTER_AREA)


class AgentObsStack:
    """The last `n` agent observations, channels last (84,84,n), zero-filled on reset like VecFrameStack."""

    def __init__(self, n: int = 4) -> None:
        self.stack = np.zeros((84, 84, n), np.uint8)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Zero the stack, push `obs`, and return the stack."""
        self.stack = np.zeros_like(self.stack)  # new array, so stacks returned earlier stay intact
        return self.push(obs)

    def push(self, obs: np.ndarray) -> np.ndarray:
        """Drop the oldest observation, append `obs`, and return the stack."""
        self.stack = np.roll(self.stack, -1, axis=-1)
        self.stack[..., -1] = obs
        return self.stack
