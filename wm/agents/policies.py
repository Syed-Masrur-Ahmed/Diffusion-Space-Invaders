"""Policy interface used by the collector (ROADMAP 5, CP1-C)."""

from typing import Protocol

import numpy as np

from wm.data.schema import NUM_ACTIONS


class Policy(Protocol):
    """Chooses actions from raw MaxPoolSkip observations (210,160,3) uint8."""

    def reset(self, seed: int, raw_obs: np.ndarray) -> None:
        """Start a new episode from its first observation."""
        ...

    def act(self, raw_obs: np.ndarray) -> int:
        """Return an action in 0..5 for the current observation."""
        ...


class RandomPolicy:
    """Uniform random action, held for a 1+Geometric number of steps."""

    def __init__(self, mean_hold: float = 4) -> None:
        if mean_hold < 1:
            raise ValueError(f"mean_hold must be >= 1, got {mean_hold}")
        self.mean_hold = float(mean_hold)
        self.rng = np.random.default_rng()
        self._left = 0
        self._action = 0

    def reset(self, seed: int, raw_obs: np.ndarray) -> None:
        """Reseed the policy. The same seed repeats the same action sequence."""
        del raw_obs
        self.rng = np.random.default_rng(seed)
        self._left = 0

    def act(self, raw_obs: np.ndarray) -> int:
        """Return the current held action, sampling a new one when the hold ends."""
        del raw_obs
        if self._left == 0:
            self._action = int(self.rng.integers(0, NUM_ACTIONS))
            # numpy's geometric is the trial count until the first success (mean 1/p):
            # that is 1 + a failures-until-success geometric draw.
            self._left = int(self.rng.geometric(1.0 / self.mean_hold))
        self._left -= 1
        return self._action


class EpsilonMixPolicy:
    """With probability epsilon, hold a random action; otherwise follow `base`.

    `base.act` is called on every frame, including during a random hold, so a
    frame-stacking base policy stays aligned with the episode.
    """

    def __init__(self, base: Policy, epsilon: float, mean_hold: float = 2) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if mean_hold < 1:
            raise ValueError(f"mean_hold must be >= 1, got {mean_hold}")
        self.base = base
        self.epsilon = float(epsilon)
        self.mean_hold = float(mean_hold)
        self.rng = np.random.default_rng()
        self._left = 0
        self._action = 0

    def reset(self, seed: int, raw_obs: np.ndarray) -> None:
        """Reseed the mixer and reset the base policy on the same observation."""
        self.rng = np.random.default_rng(seed)
        self._left = 0
        self.base.reset(seed, raw_obs)

    def act(self, raw_obs: np.ndarray) -> int:
        """Return the base action, or a held random action when a mix run is active."""
        base_action = self.base.act(raw_obs)
        if self._left > 0:
            self._left -= 1
            return self._action
        if self.epsilon > 0.0 and float(self.rng.random()) < self.epsilon:
            self._action = int(self.rng.integers(0, NUM_ACTIONS))
            self._left = int(self.rng.geometric(1.0 / self.mean_hold)) - 1
            return self._action
        return base_action
