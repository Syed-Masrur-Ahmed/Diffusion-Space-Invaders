"""Policy interface used by the collector (ROADMAP 5, CP1-C). Concrete policies are added by CP1-C."""

from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Chooses actions from raw MaxPoolSkip observations (210,160,3) uint8."""

    def reset(self, seed: int, raw_obs: np.ndarray) -> None:
        """Start a new episode from its first observation."""
        ...

    def act(self, raw_obs: np.ndarray) -> int:
        """Return an action in 0..5 for the current observation."""
        ...
