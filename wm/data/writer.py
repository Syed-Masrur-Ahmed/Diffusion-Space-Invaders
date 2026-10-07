"""HDF5 shard writer (ROADMAP 4.4, [FROZEN]). Implemented by CP1-B."""

from pathlib import Path

import numpy as np

from wm.data.schema import MAX_ROWS


class ShardWriter:
    """Writes episodes to rotating HDF5 shards `shard_w<worker:03d>_<index:05d>.h5` (4.4)."""

    def __init__(self, out_dir: str | Path, worker: int, frame_shape: tuple[int, int, int],
                 attrs: dict, max_rows: int = MAX_ROWS) -> None:
        raise NotImplementedError("CP1-B")

    def begin_episode(self, frame: np.ndarray, ram: np.ndarray, lives: int, seed: int,
                      noops: int, source: int, epsilon: float) -> None:
        """Start an episode with its first frame; may rotate to a new shard first."""
        raise NotImplementedError("CP1-B")

    def add_step(self, action: int, reward: float, next_frame: np.ndarray,
                 next_ram: np.ndarray, next_lives: int) -> None:
        """Record `action` at the current row and the row it produces (buffered)."""
        raise NotImplementedError("CP1-B")

    def end_episode(self, end_reason: int, terminated: bool) -> None:
        """Finish the episode: last row gets action -1, and terminal if `terminated`."""
        raise NotImplementedError("CP1-B")

    def close(self) -> None:
        """Flush buffers and close the current shard."""
        raise NotImplementedError("CP1-B")
