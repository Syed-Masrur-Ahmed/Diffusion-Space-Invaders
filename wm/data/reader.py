"""HDF5 shard reader (ROADMAP 4.4, [FROZEN]). Implemented by CP1-B."""

from pathlib import Path
from typing import Any

import numpy as np


def list_shards(dataset_dir: str | Path) -> list[Path]:
    """Sorted shard paths in a dataset directory."""
    raise NotImplementedError("CP1-B")


class ShardReader:
    """Read access to one shard; opens the file lazily on first access so it survives pickling."""

    def __init__(self, path: str | Path) -> None:
        raise NotImplementedError("CP1-B")

    @property
    def num_rows(self) -> int:
        """Number of rows (frames) in the shard."""
        raise NotImplementedError("CP1-B")

    @property
    def attrs(self) -> dict[str, Any]:
        """File attributes (4.4)."""
        raise NotImplementedError("CP1-B")

    @property
    def episodes(self) -> dict[str, np.ndarray]:
        """Per-episode datasets from the `episodes/` group."""
        raise NotImplementedError("CP1-B")

    def episode_rows(self, e: int) -> slice:
        """Row slice of shard-local episode `e`."""
        raise NotImplementedError("CP1-B")

    def read(self, name: str, rows: slice | np.ndarray) -> np.ndarray:
        """Read a per-row dataset at a slice or sorted index array."""
        raise NotImplementedError("CP1-B")
