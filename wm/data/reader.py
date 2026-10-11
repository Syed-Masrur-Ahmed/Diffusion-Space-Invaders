"""HDF5 shard reader (ROADMAP 4.4, [FROZEN])."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from wm.data.schema import EPISODES_GROUP, ROW_FIELDS

SHARD_GLOB = "shard_w*_*.h5"


def list_shards(dataset_dir: str | Path) -> list[Path]:
    """Sorted shard paths in a dataset directory."""
    return sorted(Path(dataset_dir).glob(SHARD_GLOB))


class ShardReader:
    """Read access to one shard; opens the file lazily on first access so it survives pickling."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: h5py.File | None = None
        self._attrs: dict[str, Any] | None = None
        self._episodes: dict[str, np.ndarray] | None = None

    def __getstate__(self) -> dict:
        # Never send an open HDF5 handle across fork/pickle (pitfall 5): each worker reopens.
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    @property
    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        """Close the file handle; the next access reopens it."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def num_rows(self) -> int:
        """Number of rows (frames) in the shard."""
        return self._file["actions"].shape[0]

    @property
    def attrs(self) -> dict[str, Any]:
        """File attributes (4.4)."""
        if self._attrs is None:
            self._attrs = dict(self._file.attrs)
        return self._attrs

    @property
    def episodes(self) -> dict[str, np.ndarray]:
        """Per-episode datasets from the `episodes/` group."""
        if self._episodes is None:
            group = self._file[EPISODES_GROUP]
            self._episodes = {name: group[name][()] for name in group}
        return self._episodes

    def episode_rows(self, e: int) -> slice:
        """Row slice of shard-local episode `e`."""
        start = int(self.episodes["start"][e])
        return slice(start, start + int(self.episodes["length"][e]))

    def read(self, name: str, rows: slice | np.ndarray) -> np.ndarray:
        """Read a per-row dataset at a slice or sorted index array."""
        if name not in ROW_FIELDS:
            raise KeyError(f"{name!r} is not a per-row dataset")
        ds = self._file[name]
        if isinstance(rows, slice):
            return ds[rows]
        rows = np.asarray(rows)
        if np.any(np.diff(rows) < 0):
            raise ValueError("row indices must be sorted")
        # h5py fancy indexing needs strictly increasing indices, so read unique rows and expand.
        unique, inverse = np.unique(rows, return_inverse=True)
        return ds[unique][inverse]
