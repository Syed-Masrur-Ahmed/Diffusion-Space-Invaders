"""HDF5 shard writer (ROADMAP 4.4, [FROZEN])."""

import json
import subprocess
import warnings
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import h5py
import numpy as np

from wm.data.schema import (EPISODE_FIELDS, EPISODE_UID_STRIDE, EPISODES_GROUP, FILE_ATTRS,
                            FLUSH_EVERY, FRAME_CHUNK_ROWS, FRAME_COMPRESSION, LAST_ROW_ACTION,
                            MAX_ROWS, NUM_ACTIONS, RAM_SIZE, ROW_FIELDS, SCHEMA_VERSION)

# Attrs the writer always sets itself; everything else in FILE_ATTRS comes from the caller,
# except the package versions, which are filled in when the caller leaves them out.
_WRITER_ATTRS = ("schema_version", "git_commit", "created_utc")
_VERSION_ATTRS = {"ale_py_version": "ale-py", "gymnasium_version": "gymnasium"}


def _git_commit(cwd: Path) -> str:
    """`git rev-parse HEAD` in `cwd`, or "unknown" outside a repo or without git."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


class ShardWriter:
    """Writes episodes to rotating HDF5 shards `shard_w<worker:03d>_<index:05d>.h5` (4.4)."""

    def __init__(self, out_dir: str | Path, worker: int, frame_shape: tuple[int, int, int],
                 attrs: dict, max_rows: int = MAX_ROWS) -> None:
        missing = [k for k in FILE_ATTRS
                   if k not in attrs and k not in _WRITER_ATTRS and k not in _VERSION_ATTRS]
        if missing:
            raise ValueError(f"ShardWriter attrs missing: {missing}")
        if len(frame_shape) != 3 or frame_shape[2] != 3:
            raise ValueError(f"frame_shape must be (H, W, 3), got {frame_shape}")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.worker = worker
        self.frame_shape = tuple(frame_shape)
        self.max_rows = max_rows
        self._attrs = {k: json.dumps(v) if isinstance(v, dict) else v for k, v in attrs.items()}
        for k, pkg in _VERSION_ATTRS.items():
            self._attrs.setdefault(k, _package_version(pkg))
        self._attrs["schema_version"] = SCHEMA_VERSION
        self._attrs["git_commit"] = _git_commit(Path(__file__).resolve().parent)

        self._file: h5py.File | None = None
        self._shard_index = -1
        self._local_counter = 0  # episodes finished by this writer, across all its shards
        self._closed = False
        self._buffer: dict[str, list] = {name: [] for name in ROW_FIELDS}
        self._pending: tuple[np.ndarray, np.ndarray, int] | None = None  # newest row; action unknown
        self._episode: dict | None = None

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def _rows_on_disk(self) -> int:
        return self._file["actions"].shape[0]

    def _check_row(self, frame: np.ndarray, ram: np.ndarray) -> None:
        if frame.shape != self.frame_shape or frame.dtype != np.uint8:
            raise ValueError(f"frame must be uint8 {self.frame_shape}, got {frame.dtype} {frame.shape}")
        if ram.shape != (RAM_SIZE,) or ram.dtype != np.uint8:
            raise ValueError(f"ram must be uint8 ({RAM_SIZE},), got {ram.dtype} {ram.shape}")

    def _open_shard(self) -> None:
        self._shard_index += 1
        path = self.out_dir / f"shard_w{self.worker:03d}_{self._shard_index:05d}.h5"
        f = h5py.File(path, "w-")  # "w-" fails if the file exists: never clobber collected data
        f.attrs.update(self._attrs)
        f.attrs["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for name, (dtype, trailing) in ROW_FIELDS.items():
            if name == "frames":
                f.create_dataset(name, (0, *self.frame_shape), dtype, maxshape=(None, *self.frame_shape),
                                 chunks=(FRAME_CHUNK_ROWS, *self.frame_shape),
                                 compression=FRAME_COMPRESSION)
            else:
                f.create_dataset(name, (0, *trailing), dtype, maxshape=(None, *trailing))
        group = f.create_group(EPISODES_GROUP)
        for name, dtype in EPISODE_FIELDS.items():
            group.create_dataset(name, (0,), dtype, maxshape=(None,))
        self._file = f

    def _push_row(self, action: int, reward: float, terminal: bool) -> None:
        """Move the pending row into the buffer now that its action is known."""
        frame, ram, lives = self._pending
        ep = self._episode
        b = self._buffer
        b["frames"].append(frame)
        b["ram"].append(ram)
        b["lives"].append(lives)
        b["actions"].append(action)
        b["rewards"].append(reward)
        b["terminal"].append(terminal)
        b["episode_idx"].append(ep["episode_idx"])
        b["step_in_episode"].append(ep["steps"])
        self._pending = None
        if len(b["actions"]) >= FLUSH_EVERY:
            self._flush()

    def _flush(self) -> None:
        n = len(self._buffer["actions"])
        if n == 0:
            return
        start = self._rows_on_disk
        for name, (dtype, _) in ROW_FIELDS.items():
            ds = self._file[name]
            ds.resize(start + n, axis=0)
            ds[start:] = np.asarray(self._buffer[name], dtype=dtype)
            self._buffer[name].clear()

    def begin_episode(self, frame: np.ndarray, ram: np.ndarray, lives: int, seed: int,
                      noops: int, source: int, epsilon: float) -> None:
        """Start an episode with its first frame; may rotate to a new shard first."""
        if self._closed:
            raise RuntimeError("ShardWriter is closed")
        if self._episode is not None:
            raise RuntimeError("begin_episode called while an episode is already open")
        self._check_row(frame, ram)
        if self._local_counter >= EPISODE_UID_STRIDE:
            raise RuntimeError(f"worker {self.worker} exceeded {EPISODE_UID_STRIDE} episodes")
        if self._file is not None and self._rows_on_disk >= self.max_rows:
            self._file.close()
            self._file = None
        if self._file is None:
            self._open_shard()
        self._episode = {"start": self._rows_on_disk, "steps": 0, "return_": 0.0,
                         "episode_idx": self._file[EPISODES_GROUP]["start"].shape[0],
                         "seed": seed, "noops": noops, "source": source, "epsilon": epsilon}
        self._pending = (frame, ram, lives)

    def add_step(self, action: int, reward: float, next_frame: np.ndarray,
                 next_ram: np.ndarray, next_lives: int) -> None:
        """Record `action` at the current row and the row it produces (buffered)."""
        if self._episode is None:
            raise RuntimeError("add_step called without begin_episode")
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"action must be in 0..{NUM_ACTIONS - 1}, got {action}")
        self._check_row(next_frame, next_ram)
        self._push_row(action, reward, False)
        self._episode["steps"] += 1
        self._episode["return_"] += reward
        self._pending = (next_frame, next_ram, next_lives)

    def end_episode(self, end_reason: int, terminated: bool) -> None:
        """Finish the episode: last row gets action -1, and terminal if `terminated`."""
        if self._episode is None:
            raise RuntimeError("end_episode called without begin_episode")
        ep = self._episode
        self._push_row(LAST_ROW_ACTION, 0.0, bool(terminated))
        self._flush()
        values = {"start": ep["start"], "length": ep["steps"] + 1,
                  "episode_uid": self.worker * EPISODE_UID_STRIDE + self._local_counter,
                  "seed": ep["seed"], "noops": ep["noops"], "source": ep["source"],
                  "epsilon": ep["epsilon"], "end_reason": end_reason, "return_": ep["return_"]}
        group = self._file[EPISODES_GROUP]
        for name, value in values.items():
            ds = group[name]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = value
        self._file.flush()
        self._local_counter += 1
        self._episode = None

    def close(self) -> None:
        """Flush buffers and close the current shard; an unfinished episode is discarded with a warning."""
        if self._closed:
            return
        self._closed = True
        if self._file is None:
            return
        if self._episode is not None:
            start = self._episode["start"]
            warnings.warn(f"ShardWriter.close: discarding unfinished episode "
                          f"({self._episode['steps'] + 1} rows)", RuntimeWarning, stacklevel=2)
            for rows in self._buffer.values():
                rows.clear()
            for name in ROW_FIELDS:
                self._file[name].resize(start, axis=0)
            self._episode = self._pending = None
        self._file.close()
        self._file = None
