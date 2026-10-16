"""Training windows over HDF5 shards (ROADMAP 4.5, [FROZEN]). Implemented by CP2-B."""

import functools
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from wm.data.reader import ShardReader, list_shards

SPLITS = ("train", "val")
LUMA = torch.tensor([0.299, 0.587, 0.114])


@dataclass
class DataConfig:
    """Fields `make_dataloader` reads; the trainer config nests or duck-types these."""

    dataset_dirs: list[str]
    K: int
    batch_size: int = 64
    num_workers: int = 4
    grayscale: bool = False
    val_fraction: float = 0.05
    seed: int = 0
    in_memory: bool = False


def is_val_episode(shard_name: str, episode_uid: int, val_fraction: float) -> bool:
    """Episode-level split rule from 4.5."""
    return zlib.crc32(f"{shard_name}:{episode_uid}".encode()) % 10_000 < val_fraction * 10_000


class WindowDataset(Dataset):
    """K context frames and actions plus the target frame, split by episode (4.5)."""

    def __init__(self, dataset_dirs: list[str | Path], K: int, split: str, grayscale: bool = False,
                 val_fraction: float = 0.05, seed: int = 0, *, in_memory: bool = False) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K, self.split, self.grayscale, self.seed = K, split, grayscale, seed
        self.shard_paths = sorted(p for d in dataset_dirs for p in list_shards(d))
        if not self.shard_paths:
            raise FileNotFoundError(f"no shards in {list(map(str, dataset_dirs))}")
        self._readers = [ShardReader(p) for p in self.shard_paths]

        want_val = split == "val"
        self._targets: list[np.ndarray] = []  # valid target rows per shard, int64
        frame_shapes = set()
        for path, reader in zip(self.shard_paths, self._readers):
            eps = reader.episodes
            frame_shapes.add(reader.read("frames", slice(0, 0)).shape[1:])
            rows = [np.arange(s + K, s + n, dtype=np.int64)
                    for s, n, uid in zip(eps["start"], eps["length"], eps["episode_uid"])
                    if n > K and is_val_episode(path.name, int(uid), val_fraction) == want_val]
            self._targets.append(np.concatenate(rows) if rows else np.zeros(0, np.int64))
            reader.close()  # no HDF5 handle may stay open in the parent (pitfall 5)
        if len(frame_shapes) != 1:
            raise ValueError(f"shards have different frame shapes: {sorted(frame_shapes)}")
        self._offsets = np.cumsum([0] + [len(t) for t in self._targets])

        # Tensors in shared memory, so spawn workers map them instead of each getting a copy.
        self._frames: list[torch.Tensor | None] = [None] * len(self._readers)
        self._actions: list[torch.Tensor | None] = [None] * len(self._readers)
        if in_memory:
            for s, reader in enumerate(self._readers):
                if len(self._targets[s]):
                    self._frames[s] = torch.from_numpy(reader.read("frames", slice(None))).share_memory_()
                    self._actions[s] = torch.from_numpy(reader.read("actions", slice(None))).share_memory_()
                reader.close()

    def close(self) -> None:
        """Close every shard handle; the next access reopens it in the calling process."""
        for reader in self._readers:
            reader.close()

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _window(self, s: int, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Frames i-K..i (uint8, (K+1,H,W,3)) and actions i-K..i-1, each from one contiguous slice."""
        if self._frames[s] is not None:
            return self._frames[s][i - self.K:i + 1], self._actions[s][i - self.K:i]
        reader = self._readers[s]
        return (torch.from_numpy(reader.read("frames", slice(i - self.K, i + 1))),
                torch.from_numpy(reader.read("actions", slice(i - self.K, i))))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Dict with `context` (K,C,H,W), `actions` (K,), `target` (C,H,W), `row`, `shard`."""
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        s = int(np.searchsorted(self._offsets, idx, side="right")) - 1
        i = int(self._targets[s][idx - self._offsets[s]])
        frames, actions = self._window(s, i)
        x = frames.permute(0, 3, 1, 2).float()  # float before any arithmetic on uint8
        if self.grayscale:
            x = (x * LUMA[:, None, None]).sum(1, keepdim=True)
        x = x / 127.5 - 1
        return {"context": x[:-1], "actions": actions.long(), "target": x[-1],
                "row": torch.tensor(i, dtype=torch.int64), "shard": torch.tensor(s, dtype=torch.int64)}


def _init_worker(seed: int, worker_id: int) -> None:
    """Seed this worker with seed + worker_id and drop HDF5 handles inherited through fork."""
    s = seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    get_worker_info().dataset.close()


def make_dataloader(cfg: DataConfig, split: str) -> DataLoader:
    """DataLoader over a WindowDataset with seeded workers that open HDF5 lazily."""
    dataset = WindowDataset(cfg.dataset_dirs, cfg.K, split, grayscale=cfg.grayscale,
                            val_fraction=cfg.val_fraction, seed=cfg.seed, in_memory=cfg.in_memory)
    train = split == "train"
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=functools.partial(_init_worker, cfg.seed),
        generator=torch.Generator().manual_seed(cfg.seed),
    )
