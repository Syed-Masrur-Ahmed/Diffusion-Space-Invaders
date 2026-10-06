"""Training windows over HDF5 shards (ROADMAP 4.5, [FROZEN]). Implemented by CP2-B."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    """K context frames and actions plus the target frame, split by episode (4.5)."""

    def __init__(self, dataset_dirs: list[str | Path], K: int, split: str, grayscale: bool = False,
                 val_fraction: float = 0.05, seed: int = 0) -> None:
        raise NotImplementedError("CP2-B")

    def __len__(self) -> int:
        raise NotImplementedError("CP2-B")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Dict with `context` (K,C,H,W), `actions` (K,), `target` (C,H,W), `row`, `shard`."""
        raise NotImplementedError("CP2-B")


def make_dataloader(cfg: Any, split: str) -> DataLoader:
    """DataLoader over a WindowDataset with seeded workers that open HDF5 lazily."""
    raise NotImplementedError("CP2-B")
