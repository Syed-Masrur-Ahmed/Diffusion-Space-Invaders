"""Benchmark the training DataLoader in samples/s for several num_workers, lazy and in-memory.

Target (CP2-B): >= 2,000 samples/s at K=4, 84x84. Uses the shuffled train split, so reads are
random access as in training. The first batch (worker start-up) is not timed.

    python scripts/bench_loader.py                                   # synthetic data, made once
    python scripts/bench_loader.py --set dataset_dir=data/debug --set synthetic=false

With `synthetic=true` and no shards in `dataset_dir`, it first writes a synthetic dataset
there: `rows` rows of 84x84 frames that look roughly like the game (black background, an
alien grid that moves, shields, a player), in episodes of 300-2,000 rows over several shards.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from wm.config import load_config
from wm.data.dataset import DataConfig, make_dataloader
from wm.data.reader import list_shards
from wm.data.schema import END_TERMINATED, NUM_ACTIONS, RAM_SIZE, SOURCE_AGENT
from wm.data.writer import ShardWriter

DEFAULTS = {
    "dataset_dir": "data/bench_synthetic",
    "synthetic": True,
    "rows": 100_000,
    "shard_rows": 25_000,
    "size": 84,
    "K": 4,
    "batch_size": 64,
    "batches": 200,
    "num_workers": [0, 2, 4, 8],
    "in_memory": [False, True],
    "val_fraction": 0.05,
    "seed": 0,
}


def synthetic_frame(rng: np.random.Generator, step: int, size: int) -> np.ndarray:
    """A sparse game-like frame: alien grid (moves with `step`), three shields, player, a laser."""
    f = np.zeros((size, size, 3), np.uint8)
    u = size // 21
    dx = (step // 8) % (4 * u)
    for r in range(5):
        for c in range(6):
            y, x = 2 * u + 2 * r * u, u + dx + 2 * c * u + u
            f[y:y + u, x:x + u] = (134, 134, 29)
    for c in range(3):
        x = 3 * u + 6 * c * u
        f[16 * u:17 * u + 1, x:x + 2 * u] = (181, 83, 40)
    px = int(rng.integers(0, size - 2 * u))
    f[19 * u:20 * u, px:px + u + 1] = (50, 132, 50)
    f[int(rng.integers(5 * u, 15 * u)), int(rng.integers(0, size))] = (142, 142, 142)
    return f


def make_synthetic(out_dir: Path, rows: int, shard_rows: int, size: int, seed: int) -> None:
    """Write `rows` rows of synthetic episodes with ShardWriter."""
    rng = np.random.default_rng(seed)
    attrs = {"env_id": "synthetic", "frameskip": 4, "maxpool": True, "resolution": size,
             "resize_method": "max", "collector_config": {"bench_loader": "synthetic"}}
    ram = np.zeros(RAM_SIZE, np.uint8)
    written = 0
    with ShardWriter(out_dir, 0, (size, size, 3), attrs, max_rows=shard_rows) as w:
        e = 0
        while written < rows:
            n = int(min(rng.integers(300, 2001), max(rows - written, 1)))
            w.begin_episode(synthetic_frame(rng, 0, size), ram, 3, e, 0, SOURCE_AGENT, 0.2)
            for step in range(1, n):
                w.add_step(int(rng.integers(NUM_ACTIONS)), 0.0, synthetic_frame(rng, step, size), ram, 3)
            w.end_episode(END_TERMINATED, True)
            written += n
            e += 1


def bench(cfg: dict, num_workers: int, in_memory: bool) -> tuple[float, float]:
    """(seconds to build the loader and get the first batch, timed samples/s)."""
    data_cfg = DataConfig(dataset_dirs=[cfg["dataset_dir"]], K=int(cfg["K"]),
                          batch_size=int(cfg["batch_size"]), num_workers=num_workers,
                          val_fraction=float(cfg["val_fraction"]), seed=int(cfg["seed"]),
                          in_memory=in_memory)
    t0 = time.perf_counter()
    loader = make_dataloader(data_cfg, "train")
    it = iter(loader)
    next(it)
    startup = time.perf_counter() - t0
    n_batches = min(int(cfg["batches"]), len(loader) - 1)
    t0 = time.perf_counter()
    samples = sum(next(it)["target"].shape[0] for _ in range(n_batches))
    rate = samples / (time.perf_counter() - t0)
    del it, loader  # shut persistent workers down before the next configuration
    return startup, rate


def resolve_config(argv: list[str] | None = None) -> dict:
    """Merge DEFAULTS with an optional YAML file and `--set key=value` overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config is not None:
        cfg.update(load_config(args.config))
    for item in args.overrides:
        key, sep, raw = item.partition("=")
        if not sep or key not in cfg:
            raise KeyError(f"bad override {item!r}; keys: {sorted(cfg)}")
        cfg[key] = yaml.safe_load(raw)
    return cfg


def main(argv: list[str] | None = None) -> None:
    cfg = resolve_config(argv)
    out_dir = Path(cfg["dataset_dir"])
    if not list_shards(out_dir):
        if not cfg["synthetic"]:
            raise FileNotFoundError(f"no shards in {out_dir}")
        t0 = time.perf_counter()
        make_synthetic(out_dir, int(cfg["rows"]), int(cfg["shard_rows"]), int(cfg["size"]), int(cfg["seed"]))
        print(f"wrote synthetic dataset to {out_dir} in {time.perf_counter() - t0:.1f} s")
    shards = list_shards(out_dir)
    disk = sum(p.stat().st_size for p in shards) / 2**20
    print(f"{out_dir}: {len(shards)} shards, {disk:.0f} MiB; K={cfg['K']} batch_size={cfg['batch_size']}")
    print(f"{'in_memory':>9} {'workers':>7} {'startup s':>9} {'samples/s':>10}")
    for in_memory in cfg["in_memory"]:
        for nw in cfg["num_workers"]:
            startup, rate = bench(cfg, int(nw), bool(in_memory))
            print(f"{str(bool(in_memory)):>9} {nw:>7} {startup:9.1f} {rate:10.0f}", flush=True)


if __name__ == "__main__":
    main()
