# cp2-B: PyTorch window dataset and loader
## Summary
- `wm/data/dataset.py`: `WindowDataset` as in 4.5. It precomputes the valid target rows per shard (int64) and maps a global index to a shard through cumulative offsets. Each item reads **one contiguous slice** of frames `i-K..i` and one of actions `i-K..i-1` (no per-row or fancy indexing). The parent process never keeps an HDF5 handle open.
- Optional `in_memory=True` loads each shard's frames and actions once, as torch uint8 tensors in shared memory, so spawn workers map them rather than receiving copies. Outputs are identical to the lazy path.
- `DataConfig` dataclass and `make_dataloader(cfg, split)`:
  - Train: shuffled with a seeded generator, `drop_last=True`.
  - Val: in order, keeps the last batch.
  - `persistent_workers` when `num_workers > 0`; `pin_memory` when CUDA is available.
  - Module-level `_init_worker` (through `functools.partial`) seeds `random`/numpy/torch with `seed + worker_id` and closes any HDF5 handle inherited through fork.
- `scripts/bench_loader.py` measures samples/s for `num_workers ∈ {0,2,4,8}` × `in_memory ∈ {False, True}` on any dataset dir. If the dir is empty, it generates a synthetic 100k-row dataset there first.
- **Target met:** ≥ 2,000 samples/s at K=4, 84×84 with 2 or more workers lazily (2,295 on real frames), or with any worker count in memory.

## Files
Created: `scripts/bench_loader.py`, `tests/test_dataset.py`, `progress/cp2-B.md`.
Modified: `wm/data/dataset.py` (Phase 0 stub replaced).
Not touched: `wm/config.py` and `tests/fixtures.py` (no new fixtures needed).

## Tests
`tests/test_dataset.py`, 26 tests, all on `make_fake_shard`. Frames are 6×5 (non-square), and episode lengths are [5, 1, 40, 3, 4, 300, 12, 2, 60, 9] rows, including episodes shorter than K:
- **Exact alignment** (K = 1, 4, 8; every item of train and val): the decoded context rows are `i-K..i-1`, oldest first, all from the target's episode. The actions equal `actions[i-K..i-1]` read from the shard, and the decoded target row is `i`. A mutation check confirmed these tests fail when the action slice or the first valid target is shifted by one.
- No action is -1 or `NULL_ACTION` (6) in any window, for K = 1, 4, 8.
- Window count is Σ max(0, length − K), and the set of targets equals every `start+K .. start+length-1`, so no window crosses an episode start. K larger than every episode gives an empty dataset.
- Split:
  - train and val share no episode, and val membership follows the crc32 rule
  - the split is deterministic and independent of `seed`
  - `val_fraction` 0 and 1 behave at the extremes
  - an unknown split name raises
- Multiple dirs: `shard` indexes the sorted global shard list, and rows decode within that shard. Mismatched frame shapes across shards raise; no shards raises.
- Format: dtypes and shapes (`context` (K,C,H,W) float32, `actions` (K,) int64, `target` (C,H,W) float32, `row`/`shard` int64 scalars). Range in [-1,1] on full-range random pixels, for RGB and greyscale (C=1). The RGB values are exactly `uint8/127.5 - 1`, and greyscale is float32 luma `/127.5 - 1`.
- `in_memory=True` equals the lazy path item by item (both splits, RGB and grey).
- No HDF5 handle is open after `__init__` or after pickling.
- Loaders:
  - train drops the last batch, shuffles, and repeats with the same seed
  - val is in order
  - `num_workers=2` (spawn, macOS) returns the same batches as `num_workers=0`
  - `num_workers=2` with `in_memory=True` works

Commands (Python 3.13.2, torch 2.14.1, h5py 3.16.0, macOS arm64, 10 CPUs, CPU only):
- `pytest tests/test_dataset.py` → 26 passed in about 14 s (most of that is spawn worker start-up).
- `pytest -m "not slow and not gpu and not network"` → 111 passed in 18 s.
- This workflow has no `slow`, `gpu`, or `network` tests.

## Results
`python scripts/bench_loader.py`: K=4, batch 64, shuffled train split, 200 timed batches after the first. "Startup" is loader construction plus the first batch (worker spawn, and the RAM load when `in_memory`).

Real frames, `data/bench_real`: 30 episodes, 28,241 rows, 1 shard, 38 MiB, compression 14.9. Collected with `scripts/smoke_cp1.py --set out_dir=data/bench_real --set episodes=30 --set max_steps=27000 --set seed=100`; the validator passes.

| in_memory | workers | startup s | samples/s |
|---|---:|---:|---:|
| False | 0 | 0.1 | 1,130 |
| False | 2 | 1.3 | 2,295 |
| False | 4 | 2.4 | 3,835 |
| False | 8 | 4.6 | 5,964 |
| True | 0 | 0.8 | 8,571 |
| True | 2 | 1.8 | 15,406 |
| True | 4 | 2.4 | 20,315 |
| True | 8 | 4.5 | 17,375 |

Synthetic, `data/bench_synthetic` (default run): 100,000 rows, 4 shards, 82 MiB, compression about 26, episodes of 300–2,000 rows:

| in_memory | workers | startup s | samples/s |
|---|---:|---:|---:|
| False | 0 | 0.1 | 1,013 |
| False | 2 | 1.3 | 2,056 |
| False | 4 | 2.3 | 3,599 |
| False | 8 | 4.4 | 5,435 |
| True | 0 | 2.7 | 7,767 |
| True | 2 | 3.6 | 11,885 |
| True | 4 | 4.4 | 16,241 |
| True | 8 | 6.3 | 16,972 |

The lazy path is bound by LZF decompression: a random window touches one or two 32-row frame chunks (about 0.7 MB each), so throughput scales close to linearly with workers.

## Deviations
- **`DataConfig` lives in `wm/data/dataset.py`, not `wm/config.py`.** No data or train config dataclass existed, and `make_dataloader(cfg: Any, split)` was left open in cp0. Fields: `dataset_dirs, K, batch_size=64, num_workers=4, grayscale=False, val_fraction=0.05, seed=0, in_memory=False`. `make_dataloader` only reads these attributes, so CP3-B's trainer config can nest a `DataConfig` or pass any object with the same fields. `wm/config.py` is unchanged.
- **`in_memory` is an extra keyword-only argument** of `WindowDataset` (`*, in_memory=False`). The 4.5 positional signature is unchanged, and CP2-B task 3 asks for the option.
- **`seed` is stored but not used by `WindowDataset`.** The split is the crc32 hash in 4.5 and doesn't depend on it. The shuffle and worker seeds come from `cfg.seed` in `make_dataloader`.
- **Shard order:** `shard_paths` is the global list of shards from all `dataset_dirs`, sorted by full path. `shard` in each item indexes it.
- **Greyscale** is float32 luma on the uint8 values, then `/127.5 - 1`, with no rounding back to uint8. This is my reading of "luma applied to uint8 before normalizing"; if the spec meant rounding to uint8, that would be a one-line change.
- **In-memory loads whole shards** (all rows, not only the split's rows), skipping shards with no targets in the split. A train and a val dataset therefore each hold their own copy.
- **Validation in `__init__`:** `K >= 1`, the split must be "train" or "val", there must be at least one shard, and all shards must have the same frame shape. Out-of-range indices raise `IndexError`; negative indices are not supported.
- **`bench_loader.py` takes `--config` / `--set`.** The dataset dir is `--set dataset_dir=...`. Synthetic generation runs only when `synthetic=true` (the default) and the dir has no shards.

## Interface change requests
None. `in_memory` is an additive keyword argument; the integrator may want to add it to 4.5.

## Follow-ups
- **CP2 integration:** run `python scripts/bench_loader.py --set dataset_dir=data/debug --set synthetic=false` and record the numbers.
- **CP3-B:**
  - Use the lazy loader with 4–8 workers (3.8k–6k samples/s here), unless profiling shows the GPU waiting on data.
  - `in_memory` on the 500k-step debug set needs about 10.6 GB of uint8 frames, plus another copy for the val loader, so it only fits on a cluster node with enough RAM.
  - With `persistent_workers`, build each loader once per run.
- On Linux (fork), `_init_worker` closes any HDF5 handle the parent opened before forking, for example after a `num_workers=0` val pass in the main process.
- `data/bench_synthetic/` and `data/bench_real/` are local benchmark data (gitignored); delete them when not needed.
