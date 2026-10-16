# cp1-B: HDF5 storage
## Summary
- `wm/data/writer.py`: `ShardWriter` as in 4.4. It writes resizable per-row and `episodes/` datasets with the 4.4 dtypes, buffers rows and flushes every 1024 rows and on `end_episode`, and rotates shards only between episodes. It sets `schema_version`, `git_commit`, `created_utc`, and the ale-py/gymnasium versions. It works as a context manager.
- `wm/data/reader.py`: `list_shards` and `ShardReader` (lazy, read-only, picklable, cached `attrs` and `episodes`, `read` with a slice or a sorted index array).
- `scripts/validate_data.py <dataset_dir>` checks every 4.4 invariant across all shards and prints the dataset statistics. It exits 1 on any violation and prints each one with its shard and row.
- `tests/fixtures.py::make_fake_shard` is implemented with `ShardWriter`. New helpers `encode_frame` and `decode_frame` are appended at the end of the file.
- No dependency on `wm.env` or CP1-A/CP1-C code (a test checks this).

## Files
Modified (Phase 0 stubs replaced): `wm/data/writer.py`, `wm/data/reader.py`, `tests/fixtures.py` (only `make_fake_shard`, its imports, and the two appended helpers).
Created: `scripts/validate_data.py`, `tests/test_storage.py`, `progress/cp1-B.md`.

## Tests
`tests/test_storage.py`, 35 tests. Episode lengths are in **rows** throughout (T steps → T+1 rows):
- Round trip of 3 episodes with [5, 1, 40] rows (the 1-row episode has 0 steps). Every per-row field, every `episodes/` entry, and the dtypes, `maxshape`, chunks, and lzf are checked. Frames are 6×5, non-square, to catch H/W swaps.
- Attrs: all of `FILE_ATTRS` are present, `collector_config` is stored as JSON, caller-given versions are kept, a missing caller attr raises, and `_git_commit` returns "unknown" outside a repo.
- Rotation with `max_rows=10` and episodes of [12, 3, 15, 4, 4] rows → 3 shards holding [12], [3, 15], [4, 4]. `episode_idx` restarts per shard, and `episode_uid` = 3·10⁷ + 0..4 increases across shards (worker 3).
- Flush boundary: episodes of 1024, 1025, 2500, and 1 rows round-trip exactly.
- Errors: `add_step`/`end_episode` without `begin_episode`, `begin_episode` twice, a wrong frame shape or dtype, an out-of-range action, wrong RAM size, `begin_episode` after `close`, and refusing to overwrite an existing shard.
- `close()` with an open episode of 1500 rows (partly flushed) warns, discards it, and leaves a file that validates.
- Reader: index arrays (including duplicates) match slices, unsorted arrays raise, non-row names raise, the file opens read-only, it survives pickling before and after first access, and `list_shards` sorts.
- `make_fake_shard`: decoding the pixels recovers (row, episode_idx) for 324 rows, so G is non-zero. Actions cover all of 0..5, rewards, lives, and RAM vary, output is deterministic, and bad lengths are rejected.
- Validator (run as a subprocess): passes on a clean shard. Exits non-zero and names the shard and row for each of these:
  - a misplaced -1
  - a missing -1 on a last row
  - action 6
  - a misplaced terminal
  - a gap
  - an overlap
  - a wrong `step_in_episode`
  - a wrong `episode_idx`
  - a reward on a last row
  - wrong `schema_version`
  - wrong dtype and length
  - duplicate `episode_uid` across shards
  - an empty directory
- `test_write_speed_84` (`slow`): 20,000 rows at 84×84.

Commands (Python 3.13.2, h5py 3.16.0 / HDF5 2.0.0, numpy 2.5.3, macOS arm64, CPU):
- `pytest -m "not slow and not gpu and not network"` → 60 passed in about 4 s
- `pytest -m slow tests/test_storage.py -s` → 1 passed
- `pytest -m "slow or network or env"` → 5 passed

## Results
- **Write speed: 54,886 rows/s at 84×84** (target ≥ 2,000). Frames are mostly black with sprite-like blocks, and the test includes lzf compression and the per-episode `flush()`.
- `validate_data.py` on an 84×84 fake shard of [5, 1, 40] rows reports a compression ratio of 5.79. Those frames are flat colour, so real frames will compress differently. Measure it on the CP1 smoke shard.

## Deviations
- **`close()` with an open episode warns (`RuntimeWarning`) and discards that episode.** Rows already flushed are truncated away, so the shard stays valid. I chose a warning over raising so that `with ShardWriter(...)` doesn't hide the real exception when a collector crashes mid-episode. Callers who want to keep a stopped human episode must call `end_episode(END_STOPPED, False)` first.
- **`make_fake_shard(path, ...)`: `path` is the output directory.** It writes `shard_w000_00000.h5` there (worker 0, a single shard) and returns that `Path`. `lengths` are in rows and must have `n_episodes` entries, each ≥ 1. The signature is unchanged.
- **Fake data formulas** (for CP2-B alignment tests):
  - action = (5·row + e) % 6, so neighbouring rows always differ
  - reward = 5·(row % 4); the last row is still -1 / 0
  - lives = 3 − min(step // 10, 3)
  - ram[j] = (row + j) % 256
  - source = e % 3, seed = 1000 + e, noops = e % 31
  - episodes with even e end terminated, odd e truncated
  - With `encode_rows=False`, frames are seeded random noise.
- **Added `encode_frame` and `decode_frame` at the end of `tests/fixtures.py`.** `decode_frame` takes a uint8 (H,W,3) frame and returns (row % 65536, episode_idx % 256).
- **The writer requires the caller to pass** `env_id, frameskip, maxpool, resolution, resize_method, collector_config`, and raises `ValueError` if any is missing. A dict value (e.g. `collector_config`) is stored as a JSON string. `schema_version`, `git_commit`, and `created_utc` are always set by the writer and override caller values. The ale-py and gymnasium versions come from `importlib.metadata` unless the caller passes them.
- **`git_commit` is taken in the directory of `writer.py`,** so it records the code version wherever the script runs from. `created_utc` is ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`), set per shard.
- **Shards open with mode `"w-"`:** the writer raises `FileExistsError` rather than overwrite an existing shard. A shard file is created at the first `begin_episode`, so a writer that records nothing leaves no file.
- **The writer validates its inputs:** frames must be uint8 of `frame_shape`, RAM uint8 (128,), and actions 0..5. Anything else raises `ValueError`.
- **`ShardReader.read` accepts sorted arrays with duplicates.** It reads unique rows and expands them, since h5py needs strictly increasing indices. Unsorted arrays raise `ValueError`. `ShardReader.close()` was added.
- **Validator checks beyond the 4.4 list:**
  - reward 0 on last rows (from the table)
  - known `source` / `end_reason` values
  - frames lzf with chunks (32,H,W,3)
  - all `FILE_ATTRS` present
  - one frame shape across shards
- **Validator output is capped:** it lists up to 20 rows per check per shard, then a "… N more rows" line.
- **`validate_data.py` takes a positional `dataset_dir`, with no `--config`/`--set`** (same reasoning as `play_real.py` in cp0). It has no parameters to configure.

## Interface change requests
None. `wm/data/schema.py` had everything needed.

## Follow-ups
- **CP1 integration:** run `validate_data.py` on the smoke shard and record its real compression ratio and write speed. Real `collect.py` rates will be bounded by the env and the agent, not the writer.
- **CP2-B:** `ShardReader.read` with a scattered index array uses h5py fancy indexing, which is slow over chunked lzf frames. For random-access windows, prefer reading a contiguous slice `i-K..i` per item (windows are contiguous) or `in_memory=True`.
- The validator reads all 1-D per-row datasets of a shard into memory (about 1.4 MB per 100k rows, without frames or RAM). That is fine at the planned scale.
