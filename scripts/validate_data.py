"""Check every ROADMAP 4.4 invariant across a dataset's shards and print statistics.

Usage: python scripts/validate_data.py <dataset_dir>
Exits non-zero on any violation; each violation is printed with its shard and row.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from wm.data.reader import list_shards
from wm.data.schema import (ACTION_MEANINGS, END_STOPPED, END_TERMINATED, END_TRUNCATED,
                            EPISODE_FIELDS, EPISODES_GROUP, FILE_ATTRS, FRAME_COMPRESSION,
                            LAST_ROW_ACTION, NUM_ACTIONS, ROW_FIELDS, SCHEMA_VERSION,
                            SOURCE_AGENT, SOURCE_HUMAN, SOURCE_RANDOM)

SOURCE_NAMES = {SOURCE_AGENT: "agent", SOURCE_RANDOM: "random", SOURCE_HUMAN: "human"}
END_REASONS = {END_TERMINATED, END_TRUNCATED, END_STOPPED}
MAX_PER_CHECK = 20  # rows listed per check and shard, so a broken shard doesn't flood the output


def _rows_violation(errors: list[str], shard: str, bad: np.ndarray, what: str) -> None:
    """Record one violation per bad row (capped at MAX_PER_CHECK)."""
    rows = np.flatnonzero(bad)
    for r in rows[:MAX_PER_CHECK]:
        errors.append(f"{shard}: row {r}: {what}")
    if len(rows) > MAX_PER_CHECK:
        errors.append(f"{shard}: ... {len(rows) - MAX_PER_CHECK} more rows: {what}")


def _check_schema(f: h5py.File, shard: str, errors: list[str]) -> bool:
    """Attrs, dataset presence, dtypes, and shapes. Returns False if content checks can't run."""
    for key in FILE_ATTRS:
        if key not in f.attrs:
            errors.append(f"{shard}: missing file attr {key!r}")
    if f.attrs.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{shard}: schema_version is {f.attrs.get('schema_version')!r}, "
                      f"expected {SCHEMA_VERSION}")
    ok = True
    n_rows = f["actions"].shape[0] if "actions" in f else None
    for name, (dtype, trailing) in ROW_FIELDS.items():
        if name not in f:
            errors.append(f"{shard}: missing dataset {name!r}")
            ok = False
            continue
        ds = f[name]
        if name == "frames":
            trailing = ds.shape[1:] if ds.ndim == 4 and ds.shape[3] == 3 else ("H", "W", 3)
            if ds.compression != FRAME_COMPRESSION or ds.chunks != (32, *ds.shape[1:]):
                errors.append(f"{shard}: frames must be lzf-compressed with chunks (32,H,W,3), "
                              f"got {ds.compression}, {ds.chunks}")
        if ds.dtype != dtype or ds.shape != (n_rows, *trailing):
            errors.append(f"{shard}: dataset {name!r} is {ds.dtype} {ds.shape}, "
                          f"expected {dtype} {(n_rows, *trailing)}")
            ok = False
    group = f.get(EPISODES_GROUP)
    n_eps = group["start"].shape[0] if group is not None and "start" in group else None
    for name, dtype in EPISODE_FIELDS.items():
        if group is None or name not in group:
            errors.append(f"{shard}: missing dataset {EPISODES_GROUP}/{name}")
            ok = False
        elif group[name].dtype != dtype or group[name].shape != (n_eps,):
            errors.append(f"{shard}: {EPISODES_GROUP}/{name} is {group[name].dtype} "
                          f"{group[name].shape}, expected {dtype} ({n_eps},)")
            ok = False
    return ok


def _check_content(rows: dict, eps: dict, shard: str, errors: list[str]) -> None:
    """Episode coverage and the per-row invariants that depend on it."""
    n = len(rows["actions"])
    start, length = eps["start"], eps["length"]
    expected_start = 0
    for e in range(len(start)):
        s, ln = int(start[e]), int(length[e])
        if ln < 1:
            errors.append(f"{shard}: episode {e}: length {ln} < 1")
        if s > expected_start:
            errors.append(f"{shard}: gap: rows {expected_start}..{s - 1} before episode {e} "
                          f"belong to no episode")
        elif s < expected_start:
            errors.append(f"{shard}: overlap: episode {e} starts at row {s}, "
                          f"previous episode ends at row {expected_start - 1}")
        expected_start = s + ln
    if expected_start != n:
        errors.append(f"{shard}: gap: episodes end at row {expected_start - 1}, shard has {n} rows")

    # Expected per-row values, built from episodes/ (rows outside every episode stay -1).
    episode_idx = np.full(n, -1, np.int64)
    step = np.full(n, -1, np.int64)
    is_last = np.zeros(n, bool)
    for e in range(len(start)):
        s, ln = max(int(start[e]), 0), int(length[e])
        end = min(s + ln, n)
        if end <= s:
            continue
        episode_idx[s:end] = e
        step[s:end] = np.arange(end - s) + (s - int(start[e]))
        if s + ln <= n:
            is_last[s + ln - 1] = True

    a = rows["actions"]
    checks = [
        (is_last & (a != LAST_ROW_ACTION), "last row of an episode: action != -1"),
        (~is_last & ((a < 0) | (a >= NUM_ACTIONS)),
         f"action outside 0..{NUM_ACTIONS - 1} (or -1 before the episode's last row)"),
        (~is_last & rows["terminal"], "terminal on a row that is not an episode's last"),
        (is_last & (rows["rewards"] != 0), "non-zero reward on an episode's last row"),
        (rows["episode_idx"] != episode_idx, "episode_idx disagrees with episodes/"),
        (rows["step_in_episode"] != step, "step_in_episode disagrees with episodes/"),
    ]
    for bad, what in checks:
        _rows_violation(errors, shard, bad, what)
    for e in np.flatnonzero(~np.isin(eps["source"], list(SOURCE_NAMES)))[:MAX_PER_CHECK]:
        errors.append(f"{shard}: episode {e}: unknown source {eps['source'][e]}")
    for e in np.flatnonzero(~np.isin(eps["end_reason"], list(END_REASONS)))[:MAX_PER_CHECK]:
        errors.append(f"{shard}: episode {e}: unknown end_reason {eps['end_reason'][e]}")


def validate(dataset_dir: Path) -> tuple[list[str], dict]:
    """All violations in a dataset, and its statistics."""
    errors: list[str] = []
    shards = list_shards(dataset_dir)
    if not shards:
        return [f"{dataset_dir}: no shard_w*_*.h5 files"], {}
    stats = {"rows": 0, "episodes": 0, "frame_bytes": 0, "file_bytes": 0, "lengths": [],
             "actions": np.zeros(NUM_ACTIONS, np.int64),
             "by_source": {k: [0, 0] for k in SOURCE_NAMES}}
    uid_owner: dict[int, str] = {}
    frame_shapes = set()
    for path in shards:
        shard = path.name
        stats["file_bytes"] += path.stat().st_size
        try:
            f = h5py.File(path, "r")
        except OSError as exc:
            errors.append(f"{shard}: cannot open: {exc}")
            continue
        with f:
            if not _check_schema(f, shard, errors):
                continue
            frames = f["frames"]
            frame_shapes.add(frames.shape[1:])
            rows = {name: f[name][()] for name in ROW_FIELDS if name not in ("frames", "ram")}
            eps = {name: f[EPISODES_GROUP][name][()] for name in EPISODE_FIELDS}
        _check_content(rows, eps, shard, errors)
        for e, uid in enumerate(eps["episode_uid"].tolist()):
            if uid in uid_owner:
                errors.append(f"{shard}: episode {e}: episode_uid {uid} "
                              f"already used in {uid_owner[uid]}")
            uid_owner.setdefault(uid, f"{shard} episode {e}")
        n = len(rows["actions"])
        stats["rows"] += n
        stats["episodes"] += len(eps["start"])
        stats["frame_bytes"] += n * int(np.prod(frames.shape[1:]))
        stats["lengths"].extend(eps["length"].tolist())
        valid = rows["actions"][(rows["actions"] >= 0) & (rows["actions"] < NUM_ACTIONS)]
        stats["actions"] += np.bincount(valid, minlength=NUM_ACTIONS)
        for src, ln in zip(eps["source"].tolist(), eps["length"].tolist()):
            if src in stats["by_source"]:
                stats["by_source"][src][0] += 1
                stats["by_source"][src][1] += ln
    if len(frame_shapes) > 1:
        errors.append(f"{dataset_dir}: shards have different frame shapes {sorted(frame_shapes)}")
    return errors, stats


def print_stats(stats: dict) -> None:
    """Print the dataset summary."""
    print(f"rows: {stats['rows']}")
    print(f"episodes: {stats['episodes']}")
    for src, (n_eps, n_rows) in stats["by_source"].items():
        print(f"  {SOURCE_NAMES[src]}: {n_eps} episodes, {n_rows} rows")
    mean_len = np.mean(stats["lengths"]) if stats["lengths"] else 0.0
    print(f"mean episode length: {mean_len:.1f} rows")
    total = max(int(stats["actions"].sum()), 1)
    print("action histogram:")
    for name, count in zip(ACTION_MEANINGS, stats["actions"].tolist()):
        print(f"  {name:<10} {count:>10}  {100 * count / total:5.1f}%")
    print(f"disk size: {stats['file_bytes'] / 2**20:.2f} MiB")
    ratio = stats["frame_bytes"] / max(stats["file_bytes"], 1)
    print(f"compression ratio (uncompressed frame bytes / file bytes): {ratio:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    errors, stats = validate(args.dataset_dir)
    if stats:
        print_stats(stats)
    for err in errors:
        print(f"VIOLATION {err}")
    print(f"{len(errors)} violation(s)" if errors else "OK: all invariants hold")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
