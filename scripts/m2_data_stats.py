"""Milestone 2 data section: split sizes, action and source counts per split, and leakage checks.

    python scripts/m2_data_stats.py --config configs/train/debug.yaml \\
        --set data.root=<scratch>/data --set 'data.dirs=[full]' --holdout eval test --out results/milestone2/data

Splits: `train` and `val` are the training dirs split by episode exactly as the trainer does
(4.5); every `--holdout` dir (under data.root) is its own split. Leakage checks between splits:
shared episode seeds, frames that appear byte-identical in train, and identical windows (K+1 frames
and K actions: the exact input-and-target the model sees). Writes <out>/stats.json.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from wm.data.dataset import is_val_episode
from wm.data.reader import ShardReader, list_shards
from wm.data.schema import ACTION_MEANINGS, SOURCE_AGENT, SOURCE_HUMAN, SOURCE_RANDOM
from wm.train.trainer import load_run_config

SOURCES = {SOURCE_AGENT: "agent", SOURCE_RANDOM: "random", SOURCE_HUMAN: "human"}
CHUNK = 8192
_RNG = np.random.default_rng(12345)


def frame_hashes(frames: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """64-bit hash per frame (uint8 (n,H,W,C)): a random-weight sum over the bytes, mod 2^64."""
    flat = frames.reshape(len(frames), -1)
    pad = (-flat.shape[1]) % 8
    words = np.pad(flat, ((0, 0), (0, pad))).view(np.uint64)
    with np.errstate(over="ignore"):
        return (words * weights[: words.shape[1]]).sum(axis=1, dtype=np.uint64)


def window_hashes(h: np.ndarray, actions: np.ndarray, K: int, mix: np.ndarray) -> np.ndarray:
    """Hash of each window ending at row t = K..n-1 of one episode: frames t-K..t and actions t-K..t-1."""
    n = len(h)
    if n <= K:
        return np.zeros(0, np.uint64)
    acts = actions.astype(np.uint64)
    with np.errstate(over="ignore"):
        out = np.zeros(n - K, np.uint64)
        for j in range(K + 1):
            out += h[j:n - K + j] * mix[j]
        for j in range(K):
            out += (acts[j:n - K + j] + np.uint64(1)) * mix[K + 1 + j]
    return out


def collect(dirs: list[Path], K: int, val_fraction: float | None, weights: np.ndarray, mix: np.ndarray) -> dict:
    """Per split ('train'/'val', or 'all' when val_fraction is None): counts, seeds, frame and window hashes."""
    splits: dict[str, dict] = {}
    for d in dirs:
        for path in list_shards(d):
            reader = ShardReader(path)
            eps = reader.episodes
            frames_h = np.concatenate([frame_hashes(reader.read("frames", slice(s, min(s + CHUNK, reader.num_rows))),
                                                    weights) for s in range(0, reader.num_rows, CHUNK)])
            actions = reader.read("actions", slice(None))
            for e in range(len(eps["start"])):
                uid = int(eps["episode_uid"][e])
                name = "all" if val_fraction is None else (
                    "val" if is_val_episode(path.name, uid, val_fraction) else "train")
                sp = splits.setdefault(name, {"episodes": 0, "rows": 0, "windows": 0, "actions": Counter(),
                                              "sources": Counter(), "seeds": [], "frames": [], "win": []})
                rows = reader.episode_rows(e)
                a = actions[rows][:-1]  # the last row's action is LAST_ROW_ACTION, not a real action
                sp["episodes"] += 1
                sp["rows"] += rows.stop - rows.start
                sp["windows"] += max(0, rows.stop - rows.start - K)
                sp["actions"].update(a.tolist())
                sp["sources"][SOURCES.get(int(eps["source"][e]), str(int(eps["source"][e])))] += 1
                sp["seeds"].append(int(eps["seed"][e]))
                sp["frames"].append(frames_h[rows])
                sp["win"].append(window_hashes(frames_h[rows], actions[rows][:-1], K, mix))
            reader.close()
    for sp in splits.values():
        for k in ("frames", "win"):
            sp[k] = np.concatenate(sp[k]) if sp[k] else np.zeros(0, np.uint64)
        sp["seeds"] = np.asarray(sp["seeds"], np.int64)
    return splits


def summarize(splits: dict[str, dict]) -> dict:
    """JSON-ready counts per split, plus leakage of every non-train split against train."""
    out = {"splits": {}, "leakage": {}}
    for name, sp in splits.items():
        n_act = sum(sp["actions"].values())
        out["splits"][name] = {
            "episodes": sp["episodes"], "rows": sp["rows"], "windows": sp["windows"],
            "sources": dict(sp["sources"]),
            "actions": {ACTION_MEANINGS[a]: sp["actions"][a] for a in range(len(ACTION_MEANINGS))},
            "action_share": {ACTION_MEANINGS[a]: sp["actions"][a] / max(n_act, 1) for a in range(len(ACTION_MEANINGS))},
            "seed_range": [int(sp["seeds"].min()), int(sp["seeds"].max())] if len(sp["seeds"]) else None,
        }
    train = splits.get("train")
    for name, sp in splits.items():
        if name == "train" or train is None:
            continue
        frames_in_train = np.isin(sp["frames"], train["frames"])
        win_in_train = np.isin(sp["win"], train["win"])
        out["leakage"][name] = {
            "shared_seeds": int(np.isin(sp["seeds"], train["seeds"]).sum()),
            "frames": int(len(sp["frames"])), "frames_identical_to_a_train_frame": int(frames_in_train.sum()),
            "windows": int(len(sp["win"])), "windows_identical_to_a_train_window": int(win_in_train.sum()),
            "unique_frames_in_split": int(len(np.unique(sp["frames"]))),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/train/debug.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--holdout", nargs="*", default=[], help="dirs under data.root, each its own split")
    parser.add_argument("--out", type=Path, default=Path("results/milestone2/data"))
    args = parser.parse_args(argv)
    cfg = load_run_config(args.config, args.overrides)
    root, K = Path(cfg.data.root), cfg.data.K
    weights = _RNG.integers(1, 2**63, size=4096, dtype=np.uint64) | np.uint64(1)
    mix = _RNG.integers(1, 2**63, size=2 * K + 1, dtype=np.uint64) | np.uint64(1)

    splits = collect([root / d for d in cfg.data.dirs], K, cfg.data.val_fraction, weights, mix)
    for d in args.holdout:
        splits[d] = collect([root / d], K, None, weights, mix)["all"]
    stats = {"config": str(args.config), "data_root": str(root), "train_dirs": list(cfg.data.dirs),
             "holdout": args.holdout, "K": K, "val_fraction": cfg.data.val_fraction, **summarize(splits)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
