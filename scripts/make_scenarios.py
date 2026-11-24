"""Pick starting scenarios for the playable demo from held-out episodes, using RAM (ROADMAP CP6-B).

    python scripts/make_scenarios.py --config configs/demo/scenarios.yaml

A scenario is a row r of a stored episode: the demo starts from frames r-K+1..r and actions
a_{r-K+1}..a_{r-1}, and the player's first action produces frame r+1. Types, each taken from the
first matching row of an episode:
  fresh_game    step `fresh_step` of an episode
  few_invaders  1..`max_invaders` aliens left
  ufo           `ufo_lead` steps before the UFO appears
  one_life      `one_life_settle` steps into the last life
Writes `out` as a JSON list of {name, dataset, shard, row} (row = r, the shard's global row).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wm.config import load_config
from wm.data import ram as ram_map
from wm.data.reader import ShardReader, list_shards

TYPES = ("fresh_game", "few_invaders", "ufo", "one_life")
FIELDS = {"name": str, "dataset": str, "shard": str, "row": int}
MAX_SCENARIOS = 9  # number keys 1-9 in the demo


@dataclass
class Scenario:
    """What the demo needs to start from a scenario, and to replay it on the real emulator."""

    name: str
    frames: np.ndarray  # uint8 (K,H,W,3): rows r-K+1..r
    actions: np.ndarray  # int64 (K-1,): a_{r-K+1}..a_{r-1}
    seed: int
    noops: int
    prefix_actions: np.ndarray  # int64 (step,): a_0..a_{r-1} of the episode
    resolution: int
    resize_method: str


def _run_length(mask: np.ndarray) -> np.ndarray:
    """Number of consecutive True values ending at each index (0 where mask is False)."""
    out = np.zeros(len(mask), np.int64)
    for i, m in enumerate(mask):
        out[i] = out[i - 1] + 1 if m and i else int(m)
    return out


def scenario_masks(ram: np.ndarray, lives: np.ndarray, steps: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """For one episode's rows, a boolean mask per scenario type."""
    invaders = ram[:, ram_map.INVADERS_LEFT].astype(np.int64)
    ufo = ram_map.ufo_active(ram)
    lead = int(cfg["ufo_lead"])
    appears = np.zeros(len(ufo), bool)  # appears[i]: the UFO is active at i and was not at i-1
    appears[1:] = ufo[1:] & ~ufo[:-1]
    ufo_soon = np.zeros(len(ufo), bool)
    ufo_soon[:len(ufo) - lead] = appears[lead:]
    return {
        "fresh_game": steps == int(cfg["fresh_step"]),
        "few_invaders": (invaders >= 1) & (invaders <= int(cfg["max_invaders"])) & (lives >= 1),
        "ufo": ufo_soon & (lives >= 1),
        "one_life": _run_length(lives == 1) > int(cfg["one_life_settle"]),
    }


def find_scenarios(cfg: dict) -> list[dict]:
    """`per_type` entries per type, each from a different episode spread evenly over the matching ones."""
    K, min_future = int(cfg["K"]), int(cfg["min_future"])
    found = {t: [] for t in TYPES}  # type -> [(dataset, shard name, row)], one per episode
    for dataset in cfg["dataset_dirs"]:
        for path in list_shards(dataset):
            reader = ShardReader(path)
            ram, lives, steps = reader.read("ram", slice(None)), reader.read("lives", slice(None)), reader.read(
                "step_in_episode", slice(None))
            eps = reader.episodes
            for start, length in zip(eps["start"].tolist(), eps["length"].tolist()):
                rows = slice(start, start + length)
                valid = (steps[rows] >= K - 1) & (np.arange(length) + min_future <= length - 1)
                for t, mask in scenario_masks(ram[rows], lives[rows].astype(np.int64), steps[rows], cfg).items():
                    hits = np.flatnonzero(mask & valid)
                    if len(hits):
                        found[t].append((str(dataset), path.name, start + int(hits[0])))
            reader.close()
    missing = [t for t in TYPES if not found[t]]
    if missing:
        raise ValueError(f"no episode has a row for: {', '.join(missing)} (with min_future={min_future})")
    entries = []
    n = int(cfg["per_type"])
    for t in TYPES:
        picks = np.unique(np.round(np.linspace(0, len(found[t]) - 1, n)).astype(int))
        for i, p in enumerate(picks):
            dataset, shard, row = found[t][p]
            entries.append({"name": t if n == 1 else f"{t}_{i + 1}", "dataset": dataset, "shard": shard, "row": row})
    validate_entries(entries)
    return entries


def validate_entries(entries: list) -> None:
    """Raise ValueError unless `entries` is a list of 1-9 {name, dataset, shard, row} dicts with unique names."""
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_SCENARIOS:
        raise ValueError(f"need a list of 1 to {MAX_SCENARIOS} scenarios")
    for e in entries:
        if not isinstance(e, dict) or set(e) != set(FIELDS):
            raise ValueError(f"scenario {e!r} must have exactly the keys {sorted(FIELDS)}")
        for key, kind in FIELDS.items():
            if type(e[key]) is not kind:
                raise ValueError(f"scenario {e['name']!r}: {key} must be {kind.__name__}")
    names = [e["name"] for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("scenario names must be unique")


def load_scenario(entry: dict, K: int, root: str | Path = ".") -> Scenario:
    """Read a scenario's K frames, K-1 context actions and replay information from its shard."""
    reader = ShardReader(Path(root) / entry["dataset"] / entry["shard"])
    r = entry["row"]
    e = int(reader.read("episode_idx", slice(r, r + 1))[0])
    start = int(reader.episodes["start"][e])
    if r - K + 1 < start:
        raise ValueError(f"scenario {entry['name']!r}: rows {r - K + 1}..{r} cross the start of episode {e} (row {start})")
    prefix = reader.read("actions", slice(start, r)).astype(np.int64)
    scenario = Scenario(entry["name"], reader.read("frames", slice(r - K + 1, r + 1)), prefix[len(prefix) - (K - 1):],
                        int(reader.episodes["seed"][e]), int(reader.episodes["noops"][e]), prefix,
                        int(reader.attrs["resolution"]), str(reader.attrs["resize_method"]))
    reader.close()
    return scenario


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/demo/scenarios.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    entries = find_scenarios(cfg)
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2) + "\n")
    for i, e in enumerate(entries, 1):
        print(f"{i}: {e['name']:<16} {e['dataset']}/{e['shard']} row {e['row']}")
    print(f"wrote {len(entries)} scenarios to {out}")
    return entries


if __name__ == "__main__":
    main()
