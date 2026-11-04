"""Dataset coverage statistics: what game situations a collected dataset contains (ROADMAP CP2-A).

Per-frame quantities come from stored RAM (verified addresses in `wm.data.ram`), the stored
`lives`, and two pixel heuristics on the stored model frames: UFO colour in the top band and
shield colour in the shield band. Both bands are raw-screen rows scaled to the frame height.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wm.data import ram
from wm.data.reader import ShardReader, list_shards
from wm.data.schema import END_TERMINATED

RAW_HEIGHT = 210
UFO_RGB = (151, 25, 122)  # magenta; calibrated on raw frames (progress/cp2-A.md)
SHIELD_RGB = (181, 83, 40)
UFO_RAW_ROWS = (0, 31)  # above the first alien row
SHIELD_RAW_ROWS = (157, 185)

# Coverage targets (ROADMAP CP2-A task 6): name -> (minimum, is a fraction of frames).
TARGETS = {
    "frames with <= 6 invaders left": (0.05, True),
    "wave transitions": (50, False),
    "game overs": (200, False),
    "frames with the UFO visible": (0.01, True),
    "frames with shields <= 50% intact": (0.10, True),
}


def _band(raw_rows: tuple[int, int], height: int) -> tuple[int, int]:
    return raw_rows[0] * height // RAW_HEIGHT, math.ceil(raw_rows[1] * height / RAW_HEIGHT)


def shield_band(height: int) -> tuple[int, int]:
    """Frame rows [top, bottom) that hold the shields, for a frame of `height` rows."""
    return _band(SHIELD_RAW_ROWS, height)


def ufo_visible(frames: np.ndarray) -> np.ndarray:
    """(N,) bool: a magenta pixel in the top band of each (N,H,W,3) uint8 frame."""
    top, bottom = _band(UFO_RAW_ROWS, frames.shape[1])
    band = frames[:, top:bottom].astype(np.int16)
    # Magenta is the only top-band colour with blue far above green (score digits, laser
    # grey, and max-pooled mixes of them all fail this).
    magenta = (band[..., 2] - band[..., 1] >= 60) & (band[..., 0] >= 100)
    return magenta.any(axis=(1, 2))


def shield_pixels(frames: np.ndarray) -> np.ndarray:
    """(N,) int: shield-coloured pixels in the shield band of each (N,H,W,3) uint8 frame."""
    top, bottom = shield_band(frames.shape[1])
    return (frames[:, top:bottom] == np.array(SHIELD_RGB, np.uint8)).all(-1).sum(axis=(1, 2))


def count_increases(x: np.ndarray) -> int:
    """Number of steps where `x` goes up, e.g. invaders-left jumping back up at a new wave."""
    return int((np.diff(x.astype(np.int64)) > 0).sum())


def count_decreases(x: np.ndarray) -> int:
    """Number of steps where `x` goes down, e.g. a life lost."""
    return int((np.diff(x.astype(np.int64)) < 0).sum())


@dataclass
class Coverage:
    """Per-episode and per-frame coverage arrays of a dataset, plus event counts."""

    episodes: dict[str, np.ndarray]  # source, length, return_, end_reason
    frames: dict[str, np.ndarray]  # shard, row, invaders, lives, ufo, ufo_ram, shield_ratio, life_lost, wave_start, terminal
    events: dict[str, int]  # life_losses, game_overs, wave_transitions


def dataset_coverage(dataset_dir: str | Path) -> Coverage:
    """Scan every episode of a dataset (one episode in memory at a time)."""
    eps = {k: [] for k in ("source", "length", "return_", "end_reason")}
    frm = {k: [] for k in ("shard", "row", "invaders", "lives", "ufo", "ufo_ram", "shield_ratio",
                           "life_lost", "wave_start", "terminal")}
    for s, path in enumerate(list_shards(dataset_dir)):
        reader = ShardReader(path)
        for e in range(len(reader.episodes["start"])):
            for k in eps:
                eps[k].append(reader.episodes[k][e])
            rows = reader.episode_rows(e)
            frames, r, lives = reader.read("frames", rows), reader.read("ram", rows), reader.read("lives", rows)
            invaders = r[:, ram.INVADERS_LEFT]
            shields = shield_pixels(frames)
            n = len(frames)
            frm["shard"].append(np.full(n, s, np.int32))
            frm["row"].append(np.arange(rows.start, rows.stop, dtype=np.int64))
            frm["invaders"].append(invaders)
            frm["lives"].append(lives)
            frm["ufo"].append(ufo_visible(frames))
            frm["ufo_ram"].append(ram.ufo_active(r))
            frm["shield_ratio"].append(shields / shields[0] if shields[0] else np.ones(n))
            frm["life_lost"].append(np.r_[False, np.diff(lives.astype(np.int64)) < 0])
            frm["wave_start"].append(np.r_[False, np.diff(invaders.astype(np.int64)) > 0])
            frm["terminal"].append(reader.read("terminal", rows))
        reader.close()
    episodes = {k: np.array(v) for k, v in eps.items()}
    frames = {k: np.concatenate(v) if v else np.zeros(0) for k, v in frm.items()}
    events = {"life_losses": int(frames["life_lost"].sum()),
              "game_overs": int((episodes["end_reason"] == END_TERMINATED).sum()),
              "wave_transitions": int(frames["wave_start"].sum())}
    return Coverage(episodes, frames, events)


def target_values(cov: Coverage) -> dict[str, float]:
    """Measured value for each entry of TARGETS."""
    f = cov.frames
    mean = lambda x: float(np.mean(x)) if len(x) else 0.0  # noqa: E731
    return {
        "frames with <= 6 invaders left": mean(f["invaders"] <= 6),
        "wave transitions": cov.events["wave_transitions"],
        "game overs": cov.events["game_overs"],
        "frames with the UFO visible": mean(f["ufo"]),
        "frames with shields <= 50% intact": mean(f["shield_ratio"] <= 0.5),
    }
