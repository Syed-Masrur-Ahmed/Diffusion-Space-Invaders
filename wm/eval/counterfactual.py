"""Counterfactual action programs on the real emulator and on a world model (ROADMAP CP5-A).

A branch is a row t of an eval episode. The model starts from the real context f_{t-K+1}..f_t
with actions a_{t-K+1}..a_{t-1} and then follows a constant action program, so its first frame
g_{t+1} is produced by the program's first action (4.11 alignment). The real reference replays
the episode's seed and recorded actions up to t (D9), checks that it reached the stored frame,
and then runs each program from a cloned emulator state.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wm.data import ram as ram_map
from wm.data.reader import ShardReader, list_shards
from wm.data.schema import ACTION_MEANINGS
from wm.env import MaxPoolSkip, make_env, to_model_frame
from wm.eval import detectors
from wm.eval.rollout import rollout
from wm.inference import WorldModelProtocol

PROGRAMS = ("NOOP", "LEFT", "RIGHT", "FIRE", "LEFTFIRE", "RIGHTFIRE")  # constant action programs
# Player laser y (found in CP5-A): LASER_IDLE when no player shot is in flight, otherwise it falls
# by 4 per agent step from ~85. Used only to choose which branches test the fire response.
PLAYER_LASER_Y = 85
LASER_IDLE = 246
# RAM player x (wm.data.ram.PLAYER_X) runs from 35 at the left wall to 117 at the right (CP2-A, CP5-A);
# 8 units is ~4 px at 84, so from a ready row both LEFT and RIGHT can move the player > 2 px from NOOP.
PLAYER_RAM_X_RANGE = (35, 117)
WALL_CLEARANCE = 8


@dataclass
class Branch:
    """A branch point: row t of an eval episode and what both the model and the emulator need."""

    shard: str
    episode: int  # shard-local episode index
    t: int  # row within the episode
    seed: int
    noops: int
    frames: np.ndarray  # uint8 (K,H,W,3): rows t-K+1..t
    prefix_actions: np.ndarray  # int64 (t,): a_0..a_{t-1}
    laser_idle: bool  # no player laser in flight at row t (RAM)
    resolution: int
    resize_method: str

    @property
    def context_actions(self) -> np.ndarray:
        """a_{t-K+1}..a_{t-1}: the K-1 real actions in the model's first window."""
        return self.prefix_actions[len(self.prefix_actions) - (len(self.frames) - 1):]


def pick_branches(dataset_dirs: list[str | Path], n: int, K: int, horizon: int, min_step: int, seed: int,
                  require_ready: bool = True) -> list[Branch]:
    """One branch in each of n episodes spread evenly over the eligible ones, t drawn with `seed`.

    t is uniform over rows with t >= max(K, min_step) and t + horizon <= the episode's last row.
    With `require_ready`, row t must also show the player, away from both walls, with no player
    laser in flight, so every branch can test movement in both directions and the fire response.
    """
    eligible = []
    for path in sorted(p for d in dataset_dirs for p in list_shards(d)):
        reader = ShardReader(path)
        lo = max(K, min_step)
        eligible += [(path, int(e)) for e in np.flatnonzero(reader.episodes["length"] - 1 - horizon >= lo)]
        reader.close()
    if n > len(eligible):
        raise ValueError(f"asked for {n} branches, but only {len(eligible)} episodes have room for t >= {max(K, min_step)} "
                         f"and {horizon} steps after it")
    rng = np.random.default_rng(seed)
    branches = []
    for i in np.round(np.linspace(0, len(eligible) - 1, n)).astype(int):
        path, e = eligible[i]
        reader = ShardReader(path)
        start, length = int(reader.episodes["start"][e]), int(reader.episodes["length"][e])
        rows = np.arange(max(K, min_step), length - horizon)
        ram = reader.read("ram", slice(start, start + length))[rows]
        x = ram[:, ram_map.PLAYER_X].astype(np.int64)
        ready = ((ram[:, PLAYER_LASER_Y] == LASER_IDLE) & (x >= PLAYER_RAM_X_RANGE[0] + WALL_CLEARANCE)
                 & (x <= PLAYER_RAM_X_RANGE[1] - WALL_CLEARANCE))
        for t in rng.permutation(rows[ready] if require_ready else rows):
            frames = reader.read("frames", slice(start + t - K + 1, start + t + 1))
            if not require_ready or not np.isnan(detectors.player_x(frames[-1:]))[0]:
                break
        else:
            raise ValueError(f"{path} episode {e}: no eligible row is ready (player visible, off the walls, no laser)")
        ram = reader.read("ram", slice(start + t, start + t + 1))[0]
        branches.append(Branch(
            str(path), e, int(t), int(reader.episodes["seed"][e]), int(reader.episodes["noops"][e]), frames,
            reader.read("actions", slice(start, start + t)).astype(np.int64),
            bool(ram[PLAYER_LASER_Y] == LASER_IDLE), int(reader.attrs["resolution"]),
            str(reader.attrs["resize_method"])))
        reader.close()
    return branches


def _skip_env(env) -> MaxPoolSkip:
    while not isinstance(env, MaxPoolSkip):
        env = env.env
    return env


def real_branches(branch: Branch, programs: list[int], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Emulator frames uint8 (P, horizon+1, H, W, 3) for each program (step 0 = row t) and (P,) ended flags.

    A program that ends the episode repeats its last frame for the remaining steps.
    """
    env = make_env(branch.seed)
    obs, info = env.reset()
    if info["noops"] != branch.noops:
        raise RuntimeError(f"replay of {branch.shard} episode {branch.episode}: {info['noops']} no-ops, stored {branch.noops}")
    for a in branch.prefix_actions:
        obs, *_ = env.step(int(a))
    frame0 = to_model_frame(obs, branch.resolution, branch.resize_method)
    if not np.array_equal(frame0, branch.frames[-1]):
        raise RuntimeError(f"replay of {branch.shard} episode {branch.episode} does not reach the stored row {branch.t}")
    ale, skip = env.unwrapped.ale, _skip_env(env)  # step below TimeLimit: restoreState does not rewind its counter
    state = ale.cloneState()
    out = np.empty((len(programs), horizon + 1, *frame0.shape), np.uint8)
    ended = np.zeros(len(programs), bool)
    for p, action in enumerate(programs):
        ale.restoreState(state)
        out[p, 0] = frame0
        for s in range(1, horizon + 1):
            if ended[p]:
                out[p, s] = out[p, s - 1]
                continue
            obs, _, terminated, truncated, _ = skip.step(action)
            out[p, s] = to_model_frame(obs, branch.resolution, branch.resize_method)
            ended[p] = terminated or truncated
    env.close()
    return out, ended


def model_branches(model: WorldModelProtocol, branches: list[Branch], programs: list[int], horizon: int,
                   batch_size: int, device: torch.device | str = "cpu", **sample_kwargs) -> np.ndarray:
    """Model frames uint8 (B, P, horizon+1, H, W, 3) for every branch and program (step 0 = real row t).

    Rollouts run in batches of `batch_size` (branch, program) pairs, branch-major; `sample_kwargs`
    (num_steps, cfg_scale, ctx_level, generator) go to every predict_next call.
    """
    if model.C != 3:
        raise ValueError("the pixel detectors need RGB frames (C=3)")
    pairs = [(b, p) for b in range(len(branches)) for p in range(len(programs))]
    out = np.empty((len(branches), len(programs), horizon + 1, *branches[0].frames.shape[1:]), np.uint8)
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        frames0 = torch.stack([torch.from_numpy(branches[b].frames) for b, _ in chunk]).movedim(-1, -3).float() / 127.5 - 1
        actions = torch.stack([torch.from_numpy(np.concatenate([branches[b].context_actions,
                                                                np.full(horizon, programs[p])]))
                               for b, p in chunk])
        gen = detectors.to_uint8_frames(rollout(model, frames0.to(device), actions.to(device), horizon, **sample_kwargs))
        for (b, p), frames in zip(chunk, gen):
            out[b, p, 0], out[b, p, 1:] = branches[b].frames[-1], frames
    return out


def _idx(name: str) -> int:
    return PROGRAMS.index(name)


def ordered(x: np.ndarray, margin: float) -> np.ndarray:
    """(B,) bool: x_LEFT + margin < x_NOOP < x_RIGHT - margin at the last step; x is (B, P, T+1)."""
    left, noop, right = (x[:, _idx(p), -1] for p in ("LEFT", "NOOP", "RIGHT"))
    with np.errstate(invalid="ignore"):  # a missing player (NaN) counts as a failure
        return (left + margin < noop) & (noop + margin < right)


def mean_dx(x: np.ndarray) -> dict[str, float]:
    """Mean per-step change in player x for each program, over steps where both frames show the player."""
    dx = np.diff(x, axis=-1)
    return {name: float(np.nanmean(dx[:, p])) if np.isfinite(dx[:, p]).any() else float("nan")
            for p, name in enumerate(PROGRAMS)}


def fire_response(laser: np.ndarray, idle: np.ndarray, programs: list[str], window: int) -> tuple[float, int]:
    """P(laser_present at some step 1..window | no player laser in flight at step 0), pooled over `programs`, and n."""
    hit = laser[idle][:, [_idx(p) for p in programs], 1:window + 1].any(axis=-1)
    return (float(hit.mean()) if hit.size else float("nan")), int(hit.size)


def program_actions() -> list[int]:
    """Action ids of PROGRAMS, in order."""
    return [ACTION_MEANINGS.index(p) for p in PROGRAMS]
