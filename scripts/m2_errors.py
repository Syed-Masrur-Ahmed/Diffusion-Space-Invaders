"""Milestone 2 E9 error analysis: game events in rollouts, and a nearest-training-frame memorization check.

    python scripts/m2_errors.py events --config configs/eval/m2_errors.yaml --models copy_last runs/<run>/checkpoints/ckpt_last.pt
    python scripts/m2_errors.py nn --config configs/eval/m2_errors.yaml --models runs/<run>/checkpoints/ckpt_last.pt

events: roll out each model on the eval trajectories (as eval_rollouts.py does) and run the same pixel detectors on
the real and the generated frames: hits (the player-1 score changes, after snapping the score region to the nearest
score pattern seen in the real frames), life losses (the explosion sprite appears in the player band), the alien
count and the player position. The detectors are first checked against RAM on every
frame of dataset_dirs. Writes <out_dir>/events.json and events_examples.png (missed events, real above generated).

nn: for generated frames at a few horizons, and for the real frames at the same steps, the nearest frame (pixel
RMSE, 0-255) among all frames of the train split of nn.train_dirs. A model that memorizes produces frames much
closer to some training frame than real held-out frames are. Writes <out_dir>/nn.json and nn_examples.png
(generated | nearest training frame | real target).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for eval_rollouts / eval_single_step

from eval_rollouts import to_model_frames  # noqa: E402
from eval_single_step import _model_tag  # noqa: E402
from wm.config import load_config  # noqa: E402
from wm.data import ram as R  # noqa: E402
from wm.data.dataset import is_val_episode  # noqa: E402
from wm.data.reader import ShardReader, list_shards  # noqa: E402
from wm.eval.baselines import CopyLastFrame  # noqa: E402
from wm.eval.detectors import COLOUR_TOL, PLAYER_RAW_ROWS, RAW_HEIGHT, player_x, to_uint8_frames  # noqa: E402
from wm.eval.rollout import Trajectories, load_trajectories, rollout  # noqa: E402
from wm.inference import WorldModel, WorldModelProtocol  # noqa: E402

# Palette colours, checked on data/eval against RAM (see detector_check): every life loss shows the explosion sprite
# one step after RAM[73] drops, and alien pixels track RAM[17] (invaders left) with correlation 0.97.
SCORE_RGB = (50, 132, 50)       # player-1 score digits
ALIEN_RGB = (134, 134, 29)
EXPLOSION_RGB = (162, 134, 56)  # the player's explosion; also the player-2 score colour, outside the player band
SCORE_RAW_ROWS = (0, 20)
ALIEN_RAW_ROWS = (20, 185)
PIXELS_PER_ALIEN = 21           # median alien pixels per RAM invader at 84x84
NN_TOPK = 8                     # float32 candidates per chunk, re-scored exactly in integers


def _rows(raw_rows: tuple[int, int], height: int) -> slice:
    return slice(raw_rows[0] * height // RAW_HEIGHT, math.ceil(raw_rows[1] * height / RAW_HEIGHT))


def _near(pixels: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    return (np.abs(pixels.astype(np.int16) - np.array(rgb, np.int16)) < COLOUR_TOL).all(-1)


def score_masks(frames: np.ndarray) -> np.ndarray:
    """(N, D) bool: the player-1 score digit pixels of each frame, flattened."""
    return _near(frames[:, _rows(SCORE_RAW_ROWS, frames.shape[1]), :frames.shape[2] // 2], SCORE_RGB).reshape(len(frames), -1)


def score_library(dataset_dirs: list[str]) -> np.ndarray:
    """(P, D) bool: every distinct player-1 score pattern in the real frames of dataset_dirs."""
    masks = []
    for path in sorted(p for d in dataset_dirs for p in list_shards(d)):
        reader = ShardReader(path)
        masks.append(np.unique(score_masks(reader.read("frames", slice(None))), axis=0))
        reader.close()
    return np.unique(np.concatenate(masks), axis=0)


def score_changes(frames: np.ndarray, library: np.ndarray | None = None) -> np.ndarray:
    """(N,) bool: the player-1 score differs from the previous frame (False at t = 0).

    With a `library` of real score patterns each frame's score region is first snapped to the nearest pattern
    (Hamming distance), so a generated frame whose digits flicker by a few pixels keeps its score.
    """
    m = score_masks(frames)
    if library is not None:
        a, lib = m.astype(np.float32), library.astype(np.float32)
        m = (a.sum(1)[:, None] + lib.sum(1)[None] - 2 * a @ lib.T).argmin(1)[:, None]
    out = np.zeros(len(frames), bool)
    out[1:] = (m[1:] != m[:-1]).any(axis=1)
    return out


def death_onsets(frames: np.ndarray) -> np.ndarray:
    """(N,) bool: the explosion sprite appears in the player band (it was absent in the previous frame)."""
    boom = _near(frames[:, _rows(PLAYER_RAW_ROWS, frames.shape[1])], EXPLOSION_RGB).any(axis=(1, 2))
    out = np.zeros(len(frames), bool)
    out[1:] = boom[1:] & ~boom[:-1]
    return out


def alien_count(frames: np.ndarray) -> np.ndarray:
    """(N,) float: alien pixels in the alien band divided by the pixels of one alien."""
    return _near(frames[:, _rows(ALIEN_RAW_ROWS, frames.shape[1])], ALIEN_RGB).sum(axis=(1, 2)) / PIXELS_PER_ALIEN


def match_events(real: np.ndarray, gen: np.ndarray, window: int) -> tuple[int, int]:
    """(real events with a generated event within `window` steps, generated events with no real one that close)."""
    r, g = np.flatnonzero(real), np.flatnonzero(gen)
    matched = sum(bool(np.any(np.abs(g - t) <= window)) for t in r)
    false = sum(not np.any(np.abs(r - t) <= window) for t in g)
    return matched, false


def detector_check(dataset_dirs: list[str], window: int) -> dict:
    """Detectors vs RAM on every frame of dataset_dirs: hit and death recall/precision, alien-count correlation."""
    counts = {k: np.zeros(4, np.int64) for k in ("hits", "deaths")}  # real, matched, detected, detected-and-real
    aliens, invaders = [], []
    for path in sorted(p for d in dataset_dirs for p in list_shards(d)):
        reader = ShardReader(path)
        frames, ram, ep = reader.read("frames", slice(None)), reader.read("ram", slice(None)), reader.read("episode_idx", slice(None))
        reader.close()
        same = np.r_[False, ep[1:] == ep[:-1]]  # row t continues the episode of row t-1
        ram_events = {"hits": np.r_[False, np.diff(R.score(ram).astype(np.int64)) > 0] & same,
                      "deaths": np.r_[False, np.diff(ram[:, R.LIVES].astype(np.int64)) < 0] & same}
        detected = {"hits": score_changes(frames) & same, "deaths": death_onsets(frames) & same}
        for k in counts:
            matched, false = match_events(ram_events[k], detected[k], window)
            counts[k] += [ram_events[k].sum(), matched, detected[k].sum(), detected[k].sum() - false]
        aliens.append(alien_count(frames))
        invaders.append(ram[:, R.INVADERS_LEFT].astype(np.float64))
    out = {k: {"ram_events": int(c[0]), "recall": float(c[1] / max(c[0], 1)), "detected": int(c[2]),
               "precision": float(c[3] / max(c[2], 1))} for k, c in counts.items()}
    a, inv = np.concatenate(aliens), np.concatenate(invaders)
    ok = inv <= R.WAVE_SIZE
    out["alien_count_vs_ram_corr"] = float(np.corrcoef(a[ok], inv[ok])[0, 1]) if a[ok].std() > 0 and inv[ok].std() > 0 else None
    return out


def load_model(spec: str, K: int, H: int, W: int, device: torch.device) -> WorldModelProtocol:
    """copy_last or a checkpoint's EMA world model."""
    if spec == "copy_last":
        return CopyLastFrame(K, 3, H, W)
    model = WorldModel.from_checkpoint(spec, device=device)
    if (model.H, model.W) != (H, W):
        raise ValueError(f"{spec}: model frames are {model.H}x{model.W}, dataset frames are {H}x{W}")
    return model


def generate(model: WorldModelProtocol, traj: Trajectories, T: int, cfg: dict, device: torch.device) -> np.ndarray:
    """uint8 (n, T, H, W, 3) rollouts from each trajectory's first K real frames with its recorded actions."""
    K, bs = model.K, int(cfg["batch_size"])
    generator = torch.Generator().manual_seed(int(cfg["seed"]))  # CPU, so results match across devices (D21)
    out = []
    for i in range(0, len(traj.frames), bs):
        x = to_model_frames(traj.frames[i:i + bs, :K], model.C == 1).to(device)
        a = torch.from_numpy(traj.actions[i:i + bs, :K + T - 1]).to(device)
        out.append(to_uint8_frames(rollout(model, x, a, T, num_steps=cfg["num_steps"], cfg_scale=cfg["cfg_scale"],
                                           ctx_level=0, generator=generator)))
    return np.concatenate(out)


def _strip(frames: list[np.ndarray], scale: int = 3) -> np.ndarray:
    """Frames side by side, upscaled, with 2 px white gaps."""
    gap = np.full((frames[0].shape[0] * scale, 2, 3), 255, np.uint8)
    tiles = [np.kron(f, np.ones((scale, scale, 1), np.uint8)) for f in frames]
    return np.concatenate([x for t in tiles for x in (t, gap)][:-1], axis=1)


def _stack(rows: list[np.ndarray]) -> np.ndarray:
    gap = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
    return np.concatenate([x for r in rows for x in (r, gap)][:-1], axis=0)


def run_events(cfg: dict, specs: list[str], device: torch.device, out_dir: Path) -> dict:
    """Event recall and false events, alien and player errors by horizon, per model; writes events_examples.png."""
    T, K, window = int(cfg["T"]), int(cfg["K"]), int(cfg["match_window"])
    traj = load_trajectories(cfg["dataset_dirs"], int(cfg["n"]), K + T, int(cfg["start"]))
    real = traj.frames[:, K - 1:K + T]  # the last context frame, then the T real targets
    n, H, W = len(real), real.shape[2], real.shape[3]
    flat = lambda f, det: np.stack([det(f[b]) for b in range(n)])[:, 1:]  # noqa: E731  per-target, drop the context frame
    library = score_library(list(cfg["dataset_dirs"]))
    hits = lambda f: score_changes(f, library)  # noqa: E731
    real_events = {"hits": flat(real, hits), "deaths": flat(real, death_onsets)}
    real_aliens, real_x = flat(real, alien_count), flat(real, player_x)
    results, examples = {}, None
    for spec in specs:
        model = load_model(spec, K, H, W, device)
        if model.K != K:
            raise ValueError(f"{spec}: K={model.K}, the config's K is {K}")
        gen = np.concatenate([real[:, :1], generate(model, traj, T, cfg, device)], axis=1)
        gen_events = {"hits": flat(gen, hits), "deaths": flat(gen, death_onsets)}
        r = {}
        missed = {}
        for k in gen_events:
            per = [match_events(real_events[k][b], gen_events[k][b], window) for b in range(n)]
            n_real = int(real_events[k].sum())
            r[k] = {"real": n_real, "generated": int(gen_events[k].sum()), "recall": sum(m for m, _ in per) / max(n_real, 1),
                    "false_per_100_steps": 100 * sum(f for _, f in per) / (n * T)}
            missed[k] = [(b, int(t)) for b in range(n) for t in np.flatnonzero(real_events[k][b])
                         if not np.any(np.abs(np.flatnonzero(gen_events[k][b]) - t) <= window)]
        gen_aliens, gen_x = flat(gen, alien_count), flat(gen, player_x)
        hs = [h for h in cfg["report_horizons"] if h <= T]
        r["alien_abs_error"] = {str(h): float(np.abs(gen_aliens[:, h - 1] - real_aliens[:, h - 1]).mean()) for h in hs}
        r["player_x_abs_error"] = {str(h): float(np.nanmean(np.abs(gen_x[:, h - 1] - real_x[:, h - 1])))
                                   if np.isfinite(gen_x[:, h - 1] - real_x[:, h - 1]).any() else None for h in hs}
        r["player_missing_when_real_visible"] = float((np.isnan(gen_x) & ~np.isnan(real_x)).mean())
        results[_model_tag(spec)] = r
        print(f"{_model_tag(spec)}: hits {r['hits']}, deaths {r['deaths']}")
        if spec != "copy_last":
            examples = (_model_tag(spec), real[:, 1:], gen[:, 1:], missed)  # the last checkpoint is shown

    if examples is not None:
        tag, real_t, gen_t, missed = examples
        rows = []
        for k in ("deaths", "hits"):
            for b, t in missed[k][:int(cfg["n_examples"])]:
                steps = [min(max(t + d, 0), T - 1) for d in (-1, 0, 2, 5)]
                rows += [_strip([real_t[b, s] for s in steps]), _strip([gen_t[b, s] for s in steps])]
        if rows:
            iio.imwrite(out_dir / "events_examples.png", _stack(rows))
            print(f"events_examples.png: missed deaths then missed hits of {tag}; each pair is real above generated, "
                  "at t-1, t, t+2, t+5")
    return {"trajectories": traj.episodes, "T": T, "match_window": window, "models": results}


class _TrainChunks(Dataset):
    """Chunks of train-split rows of the training shards, decoded in loader workers."""

    def __init__(self, train_dirs: list[str], val_fraction: float, chunk_rows: int) -> None:
        self.paths = sorted(p for d in train_dirs for p in list_shards(d))
        self.items = []  # (shard index, train rows in one contiguous range of at most chunk_rows rows)
        for s, path in enumerate(self.paths):
            reader = ShardReader(path)
            eps = reader.episodes
            reader.close()
            keep = np.zeros(int((eps["start"] + eps["length"]).max()), bool)
            for start, length, uid in zip(eps["start"], eps["length"], eps["episode_uid"]):
                keep[start:start + length] = not is_val_episode(path.name, int(uid), val_fraction)
            for lo in range(0, len(keep), chunk_rows):
                rows = lo + np.flatnonzero(keep[lo:lo + chunk_rows])
                if len(rows):
                    self.items.append((s, rows))
        self._readers: dict[int, ShardReader] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, np.ndarray]:
        s, rows = self.items[i]
        reader = self._readers.setdefault(s, ShardReader(self.paths[s]))  # lazily, per worker (pitfall 5)
        frames = reader.read("frames", slice(int(rows[0]), int(rows[-1]) + 1))[rows - rows[0]]
        return torch.from_numpy(frames), s, rows


def nearest_train_frames(queries: np.ndarray, train_dirs: list, val_fraction: float, chunk_rows: int,
                         num_workers: int, device: torch.device | None = None) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Pixel RMSE (0-255) to the nearest train-split frame of train_dirs for each uint8 query, and its (shard, row)."""
    device = device or torch.device("cpu")
    chunks = _TrainChunks([str(d) for d in train_dirs], val_fraction, chunk_rows)
    q_int = torch.from_numpy(queries.reshape(len(queries), -1)).to(device).int()
    q = q_int.float()
    q_norm = (q ** 2).sum(1)
    best = torch.full((len(queries),), torch.iinfo(torch.int64).max, dtype=torch.int64, device=device)
    where = [("", -1)] * len(queries)
    loader = DataLoader(chunks, batch_size=None, num_workers=num_workers)
    for frames, s, rows in loader:
        x_int = frames.reshape(len(frames), -1).to(device).int()
        x = x_int.float()
        # float32 squared distances only pick candidates; the top few are re-scored exactly in integers
        d2 = q_norm[:, None] + (x ** 2).sum(1)[None] - 2 * q @ x.T
        cand = d2.topk(min(NN_TOPK, len(x)), dim=1, largest=False).indices  # (nq, k)
        exact = ((x_int[cand] - q_int[:, None]).long() ** 2).sum(-1)  # (nq, k)
        val, pos = exact.min(1)
        better = val < best
        best = torch.where(better, val, best)
        for i in torch.nonzero(better).flatten().tolist():
            where[i] = (str(chunks.paths[s]), int(rows[int(cand[i, pos[i]])]))
    rmse = torch.sqrt(best.double() / q.shape[1]).cpu().numpy()
    return rmse, where


def _rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a.astype(np.float64) - b.astype(np.float64)
    return np.sqrt((d ** 2).reshape(len(d), -1).mean(1))


def run_nn(cfg: dict, specs: list[str], device: torch.device, out_dir: Path) -> dict:
    """Nearest-training-frame distances of generated and real frames at nn.horizons; writes nn_examples.png."""
    T, K, nn = int(cfg["T"]), int(cfg["K"]), cfg["nn"]
    hs = [h for h in nn["horizons"] if h <= T]
    traj = load_trajectories(cfg["dataset_dirs"], int(cfg["n"]), K + T, int(cfg["start"]))
    n, H, W = len(traj.frames), traj.frames.shape[2], traj.frames.shape[3]
    real = np.stack([traj.frames[:, K + h - 1] for h in hs], axis=1)  # (n, len(hs), H, W, 3)
    gens = {}
    for spec in specs:
        gen = generate(load_model(spec, K, H, W, device), traj, T, cfg, device)
        gens[_model_tag(spec)] = np.stack([gen[:, h - 1] for h in hs], axis=1)
    queries = np.concatenate([real.reshape(-1, H, W, 3)] + [g.reshape(-1, H, W, 3) for g in gens.values()])
    rmse, where = nearest_train_frames(queries, nn["train_dirs"], float(nn["val_fraction"]), int(nn["chunk_rows"]),
                                       int(nn["num_workers"]), device)
    per = n * len(hs)
    real_nn = rmse[:per].reshape(n, len(hs))
    results = {}
    for m, (tag, g) in enumerate(gens.items()):
        gen_nn = rmse[per * (m + 1):per * (m + 2)].reshape(n, len(hs))
        results[tag] = {str(h): {
            "generated": {"nn_rmse": gen_nn[:, j].tolist(), "target_rmse": _rmse(g[:, j], real[:, j]).tolist(),
                          "median_nn_rmse": float(np.median(gen_nn[:, j])), "copies": float((gen_nn[:, j] < 1).mean())},
            "real": {"nn_rmse": real_nn[:, j].tolist(), "median_nn_rmse": float(np.median(real_nn[:, j])),
                     "copies": float((real_nn[:, j] < 1).mean())}} for j, h in enumerate(hs)}
        print(f"{tag}: median nearest-train RMSE generated / real: " + ", ".join(
            f"h{h} {results[tag][str(h)]['generated']['median_nn_rmse']:.2f} / {results[tag][str(h)]['real']['median_nn_rmse']:.2f}"
            for h in hs))

    # Examples for the first model at the last horizon: generated | nearest training frame | real target.
    tag, g = next(iter(gens.items()))
    j, rows = len(hs) - 1, []
    for b in range(min(n, 6)):
        path, row = where[per + b * len(hs) + j]
        reader = ShardReader(path)
        rows.append(_strip([g[b, j], reader.read("frames", slice(row, row + 1))[0], real[b, j]]))
        reader.close()
    iio.imwrite(out_dir / "nn_examples.png", _stack(rows))
    return {"train_dirs": [str(d) for d in nn["train_dirs"]], "horizons": hs, "trajectories": traj.episodes,
            "example_model": tag, "models": results}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["events", "nn"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--models", nargs="+", help="copy_last and/or checkpoint paths (overrides the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    specs = args.models or list(cfg["models"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "events":
        result = {"detectors": detector_check(list(cfg["dataset_dirs"]), int(cfg["match_window"])),
                  **run_events(cfg, specs, device, out_dir)}
        print("detectors vs RAM:", json.dumps(result["detectors"]))
    else:
        result = run_nn(cfg, specs, device, out_dir)
    (out_dir / f"{args.mode}.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_dir / (args.mode + '.json')}")
    return out_dir


if __name__ == "__main__":
    main()
