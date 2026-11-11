"""Autoregressive rollouts on held-out eval trajectories: metrics vs horizon (ROADMAP 4.11, CP4-A).

    python scripts/eval_rollouts.py --config configs/eval/rollouts.yaml \
        --models copy_last runs/<run>/checkpoints/ckpt_last.pt --T 200 --n 64 --set ctx_levels=[0,1,2]

Each trajectory starts from K real frames `start` rows into its episode and then feeds the model its own
frames, with the recorded actions. Writes to <out_dir>/: metrics.json and metrics.csv (mean,
std and 95% CI per horizon h = 1..T), horizon.png (all models), and videos/<model>_traj<i>.mp4.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for eval_single_step, also when imported by path

from eval_single_step import _model_tag, pick_indices, summarize  # noqa: E402
from wm.config import load_config  # noqa: E402
from wm.data.dataset import LUMA  # noqa: E402
from wm.eval.baselines import CopyLastFrame  # noqa: E402
from wm.eval.plots import plot_horizon, save_rollout_video  # noqa: E402
from wm.eval.rollout import horizon_metrics, load_trajectories, rollout  # noqa: E402
from wm.inference import WorldModel, WorldModelProtocol  # noqa: E402


def to_model_frames(frames: np.ndarray, grayscale: bool) -> torch.Tensor:
    """uint8 (..., H, W, 3) -> float (..., C, H, W) in [-1,1], as WindowDataset normalizes (4.5)."""
    x = torch.from_numpy(frames).movedim(-1, -3).float()  # float before any arithmetic on uint8
    if grayscale:
        x = (x * LUMA[:, None, None]).sum(-3, keepdim=True)
    return x / 127.5 - 1


def build_models(specs: list[str], cfg: dict, device: torch.device) -> list[tuple[str, str, WorldModelProtocol, int]]:
    """(tag, spec, model, ctx_level) for copy_last once and for each checkpoint at every ctx level."""
    entries = []
    for spec in specs:
        if spec == "copy_last":
            entries.append((spec, spec, CopyLastFrame(int(cfg["K"]), 1 if cfg["grayscale"] else 3, 0, 0), 0))
            continue
        model = WorldModel.from_checkpoint(spec, device=device)
        entries += [(f"{_model_tag(spec)}@ctx{level}", spec, model, int(level)) for level in cfg["ctx_levels"]]
    return entries


def evaluate(model: WorldModelProtocol, frames: np.ndarray, actions: np.ndarray, T: int, ctx_level: int,
             cfg: dict, device: torch.device, video_ids: set[int]) -> tuple[dict, dict]:
    """Roll out every trajectory: per-frame metric values {name: (n,T)} and {trajectory: (real, generated)}."""
    K, gray = model.K, model.C == 1
    generator = torch.Generator().manual_seed(int(cfg["seed"]))  # CPU, so results match across devices (D21)
    values, videos = {name: [] for name in cfg["metrics"]}, {}
    bs = int(cfg["batch_size"])
    for i in range(0, len(frames), bs):
        x = to_model_frames(frames[i:i + bs, :K + T], gray).to(device)
        a = torch.from_numpy(actions[i:i + bs, :K + T - 1]).to(device)
        gen = rollout(model, x[:, :K], a, T, num_steps=cfg["num_steps"], cfg_scale=cfg["cfg_scale"],
                      ctx_level=ctx_level, generator=generator)
        for name, v in horizon_metrics(gen, x[:, K:], list(cfg["metrics"])).items():
            values[name].append(v)
        for b in range(len(gen)):
            if i + b in video_ids:
                videos[i + b] = (x[b, K:].cpu(), gen[b].cpu())
    return {k: torch.cat(v) for k, v in values.items()}, videos


def per_horizon(values: torch.Tensor) -> dict[str, list]:
    """Mean, std and 95% CI over trajectories at each horizon h = 1..T, from (n, T) values."""
    stats = [summarize(values[:, t]) for t in range(values.shape[1])]
    return {"h": list(range(1, len(stats) + 1)), **{k: [s[k] for s in stats] for k in ("mean", "std", "ci95")}}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--models", nargs="+", help="copy_last and/or checkpoint paths (overrides the config)")
    parser.add_argument("--T", type=int, help="generated frames per trajectory (overrides the config)")
    parser.add_argument("--n", type=int, help="number of trajectories (overrides the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    specs, T, n = args.models or list(cfg["models"]), args.T or int(cfg["T"]), args.n or int(cfg["n"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])

    entries = build_models(specs, cfg, device)
    traj = load_trajectories(cfg["dataset_dirs"], n, max(m.K for _, _, m, _ in entries) + T, int(cfg["start"]))
    H, W = traj.frames.shape[2:4]
    for tag, _, model, _ in entries:
        if isinstance(model, CopyLastFrame):  # its frame size is only known once the data is loaded
            model.H, model.W = H, W
        elif (model.H, model.W) != (H, W):
            raise ValueError(f"{tag}: model frames are {model.H}x{model.W}, dataset frames are {H}x{W}")
    video_ids = set(pick_indices(n, int(cfg["n_videos"])).tolist()) if int(cfg["n_videos"]) > 0 else set()
    report_h = [h for h in cfg["report_horizons"] if h <= T]

    results = {}
    for tag, spec, model, level in entries:
        start = time.perf_counter()
        values, videos = evaluate(model, traj.frames, traj.actions, T, level, cfg, device, video_ids)
        metrics = {name: per_horizon(v) for name, v in values.items()}
        results[tag] = {"model": spec, "ctx_level": level,
                        "n": n, "T": T, "K": model.K, "C": model.C, "seconds": time.perf_counter() - start,
                        "metrics": metrics,
                        "report": {name: {str(h): {"mean": m["mean"][h - 1], "ci95": m["ci95"][h - 1]} for h in report_h}
                                   for name, m in metrics.items()}}
        for b, (real, gen) in videos.items():
            save_rollout_video(out_dir / "videos" / f"{tag}_traj{b:02d}.mp4", real, gen, fps=int(cfg["fps"]),
                               scale=int(cfg["video_scale"]), widen=int(cfg["video_widen"]))
        print(f"{tag} ({results[tag]['seconds']:.0f} s): " + "; ".join(
            f"{name} " + " ".join(f"h{h}={r['mean']:.4g}" for h, r in rep.items())
            for name, rep in results[tag]["report"].items()))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "dataset_dirs": list(cfg["dataset_dirs"]), "start": cfg["start"], "trajectories": traj.episodes, "num_steps": cfg["num_steps"],
        "cfg_scale": cfg["cfg_scale"], "seed": cfg["seed"], "device": str(device), "models": results}, indent=2))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        cols = [f"{name}_{s}" for name in cfg["metrics"] for s in ("mean", "ci95")]
        writer = csv.DictWriter(f, ["model", "ctx_level", "h", *cols])
        writer.writeheader()
        for tag, r in results.items():
            for t in range(T):
                writer.writerow({"model": tag, "ctx_level": r["ctx_level"], "h": t + 1,
                                 **{f"{name}_{s}": m[s][t] for name, m in r["metrics"].items() for s in ("mean", "ci95")}})
    plot_horizon(out_dir / "horizon.png", {tag: r["metrics"] for tag, r in results.items()})
    print(f"Wrote {out_dir / 'metrics.json'}, metrics.csv, horizon.png and {len(video_ids) * len(results)} videos")
    return out_dir


if __name__ == "__main__":
    main()
