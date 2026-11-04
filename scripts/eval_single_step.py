"""Teacher-forced single-step metrics on the val split (ROADMAP 4.11, CP3-A).

    python scripts/eval_single_step.py --config configs/eval/single_step.yaml --model copy_last --n 2048
    python scripts/eval_single_step.py --config configs/eval/single_step.yaml --model runs/<run>/checkpoints/ckpt_last.pt

Each val window's target is predicted from its real context. Writes
<out_dir>/<model tag>/metrics.json (per-metric mean, std, 95% CI) and grid.png (8 windows).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wm.config import load_config
from wm.data.dataset import WindowDataset
from wm.eval.baselines import CopyLastFrame
from wm.eval.metrics import METRICS, PSNR_CAP
from wm.eval.visualize import save_grid
from wm.inference import WorldModel, WorldModelProtocol

GRID_ROWS = 8


def pick_indices(total: int, n: int) -> np.ndarray:
    """`n` dataset indices spread evenly over `range(total)` (all of them if n >= total)."""
    if n >= total:
        return np.arange(total)
    return np.round(np.linspace(0, total - 1, n)).astype(np.int64)


def evaluate(model: WorldModelProtocol, dataset: WindowDataset, indices: np.ndarray, metric_names: list[str],
             batch_size: int, device: torch.device, num_workers: int = 0, num_steps: int | None = None,
             cfg_scale: float | None = None, generator: torch.Generator | None = None) -> dict:
    """Per-window metric values {name: (n,)} and GRID_ROWS evenly spread windows for a grid."""
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, num_workers=num_workers)
    grid_pos = set(pick_indices(len(indices), GRID_ROWS).tolist())
    values = {name: [] for name in metric_names}
    grid = {"context": [], "target": [], "pred": []}
    seen = 0
    for batch in loader:
        context, target = batch["context"].to(device), batch["target"].to(device)
        pred = model.predict_next(context, batch["actions"].to(device), num_steps=num_steps,
                                  cfg_scale=cfg_scale, generator=generator)
        for name in metric_names:
            values[name].append(METRICS[name]((pred + 1) / 2, (target + 1) / 2).cpu())
        keep = [i for i in range(len(target)) if seen + i in grid_pos]
        for key, x in (("context", context), ("target", target), ("pred", pred)):
            grid[key].append(x[keep].cpu())
        seen += len(target)
    return {"values": {k: torch.cat(v) for k, v in values.items()},
            "grid": {k: torch.cat(v) for k, v in grid.items()}}


def summarize(values: torch.Tensor) -> dict:
    """Mean, sample std, and normal-approximation 95% CI half-width of per-window values."""
    v = values.double()
    std = v.std().item() if len(v) > 1 else 0.0
    return {"mean": v.mean().item(), "std": std, "ci95": 1.96 * std / max(len(v), 1) ** 0.5, "n": len(v)}


def _model_tag(spec: str) -> str:
    if spec == "copy_last":
        return spec
    path = Path(spec)  # runs/<run>/checkpoints/ckpt_<step>.pt
    return f"{path.parent.parent.name}_{path.stem}"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--model", help="copy_last or a checkpoint path (overrides the config)")
    parser.add_argument("--n", type=int, help="number of val windows (overrides the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    spec = args.model or str(cfg["model"])
    n = args.n or int(cfg["n"])
    data = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if spec == "copy_last":
        model, K, grayscale = None, int(data["K"]), bool(data["grayscale"])
    else:
        model = WorldModel.from_checkpoint(spec, device=device)
        K, grayscale = model.K, model.C == 1
    dataset = WindowDataset(data["dataset_dirs"], K, "val", grayscale=grayscale,
                            val_fraction=float(data["val_fraction"]))
    C, H, W = dataset[0]["target"].shape
    dataset.close()  # loader workers reopen the shards themselves
    if model is None:
        model = CopyLastFrame(K, C, H, W)
    elif (model.H, model.W) != (H, W):
        raise ValueError(f"model frames are {model.H}x{model.W}, dataset frames are {H}x{W}")

    indices = pick_indices(len(dataset), n)
    generator = torch.Generator(device).manual_seed(int(cfg["seed"]))
    start = time.perf_counter()
    out = evaluate(model, dataset, indices, list(cfg["metrics"]), int(data["batch_size"]), device,
                   int(data["num_workers"]), cfg["num_steps"], cfg["cfg_scale"], generator)
    seconds = time.perf_counter() - start

    metrics = {name: summarize(v) for name, v in out["values"].items()}
    if "psnr" in metrics:  # identical frames sit at the cap and pull the mean up
        metrics["psnr"]["capped_fraction"] = (out["values"]["psnr"] >= PSNR_CAP).double().mean().item()
    result = {"model": spec, "split": "val", "n": len(indices), "val_windows": len(dataset),
              "dataset_dirs": list(data["dataset_dirs"]), "K": K, "C": C, "num_steps": cfg["num_steps"],
              "cfg_scale": cfg["cfg_scale"], "seed": cfg["seed"], "device": str(device),
              "seconds": seconds, "metrics": metrics}
    out_dir = Path(cfg["out_dir"]) / _model_tag(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    save_grid(out_dir / "grid.png", out["grid"]["context"], out["grid"]["target"], out["grid"]["pred"])
    print(f"{spec} on {len(indices)} of {len(dataset)} val windows ({seconds:.0f} s): " + ", ".join(
        f"{k} {m['mean']:.4f} ± {m['ci95']:.4f}" for k, m in metrics.items()))
    print(f"Wrote {out_dir / 'metrics.json'} and {out_dir / 'grid.png'}")
    return out_dir


if __name__ == "__main__":
    main()
