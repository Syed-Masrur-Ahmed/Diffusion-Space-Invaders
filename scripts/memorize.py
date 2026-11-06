"""Memorization check: fine-tune a trained checkpoint on n train windows, then score it on the same windows.

    python scripts/memorize.py --config configs/eval/single_step.yaml \\
        --checkpoint runs/20261105-0558_debug_small/checkpoints/ckpt_last.pt --out results/milestone1/memorize

Starts from the EMA weights and fine-tunes them on one full batch of `n` evenly spaced train
windows (AdamW, no action dropout, grad clip 1.0), then evaluates copy-last, the checkpoint before
fine-tuning, and after, all on those same windows with `eval_single_step.evaluate`. Writes
<out>/metrics.json and <out>/grid.png (after fine-tuning). The checkpoint file is not modified.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for eval_single_step, also when imported by path

from eval_single_step import evaluate, pick_indices, summarize  # noqa: E402
from wm.config import load_config  # noqa: E402
from wm.data.dataset import WindowDataset  # noqa: E402
from wm.eval.baselines import CopyLastFrame  # noqa: E402
from wm.eval.visualize import save_grid  # noqa: E402
from wm.inference import WorldModel  # noqa: E402


def run(ckpt: str, cfg: dict, n: int, steps: int, lr: float, device: torch.device, seed: int = 0) -> dict:
    """Fine-tune `ckpt` on n train windows; metrics for copy_last / before / after on those windows, losses, grid."""
    model = WorldModel.from_checkpoint(ckpt, device=device)
    data = cfg["data"]
    dataset = WindowDataset(data["dataset_dirs"], model.K, "train", grayscale=model.C == 1,
                            val_fraction=float(data["val_fraction"]))
    indices = pick_indices(len(dataset), n)
    batch = default_collate([dataset[int(i)] for i in indices])
    context, actions, target = (batch[k].to(device) for k in ("context", "actions", "target"))

    def score(m) -> dict:
        out = evaluate(m, dataset, indices, list(cfg["metrics"]), len(indices), device, 0, cfg["num_steps"],
                       cfg["cfg_scale"], torch.Generator().manual_seed(int(cfg["seed"])))
        return {"metrics": {k: summarize(v) for k, v in out["values"].items()}, "grid": out["grid"]}

    copy_last = score(CopyLastFrame(model.K, model.C, model.H, model.W))
    before = score(model)

    net = model.denoiser.net.requires_grad_(True).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    ctx_level = torch.zeros_like(actions)
    torch.manual_seed(seed)  # sigma and noise draws
    losses, start = [], time.perf_counter()
    with torch.enable_grad():  # callers (the notebook) may have disabled grad globally
        for _ in range(steps):
            per, _ = model.denoiser.loss_terms(target, context, actions, ctx_level)
            loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
    seconds = time.perf_counter() - start
    net.requires_grad_(False).eval()
    after = score(model)
    dataset.close()
    return {"n": len(indices), "steps": steps, "lr": lr, "seed": seed, "device": str(device), "seconds": seconds,
            "train_windows": len(dataset), "indices": indices.tolist(), "losses": losses,
            "metrics": {"copy_last": copy_last["metrics"], "before": before["metrics"], "after": after["metrics"]},
            "grid": after["grid"]}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="eval config (data, metrics, seed)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=20, help="train windows to memorize")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=Path("results/milestone1/memorize"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    result = run(args.checkpoint, cfg, args.n, args.steps, args.lr, device)
    grid = result.pop("grid")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps({"checkpoint": args.checkpoint, **result}, indent=2))
    save_grid(args.out / "grid.png", grid["context"], grid["target"], grid["pred"])
    for name, metrics in result["metrics"].items():
        print(f"{name:9s} " + ", ".join(f"{k} {m['mean']:.4f} ± {m['ci95']:.4f}" for k, m in metrics.items()))
    print(f"{args.steps} steps in {result['seconds']:.0f} s on {device}; wrote {args.out}")
    return args.out


if __name__ == "__main__":
    main()
