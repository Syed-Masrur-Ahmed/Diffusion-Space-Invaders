"""Train the inverse dynamics model on real transitions and report its accuracy on real val (ROADMAP CP5-A).

    python scripts/train_idm.py --config configs/train/idm.yaml [--set steps=500]

Items are WindowDataset windows with K=1: context f_t, action a_t, target f_{t+1}. Writes
<out_dir>/idm.pt (weights, config, val metrics) and <out_dir>/metrics.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wm.config import load_config
from wm.data.dataset import DataConfig, make_dataloader
from wm.models.idm import IDM, action_targets, loss_and_accuracy, save_idm


@torch.inference_mode()
def evaluate(model: IDM, loader, device: torch.device, max_batches: int | None = None) -> dict:
    """Accuracy and confusion matrices (rows = true class) of both heads, with majority-class baselines."""
    model.eval()
    conf = {"move": np.zeros((3, 3), np.int64), "fire": np.zeros((2, 2), np.int64)}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        move, fire = model(batch["context"][:, -1].to(device), batch["target"].to(device))
        for name, logits, target in zip(("move", "fire"), (move, fire), action_targets(batch["actions"][:, -1])):
            np.add.at(conf[name], (target.numpy(), logits.argmax(1).cpu().numpy()), 1)
    model.train()
    out = {"n": int(conf["move"].sum())}
    for name, c in conf.items():
        out[name] = {"accuracy": float(np.trace(c) / c.sum()), "majority": float(c.sum(1).max() / c.sum()),
                     "confusion": c.tolist()}
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["seed"]))
    out_dir = Path(cfg["out_dir"])

    data = DataConfig(list(cfg["dataset_dirs"]), K=1, batch_size=int(cfg["batch_size"]),
                      num_workers=int(cfg["num_workers"]), val_fraction=float(cfg["val_fraction"]),
                      seed=int(cfg["seed"]), in_memory=bool(cfg["in_memory"]))
    train, val = make_dataloader(data, "train"), make_dataloader(data, "val")
    C, H, W = train.dataset[0]["target"].shape
    model = IDM(C, H, W, tuple(cfg["model"]["ch"]), int(cfg["model"]["hidden"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    print(f"IDM {sum(p.numel() for p in model.parameters()):,} params; {len(train.dataset):,} train / "
          f"{len(val.dataset):,} val transitions; {device}")

    step, start, batches = 0, time.perf_counter(), iter(train)
    while step < int(cfg["steps"]):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(train)
            continue
        loss, acc = loss_and_accuracy(model, batch["context"][:, -1].to(device), batch["target"].to(device),
                                      batch["actions"][:, -1].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % int(cfg["log_every"]) == 0:
            print(f"step {step} loss {loss.item():.4f} move {acc['move']:.3f} fire {acc['fire']:.3f} "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
        if step % int(cfg["eval_every"]) == 0 and step < int(cfg["steps"]):
            quick = evaluate(model, val, device, int(cfg["val_batches"]))
            print(f"  val (first {quick['n']}): move {quick['move']['accuracy']:.3f} fire {quick['fire']['accuracy']:.3f}")

    metrics = {"steps": step, "seconds": time.perf_counter() - start, "val": evaluate(model, val, device)}
    v = metrics["val"]
    print(f"real val ({v['n']:,} transitions): move {v['move']['accuracy']:.4f} (majority {v['move']['majority']:.4f}), "
          f"fire {v['fire']['accuracy']:.4f} (majority {v['fire']['majority']:.4f})")
    save_idm(out_dir / "idm.pt", model.cpu(), step=step, val=v, train_config=cfg)
    (out_dir / "metrics.json").write_text(json.dumps({"config": cfg, **metrics}, indent=2))
    return out_dir


if __name__ == "__main__":
    main()
