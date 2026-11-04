"""CP2 integration smoke test: the window DataLoader feeds EDMDenoiser(TinyDenoiserNet) on CPU.

Trains for `steps` steps on the train split of `dataset_dir`, checks that no loss is NaN and
that the mean loss over the last `window` steps is below the mean over the first `window`.
Exits non-zero otherwise.

    python scripts/smoke_cp2.py
    python scripts/smoke_cp2.py --set dataset_dir=data/bench_real --set steps=50
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for tests.fixtures

from tests.fixtures import TinyDenoiserNet  # noqa: E402
from wm.config import from_dict, load_config  # noqa: E402
from wm.data.dataset import DataConfig, make_dataloader  # noqa: E402
from wm.diffusion.edm import EDMConfig, EDMDenoiser, loss_by_sigma  # noqa: E402

DEFAULTS = {
    "dataset_dir": "data/debug",
    "edm_config": "configs/train/edm_default.yaml",
    "K": 4,
    "batch_size": 32,
    "num_workers": 4,
    "steps": 200,
    "lr": 2e-3,
    "hidden": 32,
    "seed": 0,
    "window": 20,
}


def train(cfg: dict) -> list[float]:
    """Run `cfg["steps"]` optimizer steps and return the per-step losses."""
    torch.manual_seed(int(cfg["seed"]))
    data_cfg = DataConfig(dataset_dirs=[cfg["dataset_dir"]], K=int(cfg["K"]),
                          batch_size=int(cfg["batch_size"]), num_workers=int(cfg["num_workers"]),
                          seed=int(cfg["seed"]))
    loader = make_dataloader(data_cfg, "train")
    print(f"train windows: {len(loader.dataset):,}")
    edm_cfg = from_dict(EDMConfig, load_config(cfg["edm_config"])["edm"])
    denoiser = EDMDenoiser(TinyDenoiserNet(C=3, K=int(cfg["K"]), hidden=int(cfg["hidden"])), edm_cfg)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=float(cfg["lr"]))
    losses: list[float] = []
    batches = iter(loader)
    while len(losses) < int(cfg["steps"]):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            continue
        ctx_level = torch.zeros_like(batch["actions"])
        per, sigma = denoiser.loss_terms(batch["target"], batch["context"], batch["actions"], ctx_level)
        loss = per.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if len(losses) % 50 == 0:
            print(f"step {len(losses):4d}  loss {loss.item():.4f}  by sigma {loss_by_sigma(per.detach(), sigma)}")
    return losses


def resolve_config(argv: list[str] | None = None) -> dict:
    """Merge DEFAULTS with an optional YAML file and `--set key=value` overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config is not None:
        cfg.update(load_config(args.config))
    for item in args.overrides:
        key, sep, raw = item.partition("=")
        if not sep or key not in cfg:
            raise KeyError(f"bad override {item!r}; keys: {sorted(cfg)}")
        cfg[key] = yaml.safe_load(raw)
    return cfg


def main(argv: list[str] | None = None) -> int:
    cfg = resolve_config(argv)
    start = time.perf_counter()
    losses = train(cfg)
    n = int(cfg["window"])
    first, last = sum(losses[:n]) / n, sum(losses[-n:]) / n
    ok = all(math.isfinite(v) for v in losses) and last < first
    print(f"{len(losses)} steps in {time.perf_counter() - start:.0f} s; mean loss first {n}: {first:.4f}, "
          f"last {n}: {last:.4f}; finite: {all(math.isfinite(v) for v in losses)}; {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
