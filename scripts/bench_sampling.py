"""Sampler speed and single-step quality (ROADMAP CP4-C).

    python scripts/bench_sampling.py --config configs/eval/bench_sampling.yaml \
        --set model=runs/<run>/checkpoints/ckpt_last.pt

Writes <out_dir>/results.json, results.csv, and quality_vs_speed.png.
D11 is the fewest `proposal_method` steps within `psnr_tol_db` of `reference_steps`
that reaches `min_steps_per_s` on `proposal_device` (the training GPU).
"""

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wm.config import load_config
from wm.data.dataset import WindowDataset
from wm.eval.metrics import METRICS
from wm.inference import WorldModel

COLUMNS = ["device", "method", "num_steps", "steps_per_s", "psnr", "lpips"]


def propose_d11(rows: list[dict], *, reference_steps: int = 10, method: str = "euler", tol_db: float = 0.3,
                min_steps_per_s: float = 15.0, device: str = "cuda") -> dict:
    """Fewest `method` steps on `device` within `tol_db` PSNR of `reference_steps`, at >= `min_steps_per_s`.

    Rows need device, method, num_steps, steps_per_s, and psnr (None when quality was not run).
    """
    out = {"method": method, "device": device, "reference_steps": reference_steps, "tol_db": tol_db,
           "min_steps_per_s": min_steps_per_s, "num_steps": None, "psnr": None, "steps_per_s": None,
           "reference_psnr": None, "reason": None}
    same = [r for r in rows if r["device"] == device and r["method"] == method]
    if not same:
        out["reason"] = f"{device} did not run {method}"
        return out
    ref = [r for r in same if r["num_steps"] == reference_steps and r.get("psnr") is not None]
    if not ref:
        out["reason"] = f"no PSNR for {method} with {reference_steps} steps on {device}"
        return out
    reference_psnr = ref[0]["psnr"]
    out["reference_psnr"] = reference_psnr
    ok = [r for r in same if r.get("psnr") is not None and r["steps_per_s"] >= min_steps_per_s
          and r["psnr"] >= reference_psnr - tol_db]
    if not ok:
        out["reason"] = (f"no {method} setting on {device} is within {tol_db} dB of {reference_steps} steps "
                         f"and at least {min_steps_per_s} frames/s")
        return out
    best = min(ok, key=lambda r: (r["num_steps"], -r["steps_per_s"]))
    out.update(num_steps=best["num_steps"], psnr=best["psnr"], steps_per_s=best["steps_per_s"])
    return out


def resolve_devices(names: list[str]) -> list[torch.device]:
    """Devices from the config that this machine can run. cuda and mps are skipped when absent."""
    found = []
    for name in names:
        if name == "cpu":
            found.append(torch.device("cpu"))
        elif name == "cuda" and torch.cuda.is_available():
            found.append(torch.device("cuda"))
        elif name == "cuda":
            print("skipping cuda: not available")
        elif name == "mps" and torch.backends.mps.is_available():
            found.append(torch.device("mps"))
        elif name == "mps":
            print("skipping mps: not available")
        else:
            raise ValueError(f"unknown device {name!r}; use cpu, cuda, or mps")
    if not found:
        raise RuntimeError(f"none of the requested devices are available: {list(names)}")
    return found


def parse_amp(value: str | None) -> torch.dtype | None:
    """`null`/`bf16`/`fp16` from the config into a dtype for `WorldModel.from_checkpoint`."""
    if value is None:
        return None
    table = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}
    if value not in table:
        raise ValueError(f"amp_dtype must be null, bf16, or fp16, got {value!r}")
    return table[value]


def pick_indices(total: int, n: int) -> np.ndarray:
    """`n` dataset indices spread evenly over `range(total)` (all of them if n >= total)."""
    if n >= total:
        return np.arange(total)
    return np.round(np.linspace(0, total - 1, n)).astype(np.int64)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def frames_per_second(model: WorldModel, context: torch.Tensor, actions: torch.Tensor, num_steps: int,
                      warmup: int, repeats: int, seed: int, cfg_scale: float | None, ctx_level: int) -> float:
    """Batch-1 generated frames per second. One call of `predict_next` is one frame."""
    if repeats < 1:
        raise ValueError(f"speed_repeats must be >= 1, got {repeats}")
    device = context.device

    def once(s: int) -> None:
        model.predict_next(context, actions, num_steps=num_steps, cfg_scale=cfg_scale, ctx_level=ctx_level,
                           generator=torch.Generator(device=device).manual_seed(s))

    for i in range(warmup):
        once(i)
    _sync(device)
    start = time.perf_counter()
    for i in range(repeats):
        once(seed + warmup + i)
    _sync(device)
    return repeats / (time.perf_counter() - start)


def score_windows(model: WorldModel, dataset: WindowDataset, indices: np.ndarray, batch_size: int,
                  num_workers: int, metric_names: list[str], num_steps: int, seed: int,
                  cfg_scale: float | None, ctx_level: int) -> dict[str, float | None]:
    """Mean teacher-forced metric on `indices`. Frames are mapped from [-1,1] to [0,1] for the metrics."""
    device = next(model.denoiser.parameters()).device
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, num_workers=num_workers)
    values = {name: [] for name in metric_names}
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        context, target = batch["context"].to(device), batch["target"].to(device)
        pred = model.predict_next(context, batch["actions"].to(device), num_steps=num_steps, cfg_scale=cfg_scale,
                                  ctx_level=ctx_level, generator=generator)
        pred01, tgt01 = (pred + 1) / 2, (target + 1) / 2
        for name in metric_names:
            values[name].append(METRICS[name](pred01, tgt01).detach().cpu())
    scored: dict[str, float | None] = {name: None for name in ("psnr", "lpips")}
    for name, parts in values.items():
        v = torch.cat(parts).double()
        std = float(v.std()) if len(v) > 1 else 0.0
        scored[name] = float(v.mean())
        scored[f"{name}_ci95"] = 1.96 * std / max(len(v), 1) ** 0.5
    return scored


def write_plot(path: Path, rows: list[dict]) -> None:
    """Quality against frames/s. Columns are devices, rows are PSNR and (when measured) LPIPS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("psnr", "PSNR (dB)")]
    if any(r.get("lpips") is not None for r in rows):
        metrics.append(("lpips", "LPIPS"))
    devices = list(dict.fromkeys(r["device"] for r in rows if r.get("psnr") is not None))
    fig, axes = plt.subplots(len(metrics), len(devices), figsize=(5.2 * len(devices), 3.8 * len(metrics)),
                             squeeze=False)
    for row_i, (key, ylabel) in enumerate(metrics):
        for col, device in enumerate(devices):
            ax = axes[row_i][col]
            for method, color in (("euler", "C0"), ("heun", "C1")):
                pts = [r for r in rows if r["device"] == device and r["method"] == method and r.get(key) is not None]
                if not pts:
                    continue
                ax.scatter([r["steps_per_s"] for r in pts], [r[key] for r in pts], label=method, color=color, s=40)
                for r in pts:
                    ax.annotate(str(r["num_steps"]), (r["steps_per_s"], r[key]), textcoords="offset points",
                                xytext=(5, 4), fontsize=8)
            ax.set_xlabel("frames / s (batch 1)")
            ax.set_ylabel(ylabel)
            ax.set_title(device)
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _format_table(rows: list[dict]) -> str:
    header = f"{'device':<8}{'method':<8}{'steps':>6}{'frames/s':>12}{'PSNR':>10}{'LPIPS':>10}"
    lines = [header]
    for r in rows:
        psnr = f"{r['psnr']:.3f}" if r.get("psnr") is not None else ""
        lpips = f"{r['lpips']:.4f}" if r.get("lpips") is not None else ""
        lines.append(f"{r['device']:<8}{r['method']:<8}{r['num_steps']:>6}{r['steps_per_s']:>12.3f}{psnr:>10}{lpips:>10}")
    return "\n".join(lines)


def _proposal_line(proposal: dict) -> str:
    if proposal["num_steps"] is None:
        return f"D11 proposal: none ({proposal['reason']})"
    return (f"D11 proposal: {proposal['method']} {proposal['num_steps']} steps on {proposal['device']} "
            f"({proposal['psnr']:.3f} dB, {proposal['steps_per_s']:.2f} frames/s; "
            f"{proposal['method']} {proposal['reference_steps']} steps = {proposal['reference_psnr']:.3f} dB)")


def main(argv: list[str] | None = None) -> Path:
    """Run the sweep and return the output directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if not cfg["model"]:
        raise SystemExit("set model to a checkpoint path (--set model=runs/<run>/checkpoints/ckpt_last.pt)")
    devices = resolve_devices(list(cfg["devices"]))
    amp = parse_amp(cfg["amp_dtype"])
    quality_n = int(cfg["quality_n"])
    metric_names = list(cfg["metrics"])
    unknown = sorted(set(metric_names) - set(METRICS))
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; choose from {sorted(METRICS)}")

    dataset: WindowDataset | None = None
    indices: np.ndarray | None = None
    rows: list[dict] = []
    for device in devices:
        model = WorldModel.from_checkpoint(cfg["model"], device=device, use_ema=bool(cfg["use_ema"]),
                                           compile=bool(cfg["compile"]), channels_last=bool(cfg["channels_last"]),
                                           amp_dtype=amp)
        if quality_n > 0 and dataset is None:
            data = cfg["data"]
            dataset = WindowDataset(data["dataset_dirs"], model.K, "val", grayscale=model.C == 1,
                                    val_fraction=float(data["val_fraction"]))
            if len(dataset) == 0:
                raise ValueError(f"val split is empty under {cfg['data']['dataset_dirs']}")
            target = dataset[0]["target"]
            if (model.C, model.H, model.W) != tuple(target.shape):
                raise ValueError(f"model frames are {(model.C, model.H, model.W)}, dataset frames are {tuple(target.shape)}")
            dataset.close()  # loader workers, and a num_workers=0 parent, reopen the shards
            indices = pick_indices(len(dataset), quality_n)
        context = torch.empty(1, model.K, model.C, model.H, model.W, device=device).uniform_(-1, 1)
        actions = torch.zeros(1, model.K, dtype=torch.long, device=device)
        for method in cfg["methods"]:
            model.sampler = dataclasses.replace(model.sampler, method=method)
            for steps in cfg["num_steps"]:
                steps = int(steps)
                fps = frames_per_second(model, context, actions, steps, int(cfg["speed_warmup"]),
                                        int(cfg["speed_repeats"]), int(cfg["seed"]), cfg["cfg_scale"],
                                        int(cfg["ctx_level"]))
                row = {"device": device.type, "method": method, "num_steps": steps, "steps_per_s": fps,
                       "psnr": None, "lpips": None}
                if indices is not None and dataset is not None:
                    row.update(score_windows(model, dataset, indices, int(cfg["data"]["batch_size"]),
                                             int(cfg["data"]["num_workers"]), metric_names, steps, int(cfg["seed"]),
                                             cfg["cfg_scale"], int(cfg["ctx_level"])))
                rows.append(row)
                print(f"{device.type} {method} N={steps}: {fps:.2f} frames/s", flush=True)

    proposal = propose_d11(rows, reference_steps=int(cfg["reference_steps"]), method=str(cfg["proposal_method"]),
                           tol_db=float(cfg["psnr_tol_db"]), min_steps_per_s=float(cfg["min_steps_per_s"]),
                           device=str(cfg["proposal_device"]))
    result = {"model": str(cfg["model"]), "channels_last": bool(cfg["channels_last"]), "amp_dtype": cfg["amp_dtype"],
              "compile": bool(cfg["compile"]), "use_ema": bool(cfg["use_ema"]), "seed": int(cfg["seed"]),
              "quality_n": quality_n, "n": 0 if indices is None else int(len(indices)),
              "metrics": metric_names, "rows": rows, "proposal": proposal}
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: "" if r.get(k) is None else r[k] for k in COLUMNS} for r in rows)
    if any(r.get("psnr") is not None for r in rows):
        write_plot(out_dir / "quality_vs_speed.png", rows)
    table = _format_table(rows)
    print(table)
    print(_proposal_line(proposal))
    print(f"Wrote {out_dir / 'results.json'}")
    return out_dir


if __name__ == "__main__":
    main()
