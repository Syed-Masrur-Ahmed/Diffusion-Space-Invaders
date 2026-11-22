"""Milestone 2 pipeline checks E1-E3 (look at the batch, initial loss, overfit one batch).

    python scripts/m2_checks.py e1 [--config configs/train/debug_aug.yaml] [--set data.root=...]
    python scripts/m2_checks.py e2 [--config configs/train/debug.yaml]
    python scripts/m2_checks.py e3 [--config configs/train/debug.yaml] [--steps 500]

Each check builds the real trainer (data, model, EDM loss) from a train config, so it tests the
pipeline the runs use. Outputs go to --out (default results/milestone2/<check>/): a JSON with the
numbers and PNG figures.
"""

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import default_collate

from wm.data.schema import ACTION_MEANINGS, NULL_ACTION
from wm.diffusion.edm import EDMConfig, EDMDenoiser, edm_coefficients
from wm.diffusion.sampler import sample
from wm.eval.metrics import psnr
from wm.train.trainer import Trainer, build_net, load_run_config

DEFAULT_CONFIGS = {"e1": "configs/train/debug_aug.yaml", "e2": "configs/train/debug.yaml",
                   "e3": "configs/train/debug.yaml"}
ACTION_NAMES = ACTION_MEANINGS + ["NULL"]  # index NULL_ACTION: action dropout (CFG token)
SCALE = 3  # display upscaling, nearest neighbour


# --- E1: look at the batch --------------------------------------------------------------------


def window_matches_shard(dataset, item: dict) -> bool:
    """True if the item's frames and actions equal a direct HDF5 read of its rows, all in one episode."""
    K, row = dataset.K, int(item["row"])
    with h5py.File(dataset.shard_paths[int(item["shard"])], "r") as f:
        frames = f["frames"][row - K:row + 1]
        actions = f["actions"][row - K:row]
        starts, lengths = f["episodes/start"][:], f["episodes/length"][:]
    e = int(np.searchsorted(starts, row, side="right")) - 1
    same_episode = starts[e] <= row - K and row < starts[e] + lengths[e]
    x = np.concatenate([item["context"].numpy(), item["target"].numpy()[None]])
    stored = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1
    return bool(same_episode and np.array_equal(stored.numpy(), x)
                and np.array_equal(actions, item["actions"].numpy()))


def _to_img(x: Tensor) -> np.ndarray:
    """(C,H,W) in [-1,1] (clamped: noisy context leaves the range) -> upscaled uint8 HWC."""
    img = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return img.repeat(SCALE, 0).repeat(SCALE, 1)


def save_batch_figure(path: Path, title: str, context: Tensor, actions: Tensor, target: Tensor,
                      ctx_level: Tensor, groups: int = 2) -> None:
    """Samples in `groups` side-by-side columns; each sample is K context frames (titled with the
    action taken at that frame) and the target, with its context noise level on the left."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, K = context.shape[:2]
    rows = math.ceil(n / groups)
    cols = groups * (K + 1) + groups - 1  # one narrow spacer column between groups
    widths = ([1] * (K + 1) + [0.25]) * groups
    fig, axes = plt.subplots(rows, cols, figsize=(1.45 * groups * (K + 1.25), 1.6 * rows),
                             gridspec_kw={"width_ratios": widths[:cols]})
    for ax in axes.flat:
        ax.axis("off")
    for i in range(n):
        r, g = i % rows, i // rows
        for c in range(K + 1):
            ax = axes[r, g * (K + 2) + c]
            ax.axis("on")
            ax.imshow(_to_img(context[i, c] if c < K else target[i]), interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color("#1f6f3f" if c == K else "#999")
            ax.set_title(f"{ACTION_NAMES[int(actions[i, c])]}" if c < K else "target", fontsize=7, pad=2,
                         color="#1f6f3f" if c == K else ("#b03a2e" if int(actions[i, c]) == NULL_ACTION else "black"))
        axes[r, g * (K + 2)].set_ylabel(f"level {int(ctx_level[i, 0])}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))  # room for the title
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run_e1(cfg, out: Path, n: int = 16, seed: int = 0) -> dict:
    """A training batch exactly as the model gets it (context noise, action dropout) and a val batch."""
    trainer = Trainer(cfg, None)
    batch = next(iter(trainer.train_loader))
    batch = {k: v[:n] for k, v in batch.items()}
    torch.manual_seed(seed)
    target, context, actions, ctx_level = trainer._inputs(batch, augment=True)
    val = default_collate([trainer.val_set[i] for i in np.linspace(0, len(trainer.val_set) - 1, n).astype(int)])
    v_target, v_context, v_actions, v_level = trainer._inputs(val, augment=False)

    save_batch_figure(out / "train_batch.png", f"Training batch as the model sees it: context noise (aug_max "
                      f"{cfg.train.aug_max}) and action dropout (p {cfg.train.p_action_drop}; NULL in red)", context.cpu(), actions.cpu(), target.cpu(),
                      ctx_level.cpu())
    save_batch_figure(out / "val_batch.png", "Validation batch: clean context (level 0), no action dropout", v_context.cpu(),
                      v_actions.cpu(), v_target.cpu(), v_level.cpu())
    train_ok = [window_matches_shard(trainer.train_set, {k: v[i] for k, v in batch.items()}) for i in range(n)]
    val_ok = [window_matches_shard(trainer.val_set, {k: v[i] for k, v in val.items()}) for i in range(n)]
    noise_std = (context - batch["context"].to(context.device)).flatten(2).std(-1)[:, 0].cpu()
    sigma_aug = ctx_level[:, 0].cpu().double() * cfg.train.aug_max / (cfg.model.n_levels - 1)
    return {
        "config": cfg.name, "n": n,
        "train_windows_match_shard": sum(train_ok), "val_windows_match_shard": sum(val_ok),
        "train_ctx_levels": ctx_level[:, 0].cpu().tolist(),
        "train_noise_std": noise_std.tolist(), "train_sigma_aug": sigma_aug.tolist(),
        "train_dropped_actions": int((actions == NULL_ACTION).all(1).sum()),
        "val_ctx_levels_all_zero": bool((v_level == 0).all()),
        "val_context_clean": bool(torch.equal(v_context.cpu(), val["context"])),
        "train_range": [float(context.min()), float(context.max())],
        "target_range": [float(target.min()), float(target.max())],
    }


# --- E2: initial loss -------------------------------------------------------------------------


def expected_initial_loss(m2: Tensor, sigma: Tensor, sigma_data: float) -> Tensor:
    """E[loss] for a network that outputs F = 0, per (sample, sigma): (m2 s^2 + sd^4) / (sd^2 (s^2 + sd^2)).

    With F = 0 the loss is E[F_target^2], F_target = (y - c_skip x) / c_out. `m2` is the per-sample
    mean of y^2. For data with E[y^2] = sigma_data^2 this is 1 at every sigma (EDM's design).
    """
    s2, d2 = sigma[None, :] ** 2, sigma_data**2
    return (m2[:, None] * s2 + d2**2) / (d2 * (s2 + d2))


def loss_at_sigma(denoiser: EDMDenoiser, target: Tensor, context: Tensor, actions: Tensor, sigma: float,
                  generator: torch.Generator) -> Tensor:
    """Per-sample EDM loss with every sample at the same `sigma` (fixed noise from `generator`)."""
    s = torch.full((len(target),), sigma, device=target.device)
    noise = torch.randn(target.shape, generator=generator).to(target.device)
    x = target + s.view(-1, 1, 1, 1) * noise
    c_skip, c_out, _, _ = (c.view(-1, 1, 1, 1) for c in edm_coefficients(s, denoiser.cfg.sigma_data))
    F, _, _ = denoiser._precondition(x, s, context, actions, torch.zeros_like(actions))
    return ((F - (target - c_skip * x) / c_out) ** 2).mean(dim=(1, 2, 3))


def run_e2(cfg, out: Path, draws: int = 64, seed: int = 0) -> dict:
    """Loss of a fresh model on one training batch vs. the F = 0 prediction from the batch's pixels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trainer = Trainer(cfg, None)
    batch = next(iter(trainer.train_loader))
    target, context, actions, _ = trainer._inputs(batch, augment=False)
    den = trainer.denoiser.eval()
    m2 = target.pow(2).flatten(1).mean(1).cpu().double()
    edm: EDMConfig = cfg.edm
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        torch.manual_seed(seed)
        first = den.loss(target, context, actions, torch.zeros_like(actions)).item()  # what the trainer logs
        draws_loss = []
        for _ in range(draws):
            draws_loss.append(den.loss(target, context, actions, torch.zeros_like(actions)).item())
        F, _, _ = den._precondition(target, torch.ones(len(target), device=target.device), context, actions,
                                    torch.zeros_like(actions))
        sig_grid = torch.logspace(math.log10(edm.sigma_min), math.log10(edm.sigma_max), 25)
        measured = [loss_at_sigma(den, target, context, actions, float(s), gen).mean().item() for s in sig_grid]
    # The training sigma distribution, sampled many times, for the expected batch loss.
    sig = (edm.p_mean + edm.p_std * torch.randn(200_000, generator=gen, dtype=torch.float64)).exp()
    sig = sig.clamp(edm.sigma_min, edm.sigma_max)
    expected = expected_initial_loss(m2, sig, edm.sigma_data).mean().item()
    theory_grid = expected_initial_loss(m2, sig_grid.double(), edm.sigma_data).mean(0).tolist()

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(sig_grid, theory_grid, "-", color="#3b6ea8", label=f"F = 0 prediction (E[y²] = {m2.mean():.3f})")
    ax.plot(sig_grid, measured, "o", color="#d0702c", ms=4, label="fresh model, measured")
    ax.axhline(1.0, color="#888", lw=0.8, ls="--", label="1 (data with E[y²] = σ_data² = 0.25)")
    ax.set_xscale("log"), ax.set_xlabel("noise level σ"), ax.set_ylabel("loss at init")
    ax.legend(fontsize=7), ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "initial_loss_vs_sigma.png", dpi=130)
    plt.close(fig)
    return {
        "config": cfg.name, "batch": len(target), "sigma_data": edm.sigma_data,
        "mean_y2": m2.mean().item(), "mean_y": target.mean().item(), "rms_y": m2.mean().sqrt().item(),
        "max_abs_F_at_init": F.abs().max().item(),
        "first_batch_loss": first, "loss_mean_over_draws": float(np.mean(draws_loss)),
        "loss_std_over_draws": float(np.std(draws_loss)), "draws": draws,
        "expected_loss_F0": expected, "expected_if_data_matched_sigma_data": 1.0,
        "sigma_grid": sig_grid.tolist(), "measured_by_sigma": measured, "expected_by_sigma": theory_grid,
    }


# --- E3: overfit one batch --------------------------------------------------------------------


def run_e3(cfg, out: Path, n: int = 32, steps: int = 500, lr: float = 1e-3, log_every: int = 50,
           seed: int = 0, device: str | None = None) -> dict:
    """Fresh model, n fixed train windows, Adam, no dropout/augmentation/weight decay, until memorized."""
    trainer = Trainer(cfg, None)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "mps" if torch.backends.mps.is_available() else "cpu"))
    idx = np.linspace(0, len(trainer.train_set) - 1, n).astype(int)
    batch = default_collate([trainer.train_set[int(i)] for i in idx])
    target, context, actions = (batch[k].to(dev) for k in ("target", "context", "actions"))
    level = torch.zeros_like(actions)
    torch.manual_seed(seed)
    net = build_net(cfg.model, target.shape[1], cfg.data.K, target.shape[2]).to(dev).train()
    den = EDMDenoiser(net, cfg.edm)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def fixed_loss() -> float:
        """Loss with the same 4 sigma/noise draws each time, so the curve is not sampling noise."""
        g = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            vals = []
            for _ in range(4):
                s = (cfg.edm.p_mean + cfg.edm.p_std * torch.randn(n, generator=g)).exp()
                s = s.clamp(cfg.edm.sigma_min, cfg.edm.sigma_max).to(dev)
                noise = torch.randn(target.shape, generator=g).to(dev)
                x = target + s.view(-1, 1, 1, 1) * noise
                c_skip, c_out, _, _ = (c.view(-1, 1, 1, 1) for c in edm_coefficients(s, cfg.edm.sigma_data))
                F, _, _ = den._precondition(x, s, context, actions, level)
                vals.append(((F - (target - c_skip * x) / c_out) ** 2).mean().item())
        return float(np.mean(vals))

    def sample_psnr() -> float:
        g = torch.Generator().manual_seed(seed)
        net.eval()
        with torch.no_grad():
            pred = sample(den, context, actions, level, cfg.sampler, generator=g)
        net.train()
        return psnr((pred.cpu() + 1) / 2, (target.cpu() + 1) / 2).mean().item()

    log = [{"step": 0, "train_loss": None, "fixed_loss": fixed_loss(), "psnr": sample_psnr()}]
    start = time.perf_counter()
    for step in range(1, steps + 1):
        loss = den.loss(target, context, actions, level)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0:
            log.append({"step": step, "train_loss": loss.item(), "fixed_loss": fixed_loss(), "psnr": sample_psnr()})
            print(f"step {step} train {loss.item():.4f} fixed {log[-1]['fixed_loss']:.4f} "
                  f"psnr {log[-1]['psnr']:.2f}", flush=True)
    seconds = time.perf_counter() - start

    net.eval()
    with torch.no_grad():
        pred = sample(den, context, actions, level, cfg.sampler, generator=torch.Generator().manual_seed(seed))
    rows = [np.concatenate([_to_img(context[i, -1].cpu()), _to_img(target[i].cpu()), _to_img(pred[i].cpu())], 1)
            for i in range(8)]
    iio.imwrite(out / "overfit_grid.png", np.concatenate(rows, 0))
    _plot_e3(out / "overfit_curve.png", log)
    return {"config": cfg.name, "n": n, "steps": steps, "lr": lr, "optimizer": "Adam", "device": str(dev),
            "seconds": seconds, "sampler_steps": cfg.sampler.num_steps, "log": log}


def _plot_e3(path: Path, log: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in log]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    a.plot(steps, [r["fixed_loss"] for r in log], "o-", ms=3, color="#3b6ea8", label="fixed draws")
    a.plot(steps[1:], [r["train_loss"] for r in log[1:]], ".", color="#d0702c", label="training step")
    a.axhline(0.01, color="#888", ls="--", lw=0.8, label="0.01")
    a.set_yscale("log"), a.set_xlabel("step"), a.set_ylabel("loss"), a.legend(fontsize=7), a.grid(alpha=0.3)
    b.plot(steps, [r["psnr"] for r in log], "o-", ms=3, color="#3b6ea8")
    b.set_xlabel("step"), b.set_ylabel("PSNR of samples vs targets (dB)"), b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["e1", "e2", "e3"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps", type=int, default=500, help="e3 only")
    args = parser.parse_args(argv)
    cfg = load_run_config(args.config or DEFAULT_CONFIGS[args.check], ["data.num_workers=0", *args.overrides])
    out = args.out or Path("results/milestone2") / args.check
    out.mkdir(parents=True, exist_ok=True)
    result = {"e1": lambda: run_e1(cfg, out), "e2": lambda: run_e2(cfg, out),
              "e3": lambda: run_e3(cfg, out, steps=args.steps)}[args.check]()
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("log", "sigma_grid", "measured_by_sigma",
                                                                    "expected_by_sigma")}, indent=1))


if __name__ == "__main__":
    main()
