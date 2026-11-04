"""Training loop, EMA, checkpointing, and resume (ROADMAP 4.12, 4.15). Owned by CP3-B; noise augmentation by CP4-B."""

import contextlib
import copy
import dataclasses
import functools
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Sampler, Subset, default_collate, get_worker_info

from wm.config import from_dict, load_config
from wm.data.dataset import WindowDataset
from wm.data.schema import NULL_ACTION
from wm.diffusion.edm import EDMConfig, EDMDenoiser, loss_by_sigma
from wm.diffusion.sampler import SamplerConfig, sample

CKPT_KEYS = ("model", "ema", "optimizer", "scheduler", "step", "config", "rng_states")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataSection:
    """Dataset and loader settings; dataset dirs are `root/<dir>` so the cluster can point at scratch."""

    root: str = "data"
    dirs: list[str] = field(default_factory=lambda: ["debug"])
    K: int = 4
    batch_size: int = 64
    num_workers: int = 6
    grayscale: bool = False
    val_fraction: float = 0.05
    in_memory: bool = False


@dataclass
class ModelSection:
    """Denoiser network: `unet` (4.7) or `tiny` (tests/fixtures.py::TinyDenoiserNet)."""

    arch: str = "unet"
    ch: list[int] = field(default_factory=lambda: [64, 128, 256])
    n_res: int = 2
    emb_dim: int = 256
    n_levels: int = 10
    attn_lowest: bool = False
    dropout: float = 0.0
    tiny_hidden: int = 16


@dataclass
class TrainSection:
    """Optimizer, schedule, EMA, conditioning, logging, and checkpoint settings."""

    steps: int = 50_000
    lr: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    lr_schedule: str = "constant"  # constant | cosine
    grad_clip: float = 1.0
    grad_accum: int = 1
    ema_decay: float = 0.999
    p_action_drop: float = 0.1
    aug_max: float = 0.0  # CP4-B
    ctx_noise_mode: str = "shared"  # CP4-B
    compile: bool = False
    allow_mps: bool = False
    runs_dir: str = "runs"
    log_every: int = 50
    val_every: int = 1000
    val_windows: int = 2048
    sample_every: int = 2500
    n_samples: int = 8
    save_every: int = 5000
    keep_last: int = 3


@dataclass
class RunConfig:
    """Full training config (configs/train/*.yaml)."""

    name: str
    seed: int
    data: DataSection
    model: ModelSection
    edm: EDMConfig
    sampler: SamplerConfig
    train: TrainSection


def load_run_config(path: str | Path, overrides: list[str] = ()) -> RunConfig:
    """Load a train YAML with `--set` overrides into a RunConfig."""
    cfg = from_dict(RunConfig, load_config(path, overrides))
    if cfg.model.arch not in ("unet", "tiny"):
        raise ValueError(f"model.arch must be 'unet' or 'tiny', got {cfg.model.arch!r}")
    if cfg.train.lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"train.lr_schedule must be 'constant' or 'cosine', got {cfg.train.lr_schedule!r}")
    return cfg


def drop_actions(actions: torch.LongTensor, p: float) -> torch.LongTensor:
    """Replace each sample's whole (K,) action sequence with NULL_ACTION with probability p (4.8)."""
    if p <= 0:
        return actions
    drop = torch.rand(actions.shape[0]) < p  # global CPU RNG, saved in rng_states
    return torch.where(drop[:, None].to(actions.device), torch.full_like(actions, NULL_ACTION), actions)


class EpochBatchSampler(Sampler[list[int]]):
    """Shuffled batches with drop_last; epoch e uses randperm seeded by seed + e, so any position is replayable."""

    def __init__(self, n: int, batch_size: int, seed: int) -> None:
        self.n, self.batch_size, self.seed = n, batch_size, seed
        self.epoch = self.skip = 0

    def set_position(self, epoch: int, skip: int) -> None:
        """Make the next iteration yield epoch `epoch` from batch `skip` onwards."""
        self.epoch, self.skip = epoch, skip

    def __len__(self) -> int:
        return self.n // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for b in range(self.skip, len(self)):
            yield perm[b * self.batch_size:(b + 1) * self.batch_size]


def _worker_init(seed: int, worker_id: int) -> None:
    """Seed a DataLoader worker, drop inherited HDF5 handles, and leave SIGUSR1 to the main process."""
    # SIGTERM keeps its default: torch's shutdown falls back to terminate() for workers that
    # don't exit in time (persistent workers with an open HDF5 file do not; see progress note).
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    s = seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    ds = get_worker_info().dataset
    getattr(ds, "dataset", ds).close()  # Subset wraps the WindowDataset


def _loader(dataset, seed: int, num_workers: int, **kwargs) -> DataLoader:
    # A dedicated generator, because each new iterator draws its base seed from it, and the
    # global torch RNG (sigma, eps, action dropout) must not depend on when iterators are made.
    return DataLoader(dataset, num_workers=num_workers, persistent_workers=num_workers > 0,
                      pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed),
                      worker_init_fn=functools.partial(_worker_init, seed), **kwargs)


def pick_device(allow_mps: bool) -> torch.device:
    """cuda, then mps (only if allowed), then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_net(cfg: ModelSection, C: int, K: int, H: int) -> nn.Module:
    """The denoiser network F named by `cfg.arch`."""
    if cfg.arch == "unet":
        from wm.models.unet import UNet, UNetConfig

        return UNet(UNetConfig(C=C, K=K, H=H, ch=list(cfg.ch), n_res=cfg.n_res, emb_dim=cfg.emb_dim,
                               n_levels=cfg.n_levels, attn_lowest=cfg.attn_lowest, dropout=cfg.dropout))
    if str(REPO_ROOT) not in sys.path:  # tests/ is not part of the installed package
        sys.path.insert(0, str(REPO_ROOT))
    from tests.fixtures import TinyDenoiserNet

    return TinyDenoiserNet(C=C, K=K, n_levels=cfg.n_levels, hidden=cfg.tiny_hidden)


def lr_factor(step: int, warmup: int, total: int, schedule: str) -> float:
    """Linear warmup over `warmup` steps, then constant or cosine decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def rng_states() -> dict:
    """Python, numpy, torch CPU and (if present) CUDA RNG states."""
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_states(states: dict) -> None:
    """Restore the states saved by `rng_states`."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextlib.contextmanager
def _eval_mode(module: nn.Module):
    """eval() for the block, then restore the module's previous mode."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def _to_uint8(x: Tensor) -> Tensor:
    """(N,C,H,W) in [-1,1] -> (N,3,H,W) uint8; greyscale is repeated to 3 channels."""
    x = ((x.float().cpu() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x


def _save_grid(path: Path, last_context: Tensor, target: Tensor, pred: Tensor, scale: int = 4) -> None:
    """One row per sample: last context | target | prediction, nearest-upscaled, saved as RGB PNG."""
    cols = [_to_uint8(t) for t in (last_context, target, pred)]
    grid = torch.cat([torch.cat([c[n] for c in cols], dim=2) for n in range(len(target))], dim=1)
    grid = grid.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
    iio.imwrite(path, grid.permute(1, 2, 0).numpy())


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _evenly_spaced(n: int, k: int) -> list[int]:
    return sorted(set(np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist())) if n else []


class Trainer:
    """Owns the data, the model and its EMA copy, the optimizer, and one run directory (or none)."""

    def __init__(self, cfg: RunConfig, run_dir: Path | None) -> None:
        self.cfg, self.run_dir = cfg, run_dir
        d, t = cfg.data, cfg.train
        seed_everything(cfg.seed)
        self.device = pick_device(t.allow_mps)
        dirs = [Path(d.root) / x for x in d.dirs]
        self.train_set = WindowDataset(dirs, d.K, "train", d.grayscale, d.val_fraction, cfg.seed, in_memory=d.in_memory)
        self.val_set = WindowDataset(dirs, d.K, "val", d.grayscale, d.val_fraction, cfg.seed, in_memory=d.in_memory)
        self.sampler = EpochBatchSampler(len(self.train_set), d.batch_size, cfg.seed)
        if len(self.sampler) == 0:
            raise ValueError(f"{len(self.train_set)} train windows is less than one batch of {d.batch_size}")
        self.train_loader = _loader(self.train_set, cfg.seed, d.num_workers, batch_sampler=self.sampler)
        self.val_loader = None
        if len(self.val_set):
            val = Subset(self.val_set, _evenly_spaced(len(self.val_set), t.val_windows))
            self.val_loader = _loader(val, cfg.seed, min(2, d.num_workers), batch_size=d.batch_size)
        # Fixed windows for the sample grids, read once here (then closed: no handle stays open).
        grid_set = self.val_set if len(self.val_set) else self.train_set
        self.grid_batch = default_collate([grid_set[i] for i in _evenly_spaced(len(grid_set), t.n_samples)])
        grid_set.close()

        C, H = self.grid_batch["target"].shape[1:3]
        self.net = build_net(cfg.model, C, d.K, H).to(self.device)
        print(f"model {cfg.model.arch}: {sum(p.numel() for p in self.net.parameters()):,} parameters on {self.device}")
        self.ema_net = copy.deepcopy(self.net).requires_grad_(False).eval()
        self.denoiser = EDMDenoiser(torch.compile(self.net) if t.compile else self.net, cfg.edm)
        self.ema_denoiser = EDMDenoiser(self.ema_net, cfg.edm)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=t.lr, betas=tuple(t.betas), weight_decay=t.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_factor(s, t.warmup_steps, t.steps, t.lr_schedule))
        self.step = 0
        bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.autocast = (functools.partial(torch.autocast, "cuda", dtype=torch.bfloat16) if bf16
                         else contextlib.nullcontext)
        self.tb = None
        if run_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(run_dir / "tb"))

    def _inputs(self, batch: dict, drop: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(target, context, actions, ctx_level) on the device; action dropout only when `drop`."""
        actions = drop_actions(batch["actions"], self.cfg.train.p_action_drop) if drop else batch["actions"]
        target, context, actions = (x.to(self.device, non_blocking=True)
                                    for x in (batch["target"], batch["context"], actions))
        return target, context, actions, torch.zeros_like(actions)  # ctx noise is CP4-B

    def train_step(self, batches: list[dict]) -> dict:
        """One optimizer step over `batches` (gradient accumulation), then the EMA update."""
        self.net.train()
        self.opt.zero_grad(set_to_none=True)
        pers, sigmas = [], []
        for batch in batches:
            with self.autocast():
                per, sigma = self.denoiser.loss_terms(*self._inputs(batch, drop=True))
            (per.float().mean() / len(batches)).backward()
            pers.append(per.detach().float())
            sigmas.append(sigma.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.train.grad_clip)
        self.opt.step()
        self.sched.step()
        self.step += 1
        self._update_ema()
        per = torch.cat(pers)
        return {"loss": per.mean().item(), "grad_norm": grad_norm.item(), "per": per, "sigma": torch.cat(sigmas)}

    @torch.no_grad()
    def _update_ema(self) -> None:
        # During warmup the EMA copies the weights exactly; afterwards it decays every step.
        warm = self.step <= self.cfg.train.warmup_steps
        for e, p in zip(self.ema_net.parameters(), self.net.parameters()):
            e.copy_(p) if warm else e.lerp_(p, 1 - self.cfg.train.ema_decay)
        for e, b in zip(self.ema_net.buffers(), self.net.buffers()):
            e.copy_(b)

    def eval_loss(self, batch: dict, seed: int = 0) -> float:
        """Loss of the training model on `batch` with fixed sigma and noise draws and no action dropout."""
        cuda = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda), _eval_mode(self.denoiser), torch.inference_mode(), self.autocast():
            torch.manual_seed(seed)
            per, _ = self.denoiser.loss_terms(*self._inputs(batch, drop=False))
        return per.float().mean().item()

    def val_loss(self) -> float | None:
        """Mean loss over the fixed val windows, or None if the val split is empty."""
        if self.val_loader is None:
            return None
        total = n = 0
        for i, batch in enumerate(self.val_loader):
            total += self.eval_loss(batch, seed=i) * len(batch["target"])
            n += len(batch["target"])
        return total / n

    def sample_grid_predictions(self) -> Tensor:
        """EMA-model samples for the fixed grid windows, default sampler, fixed generator seed."""
        target, context, actions, ctx_level = self._inputs(self.grid_batch, drop=False)
        gen = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        with _eval_mode(self.ema_denoiser), torch.inference_mode(), self.autocast():
            return sample(self.ema_denoiser, context, actions, ctx_level, self.cfg.sampler, generator=gen).float()

    def save_samples(self) -> Path:
        path = self.run_dir / "samples" / f"step_{self.step:07d}.png"
        g = self.grid_batch
        _save_grid(path, g["context"][:, -1], g["target"], self.sample_grid_predictions())
        return path

    def checkpoint(self) -> dict:
        """The 4.12 checkpoint dict; `model` and `ema` are the network state dicts."""
        return {"model": self.net.state_dict(), "ema": self.ema_net.state_dict(),
                "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(), "step": self.step,
                "config": dataclasses.asdict(self.cfg), "rng_states": rng_states()}

    def load(self, ckpt: dict) -> None:
        """Restore everything saved by `checkpoint`, including the RNG states."""
        self.net.load_state_dict(ckpt["model"])
        self.ema_net.load_state_dict(ckpt["ema"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.sched.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]
        set_rng_states(ckpt["rng_states"])

    def save(self, step_file: bool) -> None:
        """Write ckpt_last.pt atomically; with `step_file`, also ckpt_<step>.pt, pruned to keep_last."""
        ckdir = self.run_dir / "checkpoints"
        last, tmp = ckdir / "ckpt_last.pt", ckdir / "ckpt_last.pt.tmp"
        torch.save(self.checkpoint(), tmp)
        os.replace(tmp, last)
        if step_file:
            shutil.copyfile(last, tmp)
            os.replace(tmp, ckdir / f"ckpt_{self.step:07d}.pt")
            for old in sorted(ckdir.glob("ckpt_[0-9]*.pt"))[:-self.cfg.train.keep_last]:
                old.unlink()

    def batches(self) -> Iterator[dict]:
        """Train batches forever, starting at the position implied by `self.step` (exact resume)."""
        consumed = self.step * self.cfg.train.grad_accum
        while True:
            self.sampler.set_position(*divmod(consumed, len(self.sampler)))
            for batch in self.train_loader:
                consumed += 1
                yield batch

    def fit(self, stop: Callable[[], str | None]) -> int:
        """Train to `train.steps`, logging, validating, sampling, and saving; returns the exit code."""
        t = self.cfg.train
        batches = self.batches()
        window, n_seen, t0 = [], 0, time.perf_counter()
        while self.step < t.steps:
            out = self.train_step([next(batches) for _ in range(t.grad_accum)])
            window.append(out)
            n_seen += len(out["per"])
            s = self.step
            if s % t.log_every == 0:
                rate = n_seen / (time.perf_counter() - t0)
                lr = self.sched.get_last_lr()[0]
                loss = sum(o["loss"] for o in window) / len(window)
                print(f"step {s} loss {loss:.4f} lr {lr:.2e} grad_norm {out['grad_norm']:.3f} "
                      f"samples/s {rate:.0f}", flush=True)
                buckets = loss_by_sigma(torch.cat([o["per"] for o in window]), torch.cat([o["sigma"] for o in window]))
                for name, value in [("loss", loss), ("lr", lr), ("grad_norm", out["grad_norm"]),
                                    ("samples_per_s", rate)]:
                    self.tb.add_scalar(f"train/{name}", value, s)
                for bucket, value in buckets.items():
                    self.tb.add_scalar(f"loss_by_sigma/{bucket}", value, s)
                window, n_seen, t0 = [], 0, time.perf_counter()
            if s % t.val_every == 0 and (val := self.val_loss()) is not None:
                print(f"step {s} val_loss {val:.4f}", flush=True)
                self.tb.add_scalar("val/loss", val, s)
            if s % t.sample_every == 0:
                print(f"step {s} samples {self.save_samples()}", flush=True)
            if s % t.save_every == 0:
                self.save(step_file=True)
            if (sig := stop()) is not None:
                if s % t.save_every:
                    self.save(step_file=False)
                print(f"caught {sig}: saved ckpt_last.pt at step {s}, exiting", flush=True)
                self.tb.close()
                return 0
        if self.step % t.save_every:
            self.save(step_file=False)
        self.tb.close()
        print(f"finished at step {self.step}", flush=True)
        return 0


def find_auto_resume(runs_dir: Path, name: str) -> Path | None:
    """Most recent `runs_dir/<stamp>_<name>` that has checkpoints/ckpt_last.pt, or None."""
    if not runs_dir.is_dir():
        return None

    def stamp_key(d: Path) -> tuple:
        date, hm, *n = d.name.partition("_")[0].split("-")
        return date, hm, int(n[0]) if n else 0

    runs = [d for d in runs_dir.iterdir()
            if d.name.partition("_")[2] == name and (d / "checkpoints/ckpt_last.pt").is_file()]
    return max(runs, key=stamp_key, default=None)


def new_run_dir(runs_dir: Path, name: str) -> Path:
    """Create runs_dir/<YYYYmmdd-HHMM>[-n]_<name>/ with tb/, checkpoints/, samples/."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir, n = runs_dir / f"{stamp}_{name}", 0
    while run_dir.exists():
        n += 1
        run_dir = runs_dir / f"{stamp}-{n}_{name}"
    for sub in ("tb", "checkpoints", "samples"):
        (run_dir / sub).mkdir(parents=True)
    return run_dir


class StopOnSignal:
    """Records SIGUSR1/SIGTERM in a flag (the handler does nothing else); call the object to read it."""

    SIGNALS = (signal.SIGUSR1, signal.SIGTERM)

    def __enter__(self) -> "StopOnSignal":
        self.caught: str | None = None
        self._previous = {s: signal.signal(s, self._handle) for s in self.SIGNALS}
        return self

    def _handle(self, signum: int, frame) -> None:
        self.caught = signal.Signals(signum).name

    def __call__(self) -> str | None:
        return self.caught

    def __exit__(self, *exc) -> None:
        for s, handler in self._previous.items():
            signal.signal(s, handler)


def run(cfg: RunConfig, resume: str | None = None, dry_run: bool = False) -> int:
    """Train per `cfg`. `resume` is None (new run), a run dir, or "auto"; returns the exit code."""
    if dry_run:  # 5 steps, one val pass, one sample batch; writes nothing
        trainer = Trainer(cfg, run_dir=None)
        batches = trainer.batches()
        for _ in range(5):
            out = trainer.train_step([next(batches) for _ in range(cfg.train.grad_accum)])
        val, pred = trainer.val_loss(), trainer.sample_grid_predictions()
        print(f"dry run: 5 steps, last loss {out['loss']:.4f}, val loss {val}, samples {tuple(pred.shape)}")
        return 0

    runs_dir = Path(cfg.train.runs_dir)
    run_dir = find_auto_resume(runs_dir, cfg.name) if resume == "auto" else (Path(resume) if resume else None)
    if run_dir is not None and not (run_dir / "checkpoints/ckpt_last.pt").is_file():
        raise FileNotFoundError(f"no checkpoints/ckpt_last.pt in {run_dir}")
    fresh = run_dir is None
    run_dir = new_run_dir(runs_dir, cfg.name) if fresh else run_dir
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")

    trainer = Trainer(cfg, run_dir)
    if fresh:
        print(f"new run {run_dir}", flush=True)
    else:
        trainer.load(torch.load(run_dir / "checkpoints/ckpt_last.pt", map_location="cpu", weights_only=False))
        print(f"resumed {run_dir} at step {trainer.step}", flush=True)
    with StopOnSignal() as stop:
        return trainer.fit(stop)
