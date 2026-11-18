"""Controllability: constant action programs on world models vs the real emulator (ROADMAP CP5-A).

    python scripts/eval_counterfactual.py --config configs/eval/counterfactual.yaml \
        --models copy_last runs/<run>/checkpoints/ckpt_last.pt --set idm=runs/idm/idm.pt

Each of n eval episodes gets one branch row t (player visible, no player laser in flight). From
there the real emulator (exact replay, then clone/restore) and every model run each program in
`wm.eval.counterfactual.PROGRAMS` for `horizon` steps. Reported per source (real and each model):
- ordering accuracy x_LEFT + m < x_NOOP < x_RIGHT - m at the last step, and mean dx per step;
- fire response: P(laser in the launch zone within `fire_window` steps), pooled over the three
  fire programs, and the same for the three no-fire programs (detector false positives);
- with `idm`, inverse dynamics accuracy on every consecutive frame pair, labelled by the program.
Writes <out_dir>/metrics.json and counterfactual.png (LEFT/NOOP/RIGHT strips, real vs model).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for eval_rollouts, also when imported by path

from eval_rollouts import build_models  # noqa: E402
from wm.config import load_config  # noqa: E402
from wm.data import ram as ram_map  # noqa: E402
from wm.data.reader import ShardReader, list_shards  # noqa: E402
from wm.eval import counterfactual as cf  # noqa: E402
from wm.eval import detectors  # noqa: E402
from wm.eval.baselines import CopyLastFrame  # noqa: E402
from wm.models.idm import IDM, action_targets, load_idm  # noqa: E402

FIRE_PROGRAMS = ["FIRE", "LEFTFIRE", "RIGHTFIRE"]
NO_FIRE_PROGRAMS = ["NOOP", "LEFT", "RIGHT"]
STRIP_PROGRAMS = ["LEFT", "NOOP", "RIGHT"]


def calibrate(dataset_dirs: list[str], chunk: int = 5000) -> dict:
    """player_x vs RAM player x over every stored frame, read in chunks."""
    xs, rams = [], []
    for path in sorted(p for d in dataset_dirs for p in list_shards(d)):
        reader = ShardReader(path)
        for s in range(0, reader.num_rows, chunk):
            rows = slice(s, min(s + chunk, reader.num_rows))
            xs.append(detectors.player_x(reader.read("frames", rows)))
            rams.append(reader.read("ram", rows))
        reader.close()
    return detectors.calibrate_player_x(np.concatenate(xs), np.concatenate(rams))


@torch.inference_mode()
def idm_accuracy(model: IDM, frames: np.ndarray, programs: list[int], device: torch.device, batch: int = 512) -> dict:
    """IDM accuracy on every pair (s_j, s_{j+1}) of frames (B,P,T+1,H,W,3), labelled with program P's action."""
    pairs = frames.shape[0] * frames.shape[1] * (frames.shape[2] - 1)
    f0 = frames[:, :, :-1].reshape(pairs, *frames.shape[3:])
    f1 = frames[:, :, 1:].reshape(pairs, *frames.shape[3:])
    labels = torch.tensor(programs).repeat_interleave(frames.shape[2] - 1).repeat(frames.shape[0])
    move_t, fire_t = action_targets(labels)
    hits = {"move": 0, "fire": 0}
    for i in range(0, pairs, batch):
        to = lambda u8: torch.from_numpy(u8[i:i + batch]).movedim(-1, -3).float().div(127.5).sub(1).to(device)  # noqa: E731
        move, fire = model(to(f0), to(f1))
        hits["move"] += int((move.argmax(1).cpu() == move_t[i:i + batch]).sum())
        hits["fire"] += int((fire.argmax(1).cpu() == fire_t[i:i + batch]).sum())
    return {"move": hits["move"] / pairs, "fire": hits["fire"] / pairs, "n": pairs}


def score(frames: np.ndarray, idle: np.ndarray, cfg: dict) -> tuple[dict, np.ndarray]:
    """Controllability metrics of frames (B,P,T+1,H,W,3), and the detected player x (B,P,T+1)."""
    flat = frames.reshape(-1, *frames.shape[3:])
    x = detectors.player_x(flat).reshape(frames.shape[:3])
    laser = detectors.laser_present(flat).reshape(frames.shape[:3])
    window = int(cfg["fire_window"])
    fire_p, fire_n = cf.fire_response(laser, idle, FIRE_PROGRAMS, window)
    ctrl_p, ctrl_n = cf.fire_response(laser, idle, NO_FIRE_PROGRAMS, window)
    return {"ordering_accuracy": cf.ordering_accuracy(x, float(cfg["margin_px"])),
            "dx_per_step": cf.mean_dx(x),
            "fire_response": {"p": fire_p, "n": fire_n},
            "no_fire_response": {"p": ctrl_p, "n": ctrl_n},
            "player_missing": float(np.isnan(x[:, :, 1:]).mean())}, x


def save_strip(path: Path, branches: list[cf.Branch], real: np.ndarray, gen: np.ndarray, tag: str,
               steps: list[int], n_branches: int) -> None:
    """LEFT/NOOP/RIGHT frames at `steps` for the first n branches: real rows above model rows."""
    rows = [(b, src, p) for b in range(min(n_branches, len(branches))) for src in ("real", tag) for p in STRIP_PROGRAMS]
    fig, axes = plt.subplots(len(rows), len(steps), figsize=(1.8 * len(steps), 1.0 * len(rows)), squeeze=False)
    for r, (b, src, prog) in enumerate(rows):
        frames = (real if src == "real" else gen)[b, cf.PROGRAMS.index(prog)]
        for c, s in enumerate(steps):
            ax = axes[r, c]
            ax.imshow(frames[s], interpolation="nearest", aspect=0.5)  # x2 horizontal widening, display only
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"step {s}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"t={branches[b].t}\n{src}\n{prog}", fontsize=6, rotation=0, ha="right", va="center")
    fig.tight_layout(pad=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--models", nargs="+", help="copy_last and/or checkpoint paths (overrides the config)")
    parser.add_argument("--n", type=int, help="number of branches (overrides the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    specs, n, horizon = args.models or list(cfg["models"]), args.n or int(cfg["n"]), int(cfg["horizon"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    programs = cf.program_actions()

    entries = build_models(specs, cfg, device)
    K = {m.K for _, _, m, _ in entries}
    if len(K) != 1:
        raise ValueError(f"all models must share K to share branches, got {sorted(K)}")
    K = K.pop()
    branches = cf.pick_branches(cfg["dataset_dirs"], n, K, horizon, int(cfg["min_step"]), int(cfg["branch_seed"]))
    H, W = branches[0].frames.shape[1:3]
    for tag, _, model, _ in entries:
        if isinstance(model, CopyLastFrame):  # its frame size is only known once the data is loaded
            model.H, model.W = H, W
        elif (model.H, model.W) != (H, W):
            raise ValueError(f"{tag}: model frames are {model.H}x{model.W}, dataset frames are {H}x{W}")
    idle = np.array([b.laser_idle for b in branches])
    idm = load_idm(cfg["idm"], device) if cfg["idm"] else None

    start = time.perf_counter()
    real_runs = [cf.real_branches(b, programs, horizon) for b in branches]
    real = np.stack([r[0] for r in real_runs])
    metrics, x_real = score(real, idle, cfg)
    results = {"real": {**metrics, "ended": int(sum(r[1].sum() for r in real_runs)), "seconds": time.perf_counter() - start}}
    if idm is not None:
        results["real"]["idm"] = idm_accuracy(idm, real, programs, device)

    strip = None
    for tag, spec, model, level in entries:
        start = time.perf_counter()
        generator = torch.Generator().manual_seed(int(cfg["seed"]))  # CPU, so results match across devices (D21)
        gen = cf.model_branches(model, branches, programs, horizon, int(cfg["batch_size"]), device=device,
                                num_steps=cfg["num_steps"], cfg_scale=cfg["cfg_scale"], ctx_level=level,
                                generator=generator)
        metrics, x = score(gen, idle, cfg)
        with np.errstate(invalid="ignore"):
            err = np.abs(x[:, :, -1] - x_real[:, :, -1])
        metrics["final_x_error_px"] = {p: float(np.nanmean(err[:, i])) if np.isfinite(err[:, i]).any() else float("nan")
                                       for i, p in enumerate(cf.PROGRAMS)}  # NaN: the model never shows the player
        if idm is not None:
            metrics["idm"] = idm_accuracy(idm, gen, programs, device)
        results[tag] = {"model": spec, "ctx_level": level, **metrics, "seconds": time.perf_counter() - start}
        if strip is None or not isinstance(model, CopyLastFrame):  # show a trained model when there is one
            strip = (tag, gen)

    for tag, r in results.items():
        idm_text = f", IDM move {r['idm']['move']:.3f} fire {r['idm']['fire']:.3f}" if "idm" in r else ""
        print(f"{tag} ({r['seconds']:.0f} s): ordering {r['ordering_accuracy']:.3f}, dx/step LEFT "
              f"{r['dx_per_step']['LEFT']:+.3f} NOOP {r['dx_per_step']['NOOP']:+.3f} RIGHT {r['dx_per_step']['RIGHT']:+.3f}, "
              f"fire {r['fire_response']['p']:.3f} (no-fire {r['no_fire_response']['p']:.3f}, n={r['fire_response']['n']})"
              + idm_text)

    calibration = calibrate(cfg["dataset_dirs"]) if cfg["calibrate"] else None
    if calibration:
        print(f"player_x vs RAM[{ram_map.PLAYER_X}]: x = {calibration['slope']:.4f} * ram + {calibration['intercept']:.3f}, "
              f"MAE {calibration['mae_px']:.3f} px over {calibration['n']:,} frames ({calibration['n_no_player']:,} without a player)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "config": cfg, "programs": list(cf.PROGRAMS), "device": str(device), "calibration": calibration,
        "branches": [{"shard": b.shard, "episode": b.episode, "t": b.t, "laser_idle": b.laser_idle} for b in branches],
        "results": results}, indent=2))
    save_strip(out_dir / "counterfactual.png", branches, real, strip[1], strip[0],
               [s for s in cfg["strip_steps"] if s <= horizon], int(cfg["strip_branches"]))
    print(f"Wrote {out_dir / 'metrics.json'} and counterfactual.png")
    return out_dir


if __name__ == "__main__":
    main()
