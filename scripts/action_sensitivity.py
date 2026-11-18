"""How much a world model moves when the next action flips from LEFT to RIGHT (ROADMAP CP5-C).

For val windows, predict the next frame twice under a shared noise seed: once with the last
action set to LEFT, once to RIGHT. The score is the mean absolute difference, in 8-bit pixel
units, over the player band. Copy-last scores 0, since it ignores the action.

    python scripts/action_sensitivity.py --config configs/eval/action_sensitivity.yaml \
        --set model=runs/<run>/checkpoints/ckpt_last.pt

Writes <out_dir>/<model tag>/sensitivity.json, sensitivity.csv, and grid_w<w>.png.
Each grid is last context | LEFT | RIGHT | |LEFT-RIGHT|x4. The proposal is the smallest w
that reaches `gain` of the sensitivity rise from the smallest w. Check the grids before
adopting it: a higher w can add artifacts that this score does not see.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from wm.config import load_config
from wm.data.dataset import WindowDataset
from wm.data.schema import ACTION_MEANINGS
from wm.eval.baselines import CopyLastFrame
from wm.eval.visualize import save_grid
from wm.inference import WorldModel

LEFT = ACTION_MEANINGS.index("LEFT")
RIGHT = ACTION_MEANINGS.index("RIGHT")
_PLAYER_RAW_ROWS = (185, 195)  # coverage_report.py: the green player, not the score digits


def with_last_action(actions: Tensor, action: int) -> Tensor:
    """Copy of (B, K) actions with the last step, the one that produces the next frame, set to `action`."""
    out = actions.clone()
    out[:, -1] = action
    return out


def player_band(height: int, raw_rows: tuple[int, int] = _PLAYER_RAW_ROWS, raw_height: int = 210) -> tuple[int, int]:
    """Rows [top, bottom) of a model frame covering the player sprite on the raw screen."""
    top = raw_rows[0] * height // raw_height
    bottom = math.ceil(raw_rows[1] * height / raw_height)
    if not 0 <= top < bottom <= height:
        raise ValueError(f"player band [{top}, {bottom}) is outside a frame of height {height}")
    return top, bottom


def band_mae(left: Tensor, right: Tensor, band: tuple[int, int]) -> Tensor:
    """Per-sample mean |left - right| in 8-bit pixel units over rows [top, bottom). Shape (B,)."""
    top, bottom = band
    return (left - right).abs()[:, :, top:bottom].flatten(1).mean(1) * 127.5


def propose_cfg_weight(rows: list[dict], gain: float = 0.8) -> dict:
    """Smallest w that reaches `gain` of the sensitivity rise from the smallest w to the best.

    `rows` are {"w": float, "sensitivity": float}. No rise keeps the smallest w and sets `reason`.
    """
    out = {"w": None, "sensitivity": None, "baseline": None, "best": None, "best_w": None,
           "target": None, "reason": None}
    if not rows:
        out["reason"] = "no rows"
        return out
    ordered = sorted(rows, key=lambda r: float(r["w"]))
    baseline = float(ordered[0]["sensitivity"])
    best = max(rows, key=lambda r: float(r["sensitivity"]))
    best_s = float(best["sensitivity"])
    out.update(baseline=baseline, best=best_s, best_w=float(best["w"]))
    if best_s <= baseline:
        out.update(w=float(ordered[0]["w"]), sensitivity=baseline, reason="sensitivity does not rise with w")
        return out
    target = baseline + gain * (best_s - baseline)
    hit = next(r for r in ordered if float(r["sensitivity"]) >= target - 1e-9)
    out.update(w=float(hit["w"]), sensitivity=float(hit["sensitivity"]), target=target)
    return out


def _evenly_spaced(total: int, n: int) -> np.ndarray:
    """`n` indices spread over `range(total)`, or every index when n is at least total."""
    if n >= total:
        return np.arange(total)
    return np.round(np.linspace(0, total - 1, n)).astype(np.int64)


def _predict_pair(model, context: Tensor, actions: Tensor, cfg_scale: float | None, num_steps: int | None,
                  ctx_level: int, seed: int) -> tuple[Tensor, Tensor]:
    """LEFT and RIGHT predictions. The generator is re-seeded so both calls draw the same noise."""
    kwargs = dict(num_steps=num_steps, cfg_scale=cfg_scale, ctx_level=ctx_level)
    left = model.predict_next(context, with_last_action(actions, LEFT), generator=torch.Generator().manual_seed(seed),
                              **kwargs)
    right = model.predict_next(context, with_last_action(actions, RIGHT),
                               generator=torch.Generator().manual_seed(seed), **kwargs)
    return left, right


def _window_scores(model, dataset: WindowDataset, indices: np.ndarray, batch_size: int, device: torch.device,
                   band: tuple[int, int], cfg_scale: float | None, num_steps: int | None, ctx_level: int,
                   seed: int) -> Tensor:
    """Per-window player-band MAE, in the order of `indices`."""
    parts = []
    for start in range(0, len(indices), batch_size):
        chosen = [dataset[int(i)] for i in indices[start:start + batch_size]]
        context = torch.stack([item["context"] for item in chosen]).to(device)
        actions = torch.stack([item["actions"] for item in chosen]).to(device)
        left, right = _predict_pair(model, context, actions, cfg_scale, num_steps, ctx_level, seed)
        parts.append(band_mae(left, right, band).cpu())
    return torch.cat(parts)


def _summarize(model: str, w: float | None, values: Tensor) -> dict:
    v = values.double()
    std = v.std().item() if len(v) > 1 else 0.0
    return {"model": model, "w": w, "sensitivity": v.mean().item(), "std": std, "n": len(v)}


def _w_tag(w: float) -> str:
    return f"{w:g}"


def _model_tag(spec: str) -> str:
    path = Path(spec)
    return f"{path.parent.parent.name}_{path.stem}"


def _action_input(model: WorldModel) -> str:
    cfg = getattr(model.denoiser.net, "cfg", None)
    return getattr(cfg, "action_input", "adagn")


def run(cfg: dict, device: torch.device) -> dict:
    """Sensitivity rows, the D11 proposal, and LEFT/RIGHT grids for the first `grid_n` windows."""
    spec = str(cfg["model"])
    model = WorldModel.from_checkpoint(spec, device=device, use_ema=bool(cfg["use_ema"]),
                                       channels_last=bool(cfg["channels_last"]))
    data = cfg["data"]
    if int(data["K"]) != model.K:
        print(f"using checkpoint K={model.K} (config data.K={data['K']})")
    if bool(data["grayscale"]) != (model.C == 1):
        raise ValueError(f"data.grayscale is {data['grayscale']} but the checkpoint has C={model.C}")
    dataset = WindowDataset(list(data["dataset_dirs"]), model.K, "val", grayscale=model.C == 1,
                            val_fraction=float(data["val_fraction"]))
    try:
        if len(dataset) == 0:
            raise ValueError(f"no val windows in {list(data['dataset_dirs'])}")
        frame = dataset[0]["target"]
        if tuple(frame.shape) != (model.C, model.H, model.W):
            raise ValueError(f"checkpoint frames are {(model.C, model.H, model.W)}, data frames are {tuple(frame.shape)}")
        n = min(int(cfg["n"]), len(dataset))
        indices = _evenly_spaced(len(dataset), n)
        band = player_band(model.H, tuple(cfg["band_raw"]), int(cfg["raw_height"]))
        batch_size = int(data["batch_size"])
        num_steps, ctx_level, seed = cfg["num_steps"], int(cfg["ctx_level"]), int(cfg["seed"])
        rows = []
        if cfg["include_copy_last"]:
            baseline = CopyLastFrame(model.K, model.C, model.H, model.W)
            values = _window_scores(baseline, dataset, indices, batch_size, device, band, None, num_steps, ctx_level, seed)
            rows.append(_summarize("copy_last", None, values))
        weights = [float(w) for w in cfg["weights"]]
        for w in weights:
            values = _window_scores(model, dataset, indices, batch_size, device, band, w, num_steps, ctx_level, seed)
            rows.append(_summarize(spec, w, values))
        copy_last = next((r["sensitivity"] for r in rows if r["model"] == "copy_last"), 0.0)
        proposal = propose_cfg_weight([r for r in rows if r["w"] is not None], float(cfg["gain"]))
        best = proposal["best"] if proposal["best"] is not None else copy_last
        proposal["ignores_actions"] = best <= copy_last + float(cfg["ignore_below"])
        proposal["check_grids"] = proposal["w"] is not None
        if proposal["ignores_actions"]:
            proposal["w"] = None
            proposal["check_grids"] = False
            proposal["reason"] = ("sensitivity stays within ignore_below of copy-last; "
                                  "train with action_input=adagn+planes")
        grids = _grids(model, dataset, indices[:int(cfg["grid_n"])], device, weights, num_steps, ctx_level, seed)
    finally:
        dataset.close()
    return {"model": spec, "action_input": _action_input(model), "n": int(n), "band": list(band),
            "band_raw": list(cfg["band_raw"]), "num_steps": num_steps, "ctx_level": ctx_level, "seed": seed,
            "device": str(device), "rows": rows, "proposal": proposal, "grids": grids}


def _grids(model, dataset: WindowDataset, indices: np.ndarray, device: torch.device, weights: list[float],
           num_steps: int | None, ctx_level: int, seed: int) -> dict[str, dict[str, Tensor]]:
    """LEFT/RIGHT predictions for each w, keyed by the grid filename stem."""
    if len(indices) == 0:
        return {}
    chosen = [dataset[int(i)] for i in indices]
    context = torch.stack([item["context"] for item in chosen]).to(device)
    actions = torch.stack([item["actions"] for item in chosen]).to(device)
    grids = {}
    for w in weights:
        left, right = _predict_pair(model, context, actions, w, num_steps, ctx_level, seed)
        grids[f"grid_w{_w_tag(w)}"] = {"context": context.cpu(), "left": left.cpu(), "right": right.cpu()}
    return grids


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--model", help="checkpoint path (overrides the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.model:
        cfg["model"] = args.model
    if not cfg["model"]:
        raise ValueError("set model to a checkpoint path")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run(cfg, device)
    grids = result.pop("grids")
    out = Path(cfg["out_dir"]) / _model_tag(str(cfg["model"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(result, indent=2))
    with (out / "sensitivity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "w", "sensitivity", "std", "n"])
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({**row, "w": "" if row["w"] is None else row["w"]})
    for name, grid in grids.items():
        save_grid(out / f"{name}.png", grid["context"], grid["left"], grid["right"])
    for row in result["rows"]:
        label = "copy" if row["w"] is None else _w_tag(row["w"])
        print(f"{label:>6}  {row['sensitivity']:.4f} ± {row['std']:.4f}  (n={row['n']})")
    proposal = result["proposal"]
    print(f"proposal w={proposal['w']} ignores_actions={proposal['ignores_actions']} "
          f"({proposal['reason'] or 'check the grids before adopting'})")
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
