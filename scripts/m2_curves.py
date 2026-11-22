"""Export training and validation loss curves from TensorBoard logs to JSON (Milestone 2 learning curves).

    python scripts/m2_curves.py --runs r0=runs/<stamp>_m2_r0 final=runs/<stamp>_final --out results/milestone2/curves
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {"train": "train/loss", "val": "val/loss"}


def read_curves(run_dir: Path, bin_steps: int) -> dict:
    """Train loss averaged over bins of `bin_steps` steps (it is logged every 50 steps from one batch), val loss as logged."""
    acc = EventAccumulator(str(run_dir / "tb"), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for key, tag in TAGS.items():
        events = acc.Scalars(tag) if tag in acc.Tags()["scalars"] else []
        by_step = {e.step: e.value for e in events}  # a resumed run can log a step twice; keep the last value
        steps, values = np.array(sorted(by_step)), np.array([by_step[s] for s in sorted(by_step)])
        if key == "train" and len(steps):
            bins = (steps - 1) // bin_steps
            keep = np.unique(bins)
            steps = np.array([(b + 1) * bin_steps for b in keep])
            values = np.array([values[bins == b].mean() for b in keep])
        out[key] = {"step": steps.tolist(), "loss": values.tolist()}
    return out


def main(argv: list[str] | None = None) -> Path:
    """Write one <name>.json per run into --out."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="name=run_dir pairs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bin-steps", type=int, default=1000)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for pair in args.runs:
        name, run_dir = pair.split("=", 1)
        curves = read_curves(Path(run_dir), args.bin_steps)
        (args.out / f"{name}.json").write_text(json.dumps({"run_dir": run_dir, **curves}))
        print(f"{name}: {len(curves['train']['step'])} train points, {len(curves['val']['step'])} val points")
    return args.out


if __name__ == "__main__":
    main()
