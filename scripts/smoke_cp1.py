"""CP1 integration smoke test: the epsilon-mix agent plays short episodes into a shard.

Writes `<out_dir>/shard_w000_00000.h5` at the D10 resolution and `<out_dir>/frames.png`,
a grid of evenly spaced stored frames upscaled 4x nearest (no aspect correction), for the
visual check. Then run `python scripts/validate_data.py <out_dir>`.

    python scripts/smoke_cp1.py
    python scripts/smoke_cp1.py --set out_dir=data/smoke_cp1b --set episodes=1
"""

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import yaml

from wm.agents.policies import EpsilonMixPolicy
from wm.agents.pretrained import ZooAgentPolicy
from wm.config import load_config
from wm.data.schema import END_TERMINATED, END_TRUNCATED, SOURCE_AGENT
from wm.data.writer import ShardWriter
from wm.env import make_env, to_model_frame

DEFAULTS = {
    "out_dir": "data/smoke_cp1",
    "episodes": 2,
    "seed": 0,
    "max_steps": 500,
    "epsilon": 0.2,
    "resolution": 84,
    "resize_method": "max",
    "grid_frames": 20,
}


def collect(cfg: dict) -> tuple[Path, int, float]:
    """Play `cfg["episodes"]` episodes and write them; returns (out_dir, rows, seconds)."""
    size, method = int(cfg["resolution"]), cfg["resize_method"]
    attrs = {"env_id": "ALE/SpaceInvaders-v5", "frameskip": 4, "maxpool": True,
             "resolution": size, "resize_method": method, "collector_config": cfg}
    policy = EpsilonMixPolicy(ZooAgentPolicy(), float(cfg["epsilon"]))
    out_dir = Path(cfg["out_dir"])
    rows = 0
    start = time.perf_counter()
    with ShardWriter(out_dir, 0, (size, size, 3), attrs) as writer:
        for i in range(int(cfg["episodes"])):
            seed = int(cfg["seed"]) + i
            env = make_env(seed, max_episode_steps=int(cfg["max_steps"]))
            obs, info = env.reset()
            policy.reset(seed, obs)
            writer.begin_episode(to_model_frame(obs, size, method), env.unwrapped.ale.getRAM(),
                                 info["lives"], seed, info["noops"], SOURCE_AGENT,
                                 float(cfg["epsilon"]))
            rows += 1
            terminated = truncated = False
            while not (terminated or truncated):
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                writer.add_step(action, float(reward), to_model_frame(obs, size, method),
                                env.unwrapped.ale.getRAM(), info["lives"])
                rows += 1
            writer.end_episode(END_TERMINATED if terminated else END_TRUNCATED, terminated)
            env.close()
    return out_dir, rows, time.perf_counter() - start


def save_grid(out_dir: Path, n: int) -> Path:
    """Save `n` evenly spaced frames from the shard as a 4x nearest-upscaled grid."""
    from wm.data.reader import ShardReader, list_shards

    reader = ShardReader(list_shards(out_dir)[0])
    idx = np.linspace(0, reader.num_rows - 1, n).astype(np.int64)
    frames = reader.read("frames", idx).repeat(4, axis=1).repeat(4, axis=2)
    cols = 5
    rows = [np.concatenate(frames[r:r + cols], axis=1) for r in range(0, n, cols)]
    path = out_dir / "frames.png"
    iio.imwrite(path, np.concatenate(rows, axis=0))
    return path


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


def main(argv: list[str] | None = None) -> None:
    cfg = resolve_config(argv)
    out_dir, rows, seconds = collect(cfg)
    grid = save_grid(out_dir, int(cfg["grid_frames"]))
    print(f"wrote {rows} rows to {out_dir} in {seconds:.1f} s ({rows / seconds:.0f} rows/s); grid: {grid}")


if __name__ == "__main__":
    main()
