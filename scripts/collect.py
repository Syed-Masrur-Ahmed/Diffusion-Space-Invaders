"""Collect Space Invaders episodes into HDF5 shards (ROADMAP 4.4, CP2-A).

    python scripts/collect.py --config configs/data/debug.yaml --workers 4
    python scripts/collect.py --config configs/data/eval.yaml --workers 2 --set out_dir=data/eval_b

The work is split into parts, one per (job, worker). Each part has its own ShardWriter with
writer id `w = job_index * workers + worker`, so shard names and episode uids never clash,
and episode seeds are `base_seed + w * 10_000_000 + local_counter` (= base_seed + episode_uid).
A finished part leaves a `_DONE_w<w>` marker. A rerun skips finished parts and redoes the
others from scratch (deleting their partial shards); seeding makes the redone data identical.
"""

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

from wm.config import load_config
from wm.data.reader import SHARD_GLOB
from wm.data.schema import (END_TERMINATED, END_TRUNCATED, EPISODE_UID_STRIDE, EVAL_SEED_BASE,
                            HUMAN_SEED_BASE, SOURCE_AGENT, SOURCE_RANDOM)

POLICY_SOURCES = {"agent": SOURCE_AGENT, "random": SOURCE_RANDOM}
SEED_RANGES = [(0, EVAL_SEED_BASE), (EVAL_SEED_BASE, HUMAN_SEED_BASE)]  # training, eval
RUN_FILE = "collect_config.json"


def _split(total: int, weights: list[float]) -> list[int]:
    """Integers proportional to `weights` that sum exactly to `total` (largest remainders)."""
    raw = [total * w / sum(weights) for w in weights]
    out = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: out[i] - raw[i])[: total - sum(out)]:
        out[i] += 1
    return out


def plan_parts(cfg: dict, workers: int) -> list[dict]:
    """Split the step or episode budget into one part per (job, worker); validates the config."""
    if (cfg["total_steps"] is None) == (cfg["total_episodes"] is None):
        raise ValueError("set exactly one of total_steps and total_episodes")
    jobs = cfg["jobs"]
    fractions = [float(j["fraction"]) for j in jobs]
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError(f"job fractions sum to {sum(fractions)}, not 1")
    for job in jobs:
        if job["policy"] not in POLICY_SOURCES:
            raise ValueError(f"unknown policy {job['policy']!r} in job {job['name']!r}")

    n_parts = len(jobs) * workers
    base = int(cfg["base_seed"])
    if not any(lo <= base and base + n_parts * EPISODE_UID_STRIDE <= hi for lo, hi in SEED_RANGES):
        raise ValueError(f"seeds {base}..{base + n_parts * EPISODE_UID_STRIDE} cross a seed range "
                         f"boundary {SEED_RANGES}")

    total = int(cfg["total_steps"] if cfg["total_steps"] is not None else cfg["total_episodes"])
    parts = []
    for j, job_total in enumerate(_split(total, fractions)):
        for worker, budget in enumerate(_split(job_total, [1.0] * workers)):
            parts.append({"job": j, "worker": worker, "writer": j * workers + worker, "budget": budget})
    return parts


def _make_policy(job: dict):
    from wm.agents.policies import EpsilonMixPolicy, RandomPolicy

    if job["policy"] == "agent":
        from wm.agents.pretrained import ZooAgentPolicy

        return EpsilonMixPolicy(ZooAgentPolicy(), 0.0)
    return RandomPolicy()


def run_part(cfg: dict, part: dict) -> dict:
    """Collect one part into its own shards, write its _DONE marker, and return its stats."""
    import torch

    from wm.data.writer import ShardWriter
    from wm.env import make_env, to_model_frame

    torch.set_num_threads(1)  # one process per core; torch's own threads would oversubscribe the CPU
    start = time.perf_counter()
    job = cfg["jobs"][part["job"]]
    size, method = int(cfg["resolution"]), cfg["resize_method"]
    by_steps = cfg["total_steps"] is not None
    attrs = {"env_id": "ALE/SpaceInvaders-v5", "frameskip": 4, "maxpool": True, "resolution": size,
             "resize_method": method, "collector_config": {**cfg, "part": part}}
    policy = _make_policy(job)
    w = part["writer"]

    steps = episodes = 0
    returns, end_reasons = [], {"terminated": 0, "truncated": 0}
    with ShardWriter(cfg["out_dir"], w, (size, size, 3), attrs, int(cfg["max_rows"])) as writer:
        while (steps if by_steps else episodes) < part["budget"]:
            seed = int(cfg["base_seed"]) + w * EPISODE_UID_STRIDE + episodes
            # The last episode of a step budget is truncated so the part hits its budget exactly.
            cap = min(int(cfg["max_episode_steps"]), part["budget"] - steps) if by_steps \
                else int(cfg["max_episode_steps"])
            env = make_env(seed, max_episode_steps=cap)
            obs, info = env.reset()
            epsilon = -1.0
            if job["policy"] == "agent":
                lo, hi = job["epsilon"]
                # [seed, 1] keeps this stream apart from the policy's own default_rng(seed).
                epsilon = float(np.random.default_rng([seed, 1]).uniform(lo, hi))
                policy.epsilon = epsilon
            policy.reset(seed, obs)
            writer.begin_episode(to_model_frame(obs, size, method), env.unwrapped.ale.getRAM(),
                                 info["lives"], seed, info["noops"], POLICY_SOURCES[job["policy"]], epsilon)
            terminated = truncated = False
            ret = 0.0
            while not (terminated or truncated):
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                writer.add_step(action, float(reward), to_model_frame(obs, size, method),
                                env.unwrapped.ale.getRAM(), info["lives"])
                steps += 1
                ret += float(reward)
            writer.end_episode(END_TERMINATED if terminated else END_TRUNCATED, terminated)
            env.close()
            episodes += 1
            returns.append(ret)
            end_reasons["terminated" if terminated else "truncated"] += 1

    stats = {**part, "job_name": job["name"], "steps": steps, "episodes": episodes,
             "rows": steps + episodes, "mean_return": float(np.mean(returns)) if returns else 0.0,
             "end_reasons": end_reasons, "seconds": time.perf_counter() - start}
    (Path(cfg["out_dir"]) / f"_DONE_w{w:03d}").write_text(json.dumps(stats))
    return stats


def _check_run_file(out_dir: Path, run: dict) -> None:
    """Record the config on the first run; refuse to resume with different settings."""
    path = out_dir / RUN_FILE
    if path.exists():
        if json.loads(path.read_text()) != run:
            raise SystemExit(f"{out_dir}: config or --workers differs from the run recorded in {path}; "
                             f"use a new out_dir or delete the dataset")
    else:
        path.write_text(json.dumps(run, indent=2))


def summarize(out_dir: Path, parts: list[dict], cfg: dict) -> dict:
    """Combine the parts' _DONE stats into summary.json."""
    stats = [json.loads((out_dir / f"_DONE_w{p['writer']:03d}").read_text()) for p in parts]
    jobs = {}
    for job in cfg["jobs"]:
        mine = [s for s in stats if s["job_name"] == job["name"]]
        episodes = sum(s["episodes"] for s in mine)
        jobs[job["name"]] = {
            "steps": sum(s["steps"] for s in mine), "episodes": episodes,
            "mean_return": sum(s["mean_return"] * s["episodes"] for s in mine) / max(episodes, 1),
            "terminated": sum(s["end_reasons"]["terminated"] for s in mine),
            "truncated": sum(s["end_reasons"]["truncated"] for s in mine)}
    summary = {"steps": sum(s["steps"] for s in stats), "episodes": sum(s["episodes"] for s in stats),
               "rows": sum(s["rows"] for s in stats), "jobs": jobs,
               "part_seconds": sum(s["seconds"] for s in stats), "parts": stats}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    parts = plan_parts(cfg, args.workers)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_run_file(out_dir, {"config": cfg, "workers": args.workers})

    todo = [p for p in parts if not (out_dir / f"_DONE_w{p['writer']:03d}").exists()]
    for p in todo:  # partial shards from an interrupted run; the part is redone from scratch
        for shard in out_dir.glob(SHARD_GLOB.replace("w*", f"w{p['writer']:03d}")):
            shard.unlink()
    print(f"{out_dir}: {len(parts) - len(todo)} of {len(parts)} parts already done; collecting {len(todo)}")

    start = time.perf_counter()
    if todo:
        # Agent parts are the slowest, so start them first to balance the workers.
        todo.sort(key=lambda p: cfg["jobs"][p["job"]]["policy"] != "agent")
        with multiprocessing.get_context("spawn").Pool(min(args.workers, len(todo))) as pool:
            for s in pool.imap_unordered(_run_part_star, [(cfg, p) for p in todo]):
                print(f"  writer {s['writer']:3d} {s['job_name']}: {s['steps']} steps, {s['episodes']} episodes, "
                      f"{s['seconds']:.0f} s ({time.perf_counter() - start:.0f} s elapsed)", flush=True)

    summary = summarize(out_dir, parts, cfg)
    print(f"{out_dir}: {summary['steps']} steps, {summary['episodes']} episodes, {summary['rows']} rows. "
          f"Check with: python scripts/validate_data.py {out_dir}")


def _run_part_star(args: tuple[dict, dict]) -> dict:
    return run_part(*args)


if __name__ == "__main__":
    sys.exit(main())
