"""Mean and standard deviation of episode score for the zoo agent, epsilon mixes, and random.

The agent row is the zoo policy. epsilon=0.2 and epsilon=0.3 mix random holds into it.
Random is RandomPolicy. Every condition plays the same episode seeds.

    python scripts/eval_agent.py --episodes 5
    python scripts/eval_agent.py --config path.yaml --set episodes=3

`--config` is optional. With no file, `--set key=value` overrides episodes, seed,
repo_id, and epsilons. A config file's keys must already exist in that file.
`--episodes` and `--seed` override the resolved values.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from wm.agents.policies import EpsilonMixPolicy, RandomPolicy
from wm.agents.pretrained import ZooAgentPolicy
from wm.config import load_config
from wm.env import make_env

DEFAULTS = {
    "episodes": 5,
    "seed": 0,
    "repo_id": None,
    "epsilons": [0.2, 0.3],
}


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply flat key=value overrides, parsing values as YAML."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key=value")
        if key not in cfg:
            raise KeyError(f"override key {key!r} not found")
        cfg[key] = yaml.safe_load(raw)


def run_episode(policy, seed: int) -> float:
    """Play one episode from `seed` and return the summed reward."""
    env = make_env(seed)
    try:
        obs, _info = env.reset()
        policy.reset(seed, obs)
        total = 0.0
        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            done = terminated or truncated
        return total
    finally:
        env.close()


def evaluate(episodes: int, seed: int, repo_id: str | None, epsilons: list[float]) -> tuple[str, list[tuple[str, np.ndarray]]]:
    """Score the agent, each epsilon mix, and a random policy on seeds `seed .. seed+episodes-1`."""
    agent = ZooAgentPolicy(repo_id=repo_id)
    conditions: list[tuple[str, object]] = [("agent", agent)]
    for epsilon in epsilons:
        conditions.append((f"epsilon={float(epsilon)}", EpsilonMixPolicy(agent, float(epsilon))))
    conditions.append(("random", RandomPolicy()))
    rows = []
    for name, policy in conditions:
        scores = np.asarray([run_episode(policy, seed + i) for i in range(episodes)], dtype=np.float64)
        rows.append((name, scores))
    return agent.repo_id, rows


def format_table(repo_id: str, episodes: int, seed: int, rows: list[tuple[str, np.ndarray]]) -> str:
    """Render mean and sample standard deviation (ddof=1) for each policy."""
    lines = [f"repo_id={repo_id}  episodes={episodes}  seed={seed}"]
    lines.append(f"{'policy':<14} {'mean':>10} {'std':>10}  scores")
    for name, scores in rows:
        std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        listed = " ".join(f"{score:.1f}" for score in scores)
        lines.append(f"{name:<14} {float(scores.mean()):10.1f} {std:10.1f}  {listed}")
    return "\n".join(lines)


def resolve_config(argv: list[str] | None = None) -> dict:
    """Merge optional YAML, `--set` overrides, and `--episodes` / `--seed`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    if args.config is not None:
        cfg = {**DEFAULTS, **load_config(args.config, args.overrides)}
    else:
        cfg = dict(DEFAULTS)
        apply_overrides(cfg, args.overrides)
    if args.episodes is not None:
        cfg["episodes"] = args.episodes
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


def main(argv: list[str] | None = None) -> None:
    cfg = resolve_config(argv)
    repo_id, rows = evaluate(int(cfg["episodes"]), int(cfg["seed"]), cfg["repo_id"], list(cfg["epsilons"]))
    print(format_table(repo_id, int(cfg["episodes"]), int(cfg["seed"]), rows))


if __name__ == "__main__":
    main()
