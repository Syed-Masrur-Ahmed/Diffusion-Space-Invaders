import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from wm.data.reader import ShardReader, list_shards
from wm.data.schema import EVAL_SEED_BASE, LAST_ROW_ACTION, SOURCE_AGENT, SOURCE_RANDOM
from wm.env import make_env, to_model_frame

ROOT = Path(__file__).resolve().parents[1]
COLLECT = ROOT / "scripts" / "collect.py"
VALIDATE = ROOT / "scripts" / "validate_data.py"


def _load_collect():
    spec = importlib.util.spec_from_file_location("collect", COLLECT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(out_dir: Path, jobs: list[dict], total_steps: int = 600, **extra) -> dict:
    cfg = {"out_dir": str(out_dir), "resolution": 84, "resize_method": "max", "base_seed": 0,
           "max_episode_steps": 27_000, "total_steps": total_steps, "total_episodes": None,
           "max_rows": 100_000, "jobs": jobs}
    cfg.update(extra)
    return cfg


RANDOM_JOBS = [{"name": "random_a", "policy": "random", "epsilon": None, "fraction": 0.5},
               {"name": "random_b", "policy": "random", "epsilon": None, "fraction": 0.5}]


def _run_collect(cfg: dict, workers: int, tmp: Path) -> subprocess.CompletedProcess:
    path = tmp / "collect.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return subprocess.run([sys.executable, str(COLLECT), "--config", str(path), "--workers", str(workers)],
                          capture_output=True, text=True, cwd=ROOT, timeout=600)


def _episodes(dataset: Path) -> list[tuple[ShardReader, int]]:
    out = []
    for shard in list_shards(dataset):
        reader = ShardReader(shard)
        out += [(reader, e) for e in range(len(reader.episodes["start"]))]
    return out


def _assert_replays(reader: ShardReader, e: int) -> None:
    """make_env(seed) plus the stored actions reproduces every stored row exactly."""
    rows = reader.episode_rows(e)
    frames, actions = reader.read("frames", rows), reader.read("actions", rows)
    rewards, lives, ram = reader.read("rewards", rows), reader.read("lives", rows), reader.read("ram", rows)
    size, method = int(reader.attrs["resolution"]), str(reader.attrs["resize_method"])
    env = make_env(int(reader.episodes["seed"][e]))
    obs, info = env.reset()
    assert info["noops"] == reader.episodes["noops"][e]
    np.testing.assert_array_equal(to_model_frame(obs, size, method), frames[0])
    np.testing.assert_array_equal(env.unwrapped.ale.getRAM(), ram[0])
    for i, action in enumerate(actions[:-1]):
        obs, reward, _, _, info = env.step(int(action))
        np.testing.assert_array_equal(to_model_frame(obs, size, method), frames[i + 1], err_msg=f"row {i + 1}")
        np.testing.assert_array_equal(env.unwrapped.ale.getRAM(), ram[i + 1])
        assert reward == rewards[i] and info["lives"] == lives[i + 1]
    assert actions[-1] == LAST_ROW_ACTION
    env.close()


# --- Planning (no env) ---------------------------------------------------------------------


def test_plan_splits_step_budget_exactly_across_jobs_and_workers():
    collect = _load_collect()
    jobs = [{"name": "a", "policy": "agent", "epsilon": [0.2, 0.2], "fraction": 0.5},
            {"name": "b", "policy": "agent", "epsilon": [0.3, 0.3], "fraction": 0.25},
            {"name": "c", "policy": "random", "epsilon": None, "fraction": 0.25}]
    parts = collect.plan_parts(_config("x", jobs, total_steps=1003), workers=3)
    assert len(parts) == 9
    assert sorted(p["writer"] for p in parts) == list(range(9))
    assert sum(p["budget"] for p in parts) == 1003
    for j, share in enumerate([0.5, 0.25, 0.25]):
        budgets = [p["budget"] for p in parts if p["job"] == j]
        assert abs(sum(budgets) - share * 1003) <= 1
        assert max(budgets) - min(budgets) <= 1


def test_plan_splits_episode_budget():
    collect = _load_collect()
    jobs = [{"name": "a", "policy": "agent", "epsilon": [0.2, 0.2], "fraction": 0.7},
            {"name": "r", "policy": "random", "epsilon": None, "fraction": 0.3}]
    cfg = _config("x", jobs, total_steps=None, total_episodes=100, base_seed=EVAL_SEED_BASE)
    parts = collect.plan_parts(cfg, workers=2)
    assert sum(p["budget"] for p in parts if p["job"] == 0) == 70
    assert sum(p["budget"] for p in parts if p["job"] == 1) == 30


@pytest.mark.parametrize("change", [
    {"total_steps": None},                                    # neither budget set
    {"total_episodes": 5},                                    # both budgets set
    {"jobs": [{**RANDOM_JOBS[0], "fraction": 0.4}]},          # fractions don't sum to 1
    {"jobs": [{**RANDOM_JOBS[0], "fraction": 1.0, "policy": "human"}]},
    {"base_seed": EVAL_SEED_BASE - 10_000_000},              # seeds would reach the eval range
])
def test_plan_rejects_bad_configs(change):
    collect = _load_collect()
    with pytest.raises(ValueError):
        collect.plan_parts({**_config("x", RANDOM_JOBS), **change}, workers=2)


# --- Tiny collect (random policy, 2 workers x 300 steps) ----------------------------------


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> tuple[Path, Path, dict]:
    tmp = tmp_path_factory.mktemp("collect")
    cfg = _config(tmp / "ds", RANDOM_JOBS)
    result = _run_collect(cfg, 2, tmp)
    assert result.returncode == 0, result.stderr
    return tmp, tmp / "ds", cfg


@pytest.mark.env
def test_tiny_collect_validates(tiny):
    _, dataset, _ = tiny
    result = subprocess.run([sys.executable, str(VALIDATE), str(dataset)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.env
def test_tiny_collect_summary_seeds_and_sources(tiny):
    _, dataset, _ = tiny
    summary = json.loads((dataset / "summary.json").read_text())
    assert summary["steps"] == 600
    episodes = _episodes(dataset)
    assert summary["episodes"] == len(episodes)
    assert summary["rows"] == summary["steps"] + summary["episodes"]
    seeds = [int(r.episodes["seed"][e]) for r, e in episodes]
    uids = [int(r.episodes["episode_uid"][e]) for r, e in episodes]
    assert len(set(seeds)) == len(seeds)
    assert seeds == uids  # base_seed 0: seed = base_seed + episode_uid
    assert all(r.episodes["source"][e] == SOURCE_RANDOM and r.episodes["epsilon"][e] == -1
               for r, e in episodes)
    assert len(list_shards(dataset)) == 4  # 2 jobs x 2 workers, one writer each


@pytest.mark.env
def test_replay_reproduces_collected_episodes(tiny):
    _, dataset, _ = tiny
    for reader, e in _episodes(dataset):
        _assert_replays(reader, e)


@pytest.mark.env
def test_resume_skips_done_parts_and_redoes_unfinished_identically(tiny):
    tmp, dataset, cfg = tiny
    shards = list_shards(dataset)
    mtimes = {p: p.stat().st_mtime_ns for p in shards}
    result = _run_collect(cfg, 2, tmp)
    assert result.returncode == 0, result.stderr
    assert {p: p.stat().st_mtime_ns for p in list_shards(dataset)} == mtimes  # nothing rewritten

    # Simulate a crash in writer 1: its marker is missing and its shard is partial.
    victim = dataset / "shard_w001_00000.h5"
    before = ShardReader(victim).read("frames", slice(None))
    (dataset / "_DONE_w001").unlink()
    victim.write_bytes(b"partial")
    result = _run_collect(cfg, 2, tmp)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(ShardReader(victim).read("frames", slice(None)), before)
    assert (dataset / "_DONE_w001").exists()


@pytest.mark.env
def test_rerun_with_different_settings_is_refused(tiny):
    tmp, _, cfg = tiny
    result = _run_collect(cfg, 3, tmp)
    assert result.returncode != 0
    assert "differs" in result.stderr


# --- Agent collect (downloads the zoo agent) -----------------------------------------------


@pytest.mark.env
@pytest.mark.network
def test_tiny_agent_collect_records_sources_and_epsilon(tmp_path):
    jobs = [{"name": "agent", "policy": "agent", "epsilon": [0.1, 0.3], "fraction": 0.5},
            {"name": "random", "policy": "random", "epsilon": None, "fraction": 0.5}]
    cfg = _config(tmp_path / "ds", jobs)
    result = _run_collect(cfg, 2, tmp_path)
    assert result.returncode == 0, result.stderr
    episodes = _episodes(tmp_path / "ds")
    sources = {int(r.episodes["source"][e]) for r, e in episodes}
    assert sources == {SOURCE_AGENT, SOURCE_RANDOM}
    for r, e in episodes:
        eps = float(r.episodes["epsilon"][e])
        assert (0.1 <= eps <= 0.3) if r.episodes["source"][e] == SOURCE_AGENT else eps == -1
    seeds = [int(r.episodes["seed"][e]) for r, e in episodes]
    assert len(set(seeds)) == len(seeds)
    for reader, e in episodes:
        _assert_replays(reader, e)
