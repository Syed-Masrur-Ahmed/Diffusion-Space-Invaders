import importlib.util
from pathlib import Path

import numpy as np
import pygame
import pytest

from wm.data.reader import ShardReader, list_shards
from wm.data.schema import (END_STOPPED, EPISODE_UID_STRIDE, HUMAN_SEED_BASE, LAST_ROW_ACTION,
                            SOURCE_HUMAN)
from wm.env import make_env, to_model_frame

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(out_dir: Path, first_worker: int = 0) -> dict:
    return {"out_dir": str(out_dir), "resolution": 84, "resize_method": "max", "max_rows": 100_000,
            "max_episode_steps": 27_000, "first_worker": first_worker}


def _scripted(script: dict[int, list[tuple[int, int]]]):
    """get_events stand-in: on call t it returns the (event type, key) pairs scripted for tick t."""
    calls = iter(range(10**6))

    def get_events():
        return [pygame.event.Event(t, key=k) for t, k in script.get(next(calls), [])]
    return get_events


def test_next_worker(tmp_path):
    rec = _load("play_record")
    assert rec.next_worker(tmp_path, 0) == 0
    assert rec.next_worker(tmp_path, 100) == 100
    for name in ("shard_w000_00000.h5", "shard_w000_00001.h5", "shard_w001_00000.h5",
                 "shard_w003_00000.h5", "shard_w100_00000.h5", "summary.json"):
        (tmp_path / name).touch()
    assert rec.next_worker(tmp_path, 0) == 2  # another person's range (w100) doesn't push this one up
    assert rec.next_worker(tmp_path, 100) == 101
    assert rec.next_worker(tmp_path, 200) == 200


@pytest.mark.parametrize("hz", [0, -1, 15.5, 30])
def test_hz_above_real_speed_or_non_positive_is_rejected(hz):
    rec = _load("play_record")
    with pytest.raises(SystemExit):
        rec.main(["--config", str(ROOT / "configs/data/human.yaml"), "--hz", str(hz)])


@pytest.mark.env
def test_scripted_session_writes_valid_human_shard(tmp_path):
    rec, validate = _load("play_record"), _load("validate_data")
    down, up = pygame.KEYDOWN, pygame.KEYUP
    script = {0: [(down, pygame.K_d)], 3: [(down, pygame.K_SPACE)], 4: [(up, pygame.K_SPACE)],
              6: [(up, pygame.K_d), (down, pygame.K_a)], 10: [(down, pygame.K_ESCAPE)]}
    stats = rec.run(_cfg(tmp_path), hz=0, scale=1, get_events=_scripted(script))
    assert stats == {"worker": 0, "episodes": 1, "steps": 10}

    errors, _ = validate.validate(tmp_path)
    assert errors == []
    (shard,) = list_shards(tmp_path)
    r = ShardReader(shard)
    assert r.attrs["resolution"] == 84 and r.attrs["resize_method"] == "max"
    # FireLatch keeps fire on for 4 ticks after the tap at tick 3 (ticks 3-6).
    expected = [2, 2, 2, 4, 4, 4, 5, 3, 3, 3, LAST_ROW_ACTION]  # RIGHT x3, RIGHTFIRE x3, LEFTFIRE, LEFT x3
    assert r.read("actions", slice(None)).tolist() == expected
    eps = r.episodes
    assert eps["source"].tolist() == [SOURCE_HUMAN] and eps["end_reason"].tolist() == [END_STOPPED]
    assert eps["epsilon"].tolist() == [-1.0] and not r.read("terminal", slice(None)).any()
    seed = int(eps["seed"][0])
    assert seed == HUMAN_SEED_BASE + int(eps["episode_uid"][0])

    # Deterministic replay (D9) reproduces every recorded frame.
    env = make_env(seed)
    obs, info = env.reset()
    assert info["noops"] == eps["noops"][0]
    frames = [to_model_frame(obs, 84, "max")]
    for a in expected[:-1]:
        obs, *_ = env.step(a)
        frames.append(to_model_frame(obs, 84, "max"))
    assert np.array_equal(np.stack(frames), r.read("frames", slice(None)))
    r.close()


@pytest.mark.env
def test_second_session_gets_new_worker_and_seeds(tmp_path):
    rec, validate = _load("play_record"), _load("validate_data")
    rec.run(_cfg(tmp_path), hz=0, scale=1, max_ticks=3, get_events=_scripted({}))
    stats = rec.run(_cfg(tmp_path), hz=0, scale=1, max_ticks=3, get_events=_scripted({}))
    assert stats["worker"] == 1
    errors, _ = validate.validate(tmp_path)
    assert errors == []
    seeds = [int(ShardReader(p).episodes["seed"][0]) for p in list_shards(tmp_path)]
    assert seeds == [HUMAN_SEED_BASE, HUMAN_SEED_BASE + EPISODE_UID_STRIDE]
