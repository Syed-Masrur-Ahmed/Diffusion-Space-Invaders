"""CP6-B playable demo: scenario selection and loading, the play session, and the headless play loop."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.fixtures import decode_frame, encode_frame
from wm.data import ram as ram_map
from wm.data.reader import list_shards
from wm.data.schema import END_TERMINATED, END_TRUNCATED, RAM_SIZE, SOURCE_AGENT
from wm.data.writer import ShardWriter
from wm.eval.baselines import CopyLastFrame

REPO = Path(__file__).resolve().parents[1]
H = W = 8
K = 4


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_shard(path: Path, episodes: list[dict]) -> Path:
    """Shard whose frames encode their row; each episode gives per-row `lives`, `invaders`, `ufo` lists."""
    attrs = {"env_id": "fake", "frameskip": 4, "maxpool": True, "resolution": H, "resize_method": "max",
             "collector_config": {"fixture": "test_demo"}}

    def row(r: int, e: int, ep: dict, i: int):
        ram = np.zeros(RAM_SIZE, np.uint8)
        ram[ram_map.INVADERS_LEFT] = ep["invaders"][i]
        ram[ram_map.UFO_STATE] = ep["ufo"][i]
        return encode_frame(r, e, H, W), ram, ep["lives"][i]

    with ShardWriter(path, worker=0, frame_shape=(H, W, 3), attrs=attrs) as w:
        r = 0
        for e, ep in enumerate(episodes):
            w.begin_episode(*row(r, e, ep, 0), seed=100 + e, noops=e, source=SOURCE_AGENT, epsilon=0.2)
            for i in range(1, len(ep["lives"])):
                w.add_step(r % 6, 0.0, *row(r + 1, e, ep, i))
                r += 1
            r += 1
            w.end_episode(END_TERMINATED, True)
    return list_shards(path)[0]


def _episode(n: int, **changes) -> dict:
    """n rows of a full wave with 3 lives and no UFO; `changes` maps a field to {row: value}."""
    ep = {"lives": [3] * n, "invaders": [36] * n, "ufo": [0] * n}
    for field, rows in changes.items():
        for i, v in rows.items():
            ep[field][i] = v
    return ep


def _cfg(data: Path, **over) -> dict:
    cfg = {"dataset_dirs": [str(data)], "K": K, "per_type": 1, "min_future": 5, "fresh_step": 10,
           "max_invaders": 6, "ufo_lead": 3, "one_life_settle": 4}
    return {**cfg, **over}


def _scenario_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    _write_shard(data, [
        _episode(60, invaders={i: 5 for i in range(30, 60)}),                     # few invaders from row 30
        _episode(60, ufo={i: 6 for i in range(40, 50)}),                          # UFO appears at row 40
        _episode(60, lives={i: 1 for i in range(20, 60)}),                        # last life from row 20
    ])
    return data


def test_find_scenarios_picks_the_first_matching_row_of_each_type(tmp_path):
    ms = _script("make_scenarios")
    data = _scenario_data(tmp_path)
    entries = ms.find_scenarios(_cfg(data))
    ms.validate_entries(entries)
    by_name = {e["name"]: e for e in entries}
    assert list(by_name) == ["fresh_game", "few_invaders", "ufo", "one_life"]
    assert by_name["fresh_game"]["row"] == 10                # episode 0, step 10
    assert by_name["few_invaders"]["row"] == 30              # episode 0, first row with <= 6 invaders
    assert by_name["ufo"]["row"] == 60 + 40 - 3              # episode 1, 3 steps before the UFO appears
    assert by_name["one_life"]["row"] == 120 + 20 + 4        # episode 2, 4 steps after the last life starts
    assert all(e["dataset"] == str(data) and e["shard"] == list_shards(data)[0].name for e in entries)


def test_find_scenarios_respects_min_future_and_reports_missing_types(tmp_path):
    ms = _script("make_scenarios")
    data = _scenario_data(tmp_path)
    with pytest.raises(ValueError, match="few_invaders"):
        ms.find_scenarios(_cfg(data, min_future=40))  # row 30 + 40 runs past the episode's last row 59


def test_validate_entries_rejects_bad_files():
    ms = _script("make_scenarios")
    good = {"name": "a", "dataset": "data/eval", "shard": "s.h5", "row": 5}
    ms.validate_entries([good])
    for bad in ([{**good, "row": "5"}], [{**good, "extra": 1}], [good, good], [{**good, "name": str(i)} for i in range(10)]):
        with pytest.raises(ValueError):
            ms.validate_entries(bad)


def test_load_scenario_returns_k_frames_and_the_actions_between_them(tmp_path):
    ms = _script("make_scenarios")
    data = _scenario_data(tmp_path)
    for entry in ms.find_scenarios(_cfg(data)):
        s = ms.load_scenario(entry, K)
        r = entry["row"]
        assert s.frames.shape == (K, H, W, 3)
        assert [decode_frame(f)[0] for f in s.frames] == list(range(r - K + 1, r + 1))
        # fixture action at row q is q % 6 (taken at q, leads to q+1); the context needs a_{r-K+1}..a_{r-1}
        assert s.actions.tolist() == [q % 6 for q in range(r - K + 1, r)]
        assert s.prefix_actions[-(K - 1):].tolist() == s.actions.tolist()


def test_load_scenario_refuses_a_window_that_crosses_an_episode_start(tmp_path):
    ms = _script("make_scenarios")
    data = _scenario_data(tmp_path)
    entry = {"name": "x", "dataset": str(data), "shard": list_shards(data)[0].name, "row": 61}
    with pytest.raises(ValueError, match="episode"):
        ms.load_scenario(entry, K)


def test_committed_scenarios_file_is_valid_and_loads_when_data_is_present():
    ms = _script("make_scenarios")
    entries = json.loads((REPO / "demo/scenarios.json").read_text())
    ms.validate_entries(entries)
    for entry in entries:
        if not (REPO / entry["dataset"] / entry["shard"]).exists():
            pytest.skip(f"{entry['dataset']} not present")
        assert ms.load_scenario(entry, K, root=REPO).frames.shape[0] == K


class _Recorder:
    """Protocol model that records its inputs and returns the last context frame."""

    K, C, H, W = K, 3, H, W

    def __init__(self):
        self.calls = []

    def predict_next(self, context, actions, num_steps=None, cfg_scale=None, ctx_level=0, generator=None):
        self.calls.append((context.clone(), actions.clone(), num_steps))
        return context[:, -1].clone()


def test_session_feeds_the_recorded_actions_then_the_players(tmp_path):
    ms, pm = _script("make_scenarios"), _script("play_model")
    data = _scenario_data(tmp_path)
    scenarios = [ms.load_scenario(e, K) for e in ms.find_scenarios(_cfg(data))]
    model = _Recorder()
    session = pm.DemoSession(model, scenarios, num_steps=2, max_steps=10)
    s = scenarios[0]
    session.reset(0)
    session.step(5)
    session.step(3)
    ctx0, act0, steps0 = model.calls[0]
    assert act0[0].tolist() == s.actions.tolist() + [5]  # actions[:, -1] produces the new frame
    assert [decode_frame(f)[0] for f in pm.to_uint8(ctx0[0])] == [decode_frame(f)[0] for f in s.frames]
    assert steps0 == 2
    _, act1, _ = model.calls[1]
    assert act1[0].tolist() == s.actions.tolist()[1:] + [5, 3]
    session.reset()  # R: back to the same scenario
    assert np.array_equal(session.frame, s.frames[-1])


def test_session_clamps_sampling_steps():
    pm = _script("play_model")
    session = pm.DemoSession(_Recorder(), [], num_steps=2, max_steps=3)
    assert [session.change_steps(d) for d in (-1, -1, 1, 1, 1, 1)] == [1, 1, 2, 3, 3, 3]


def test_frame_conversion_round_trips_uint8():
    pm = _script("play_model")
    frames = np.random.default_rng(0).integers(0, 256, (K, H, W, 3), dtype=np.uint8)
    x = pm.to_model_input(frames, C=3)
    assert x.shape == (K, 3, H, W) and x.min() >= -1 and x.max() <= 1
    assert np.array_equal(pm.to_uint8(x), frames)
    gray = pm.to_uint8(pm.to_model_input(frames, C=1))
    assert gray.shape == (K, H, W, 3) and np.array_equal(gray[..., 0], gray[..., 2])


def test_play_loop_runs_headless_with_copy_last(tmp_path):
    # conftest sets SDL_VIDEODRIVER=dummy.
    ms, pm = _script("make_scenarios"), _script("play_model")
    data = _scenario_data(tmp_path)
    scenarios = [ms.load_scenario(e, K) for e in ms.find_scenarios(_cfg(data))]
    cfg = {"hz": 0, "num_steps": None, "max_steps": 10, "ctx_level": 0, "cfg_scale": None, "seed": 0,
           "scale": 2, "widen": 2, "scenario": 1}
    assert pm.run(cfg, CopyLastFrame(K, 3, H, W), scenarios, max_ticks=20) == 20


@pytest.mark.env
def test_real_replay_reaches_the_scenario_row_and_follows_the_actions(tmp_path):
    from wm.env import make_env, to_model_frame

    ms, pm = _script("make_scenarios"), _script("play_model")
    seed, actions = 7, [0, 1, 2, 3, 4, 5, 2, 2, 3, 3, 1, 0]
    env = make_env(seed)
    obs, info = env.reset()
    frames, rams = [to_model_frame(obs, 84, "max")], [env.unwrapped.ale.getRAM().copy()]
    for a in actions:
        obs, *_ = env.step(a)
        frames.append(to_model_frame(obs, 84, "max"))
        rams.append(env.unwrapped.ale.getRAM().copy())
    env.close()
    attrs = {"env_id": "ALE/SpaceInvaders-v5", "frameskip": 4, "maxpool": True, "resolution": 84,
             "resize_method": "max", "collector_config": {"fixture": "test_demo"}}
    with ShardWriter(tmp_path / "real", worker=0, frame_shape=(84, 84, 3), attrs=attrs) as w:
        w.begin_episode(frames[0], rams[0], 3, seed=seed, noops=info["noops"], source=SOURCE_AGENT, epsilon=0.2)
        for a, f, r in zip(actions, frames[1:], rams[1:]):
            w.add_step(a, 0.0, f, r, 3)
        w.end_episode(END_TRUNCATED, False)
    entry = {"name": "x", "dataset": str(tmp_path / "real"), "shard": list_shards(tmp_path / "real")[0].name, "row": 8}
    scenario = ms.load_scenario(entry, K)
    real = pm.RealReplay(scenario)
    assert np.array_equal(real.frame, frames[8])
    frame, done = real.step(actions[8])
    assert np.array_equal(frame, frames[9]) and not done
    real.close()


def test_model_input_matches_the_dataset_normalization():
    pm = _script("play_model")
    frames = np.full((1, 2, 2, 3), 255, np.uint8)
    assert torch.allclose(pm.to_model_input(frames, C=3), torch.ones(1, 3, 2, 2))
