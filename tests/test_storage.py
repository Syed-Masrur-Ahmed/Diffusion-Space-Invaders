"""Tests for the HDF5 shard writer, reader, validator, and make_fake_shard (CP1-B, ROADMAP 4.4).

Episode lengths in this file are in rows: an episode with T steps has T + 1 rows.
"""

import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.fixtures import decode_frame, make_fake_shard
from wm.data import writer as writer_mod
from wm.data.reader import ShardReader, list_shards
from wm.data.schema import (END_TERMINATED, END_TRUNCATED, EPISODE_FIELDS, EPISODE_UID_STRIDE,
                            FILE_ATTRS, NUM_ACTIONS, RAM_SIZE, ROW_FIELDS, SCHEMA_VERSION,
                            SOURCE_AGENT, SOURCE_RANDOM)
from wm.data.writer import ShardWriter

REPO = Path(__file__).resolve().parents[1]
VALIDATE = REPO / "scripts" / "validate_data.py"
H, W = 6, 5  # non-square on purpose, to catch H/W swaps
ATTRS = {"env_id": "ALE/SpaceInvaders-v5", "frameskip": 4, "maxpool": True, "resolution": H,
         "resize_method": "area", "collector_config": {"policy": "test", "epsilon": 0.2}}


def _random_episode(rng: np.random.Generator, n_rows: int, terminated: bool,
                    shape: tuple[int, int, int] = (H, W, 3)) -> dict:
    """Random data for one episode of `n_rows` rows, plus the episode-level arguments."""
    return {
        "frames": rng.integers(0, 256, (n_rows, *shape), dtype=np.uint8),
        "ram": rng.integers(0, 256, (n_rows, RAM_SIZE), dtype=np.uint8),
        "lives": rng.integers(0, 4, n_rows).astype(np.int8),
        "actions": rng.integers(0, NUM_ACTIONS, n_rows - 1),
        "rewards": rng.choice([0.0, 5.0, 10.0, 200.0], n_rows - 1).astype(np.float32),
        "seed": int(rng.integers(0, 10**6)), "noops": int(rng.integers(0, 31)),
        "source": SOURCE_AGENT, "epsilon": 0.2, "terminated": terminated,
        "end_reason": END_TERMINATED if terminated else END_TRUNCATED,
    }


def _write_episode(w: ShardWriter, ep: dict) -> None:
    w.begin_episode(ep["frames"][0], ep["ram"][0], int(ep["lives"][0]), ep["seed"],
                    ep["noops"], ep["source"], ep["epsilon"])
    for t in range(len(ep["frames"]) - 1):
        w.add_step(int(ep["actions"][t]), float(ep["rewards"][t]), ep["frames"][t + 1],
                   ep["ram"][t + 1], int(ep["lives"][t + 1]))
    w.end_episode(ep["end_reason"], ep["terminated"])


def _check_episodes_round_trip(r: ShardReader, eps: list[dict], first_uid: int = 0) -> None:
    """Every per-row field and every episodes/ entry of `r` equals what `eps` wrote."""
    lengths = [len(ep["frames"]) for ep in eps]
    assert r.num_rows == sum(lengths)
    e_ = r.episodes
    assert set(e_) == set(EPISODE_FIELDS)
    for name, dtype in EPISODE_FIELDS.items():
        assert e_[name].dtype == dtype and e_[name].shape == (len(eps),), name
    np.testing.assert_array_equal(e_["start"], np.cumsum([0] + lengths[:-1]))
    np.testing.assert_array_equal(e_["length"], lengths)
    np.testing.assert_array_equal(e_["episode_uid"], first_uid + np.arange(len(eps)))
    for e, ep in enumerate(eps):
        rows = r.episode_rows(e)
        n = len(ep["frames"])
        assert rows == slice(int(e_["start"][e]), int(e_["start"][e]) + n)
        assert e_["seed"][e] == ep["seed"] and e_["noops"][e] == ep["noops"]
        assert e_["source"][e] == ep["source"] and e_["end_reason"][e] == ep["end_reason"]
        assert e_["epsilon"][e] == np.float32(ep["epsilon"])
        assert e_["return_"][e] == np.float32(ep["rewards"].sum())
        np.testing.assert_array_equal(r.read("frames", rows), ep["frames"])
        np.testing.assert_array_equal(r.read("ram", rows), ep["ram"])
        np.testing.assert_array_equal(r.read("lives", rows), ep["lives"])
        np.testing.assert_array_equal(r.read("actions", rows), np.append(ep["actions"], -1))
        np.testing.assert_array_equal(r.read("rewards", rows), np.append(ep["rewards"], 0))
        np.testing.assert_array_equal(r.read("episode_idx", rows), np.full(n, e))
        np.testing.assert_array_equal(r.read("step_in_episode", rows), np.arange(n))
        terminal = np.zeros(n, bool)
        terminal[-1] = ep["terminated"]
        np.testing.assert_array_equal(r.read("terminal", rows), terminal)


def _validate(dataset_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(VALIDATE), str(dataset_dir)],
                          capture_output=True, text=True, cwd=REPO)


# --- writer + reader round trip -------------------------------------------------------------

def test_round_trip_every_field(tmp_path):
    rng = np.random.default_rng(0)
    eps = [_random_episode(rng, 5, True), _random_episode(rng, 1, False),
           _random_episode(rng, 40, True)]
    with ShardWriter(tmp_path, worker=0, frame_shape=(H, W, 3), attrs=ATTRS) as w:
        for ep in eps:
            _write_episode(w, ep)
    shards = list_shards(tmp_path)
    assert [p.name for p in shards] == ["shard_w000_00000.h5"]
    r = ShardReader(shards[0])
    _check_episodes_round_trip(r, eps)

    with h5py.File(shards[0], "r") as f:
        for name, (dtype, trailing) in ROW_FIELDS.items():
            ds = f[name]
            assert ds.dtype == dtype, name
            assert ds.maxshape[0] is None, name
            assert ds.shape[1:] == ((H, W, 3) if name == "frames" else trailing), name
        assert f["frames"].chunks == (32, H, W, 3)
        assert f["frames"].compression == "lzf"


def test_attrs(tmp_path):
    with ShardWriter(tmp_path, 0, (H, W, 3), ATTRS) as w:
        _write_episode(w, _random_episode(np.random.default_rng(0), 3, True))
    a = ShardReader(list_shards(tmp_path)[0]).attrs
    assert set(FILE_ATTRS) <= set(a)
    assert a["schema_version"] == SCHEMA_VERSION
    assert a["env_id"] == "ALE/SpaceInvaders-v5" and a["resolution"] == H
    assert a["frameskip"] == 4 and bool(a["maxpool"]) is True
    assert isinstance(a["collector_config"], str) and '"epsilon": 0.2' in a["collector_config"]
    assert len(a["git_commit"]) == 40 or a["git_commit"] == "unknown"
    assert a["created_utc"].endswith("Z")
    assert a["ale_py_version"] not in ("", None) and a["gymnasium_version"] not in ("", None)


def test_caller_versions_are_kept(tmp_path):
    attrs = {**ATTRS, "ale_py_version": "9.9.9", "gymnasium_version": "8.8.8"}
    with ShardWriter(tmp_path, 0, (H, W, 3), attrs) as w:
        _write_episode(w, _random_episode(np.random.default_rng(0), 2, False))
    a = ShardReader(list_shards(tmp_path)[0]).attrs
    assert a["ale_py_version"] == "9.9.9" and a["gymnasium_version"] == "8.8.8"


def test_missing_caller_attrs_raise(tmp_path):
    with pytest.raises(ValueError, match="resolution"):
        ShardWriter(tmp_path, 0, (H, W, 3), {k: v for k, v in ATTRS.items() if k != "resolution"})


def test_git_commit_outside_repo(tmp_path):
    assert writer_mod._git_commit(tmp_path) == "unknown"


def test_writer_does_not_import_env():
    code = "import sys, wm.data.writer, wm.data.reader; assert 'wm.env' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


# --- rotation and buffering -----------------------------------------------------------------

def test_rotation_only_between_episodes(tmp_path):
    rng = np.random.default_rng(1)
    lengths = [12, 3, 15, 4, 4]  # max_rows=10: shards get [12], [3, 15], [4, 4]
    eps = [_random_episode(rng, n, e % 2 == 0) for e, n in enumerate(lengths)]
    with ShardWriter(tmp_path, worker=3, frame_shape=(H, W, 3), attrs=ATTRS, max_rows=10) as w:
        for ep in eps:
            _write_episode(w, ep)
    shards = list_shards(tmp_path)
    assert [p.name for p in shards] == [f"shard_w003_{i:05d}.h5" for i in range(3)]
    groups = [eps[0:1], eps[1:3], eps[3:5]]
    first_uids = [0, 1, 3]
    for path, group, uid0 in zip(shards, groups, first_uids):
        r = ShardReader(path)
        np.testing.assert_array_equal(r.episodes["start"][:1], [0])  # episode_idx restarts
        _check_episodes_round_trip(r, group, first_uid=3 * EPISODE_UID_STRIDE + uid0)


def test_episode_longer_than_flush_buffer(tmp_path):
    rng = np.random.default_rng(2)
    shape = (4, 3, 3)
    eps = [_random_episode(rng, 1024, False, shape), _random_episode(rng, 1025, True, shape),
           _random_episode(rng, 2500, True, shape), _random_episode(rng, 1, True, shape)]
    with ShardWriter(tmp_path, 0, shape, ATTRS) as w:
        for ep in eps:
            _write_episode(w, ep)
    _check_episodes_round_trip(ShardReader(list_shards(tmp_path)[0]), eps)


# --- writer errors and lifecycle ------------------------------------------------------------

def _first_row(rng):
    ep = _random_episode(rng, 2, False)
    return ep, (ep["frames"][0], ep["ram"][0], 3, 0, 0, SOURCE_RANDOM, -1.0)


def test_add_step_without_begin_raises(tmp_path):
    ep, _ = _first_row(np.random.default_rng(0))
    with ShardWriter(tmp_path, 0, (H, W, 3), ATTRS) as w:
        with pytest.raises(RuntimeError, match="begin_episode"):
            w.add_step(0, 0.0, ep["frames"][1], ep["ram"][1], 3)
        with pytest.raises(RuntimeError, match="begin_episode"):
            w.end_episode(END_TERMINATED, True)


def test_begin_twice_raises(tmp_path):
    _, first = _first_row(np.random.default_rng(0))
    with ShardWriter(tmp_path, 0, (H, W, 3), ATTRS) as w:
        w.begin_episode(*first)
        with pytest.raises(RuntimeError, match="already open"):
            w.begin_episode(*first)
        w.end_episode(END_TRUNCATED, False)


def test_bad_inputs_raise(tmp_path):
    ep, first = _first_row(np.random.default_rng(0))
    with ShardWriter(tmp_path, 0, (H, W, 3), ATTRS) as w:
        with pytest.raises(ValueError):
            w.begin_episode(np.zeros((W, H, 3), np.uint8), *first[1:])
        with pytest.raises(ValueError):
            w.begin_episode(ep["frames"][0].astype(np.float32), *first[1:])
        w.begin_episode(*first)
        with pytest.raises(ValueError):
            w.add_step(NUM_ACTIONS, 0.0, ep["frames"][1], ep["ram"][1], 3)
        with pytest.raises(ValueError):
            w.add_step(0, 0.0, ep["frames"][1], ep["ram"][1][:10], 3)
        w.end_episode(END_TRUNCATED, False)


def test_close_with_open_episode_warns_and_discards(tmp_path):
    rng = np.random.default_rng(3)
    done = _random_episode(rng, 4, True)
    partial = _random_episode(rng, 1500, True)  # longer than one flush, so rows hit the disk
    w = ShardWriter(tmp_path, 0, (H, W, 3), ATTRS)
    _write_episode(w, done)
    w.begin_episode(partial["frames"][0], partial["ram"][0], 3, 0, 0, SOURCE_AGENT, 0.2)
    for t in range(1499):
        w.add_step(0, 0.0, partial["frames"][t + 1], partial["ram"][t + 1], 3)
    with pytest.warns(RuntimeWarning, match="discard"):
        w.close()
    _check_episodes_round_trip(ShardReader(list_shards(tmp_path)[0]), [done])
    assert _validate(tmp_path).returncode == 0


def test_close_is_idempotent_and_final(tmp_path):
    _, first = _first_row(np.random.default_rng(0))
    w = ShardWriter(tmp_path, 0, (H, W, 3), ATTRS)
    w.close()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w.close()
    with pytest.raises(RuntimeError, match="closed"):
        w.begin_episode(*first)


def test_never_overwrites_existing_shard(tmp_path):
    make_fake_shard(tmp_path, 1, [3], H, W)
    _, first = _first_row(np.random.default_rng(0))
    with ShardWriter(tmp_path, 0, (H, W, 3), ATTRS) as w:
        with pytest.raises(FileExistsError):
            w.begin_episode(*first)


# --- reader ---------------------------------------------------------------------------------

def test_reader_index_arrays(tmp_path):
    path = make_fake_shard(tmp_path, 2, [5, 7], H, W)
    r = ShardReader(path)
    rows = np.array([0, 3, 3, 11])
    np.testing.assert_array_equal(r.read("frames", rows), r.read("frames", slice(0, 12))[rows])
    np.testing.assert_array_equal(r.read("actions", rows), r.read("actions", slice(None))[rows])
    with pytest.raises(ValueError, match="sorted"):
        r.read("actions", np.array([4, 2]))
    with pytest.raises(KeyError):
        r.read("start", slice(0, 2))


def test_reader_is_read_only(tmp_path):
    r = ShardReader(make_fake_shard(tmp_path, 1, [3], H, W))
    assert r._file.mode == "r"


def test_reader_pickle_before_and_after_access(tmp_path):
    path = make_fake_shard(tmp_path, 3, [4, 1, 9], H, W)
    fresh = ShardReader(path)
    before = pickle.loads(pickle.dumps(fresh))
    used = ShardReader(path)
    expected = used.read("frames", slice(None))
    _ = used.episodes, used.attrs
    after = pickle.loads(pickle.dumps(used))
    for r in (before, after):
        assert r.num_rows == 14
        np.testing.assert_array_equal(r.read("frames", slice(None)), expected)
        np.testing.assert_array_equal(r.episodes["length"], [4, 1, 9])
        assert r.episode_rows(2) == slice(5, 14)


def test_list_shards_sorted(tmp_path):
    for name in ["shard_w001_00000.h5", "shard_w000_00001.h5", "shard_w000_00000.h5", "x.h5"]:
        (tmp_path / name).touch()
    assert [p.name for p in list_shards(tmp_path)] == [
        "shard_w000_00000.h5", "shard_w000_00001.h5", "shard_w001_00000.h5"]


# --- make_fake_shard ------------------------------------------------------------------------

def test_fake_shard_encodes_rows(tmp_path):
    lengths = [3, 1, 300, 20]  # 324 rows, so G (row // 256) is non-zero at the end
    path = make_fake_shard(tmp_path, len(lengths), lengths, H, W, encode_rows=True)
    assert path == tmp_path / "shard_w000_00000.h5"
    r = ShardReader(path)
    frames = r.read("frames", slice(None))
    episode_idx = r.read("episode_idx", slice(None))
    for row in range(r.num_rows):
        assert decode_frame(frames[row]) == (row, episode_idx[row])
    acts = r.read("actions", slice(None))
    last = r.episodes["start"] + r.episodes["length"] - 1
    assert set(np.unique(np.delete(acts, last))) == set(range(NUM_ACTIONS))
    assert len(np.unique(r.read("rewards", slice(None)))) > 1
    assert len(np.unique(r.read("lives", slice(None)))) > 1
    assert len(np.unique(r.read("ram", slice(None))[:, 0])) > 1


def test_fake_shard_is_deterministic(tmp_path):
    a = ShardReader(make_fake_shard(tmp_path / "a", 2, [4, 6], H, W, encode_rows=False))
    b = ShardReader(make_fake_shard(tmp_path / "b", 2, [4, 6], H, W, encode_rows=False))
    for name in ROW_FIELDS:
        np.testing.assert_array_equal(a.read(name, slice(None)), b.read(name, slice(None)))


def test_fake_shard_rejects_bad_lengths(tmp_path):
    with pytest.raises(ValueError):
        make_fake_shard(tmp_path, 2, [3], H, W)
    with pytest.raises(ValueError):
        make_fake_shard(tmp_path, 1, [0], H, W)


# --- validator ------------------------------------------------------------------------------

def test_validator_passes_on_clean_shards(tmp_path):
    make_fake_shard(tmp_path, 3, [5, 1, 40], H, W)
    res = _validate(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "rows: 46" in res.stdout and "episodes: 3" in res.stdout
    assert "compression ratio" in res.stdout


def _corrupt(path: Path, fn) -> None:
    with h5py.File(path, "r+") as f:
        fn(f)


def _set(name, idx, value):
    def fn(f):
        f[name][idx] = value
    return fn


@pytest.mark.parametrize("fn, needle", [
    (_set("actions", 3, -1), "row 3"),                         # -1 before the last row
    (_set("actions", 4, 2), "row 4"),                          # last row of episode 0 not -1
    (_set("actions", 7, 6), "row 7"),                          # out of range
    (_set("terminal", 2, True), "row 2"),                      # terminal on a middle row
    (_set("episodes/start", 1, 6), "gap"),                     # gap between episodes 0 and 1
    (_set("episodes/start", 2, 5), "overlap"),                 # overlap
    (_set("step_in_episode", 10, 0), "row 10"),
    (_set("episode_idx", 10, 0), "row 10"),
    (_set("rewards", 4, 1.0), "row 4"),                        # reward on a last row
])
def test_validator_catches_injected_errors(tmp_path, fn, needle):
    path = make_fake_shard(tmp_path, 3, [5, 1, 40], H, W)
    _corrupt(path, fn)
    res = _validate(tmp_path)
    assert res.returncode != 0
    assert path.name in res.stdout and needle in res.stdout, res.stdout


def test_validator_catches_schema_errors(tmp_path):
    path = make_fake_shard(tmp_path, 2, [3, 4], H, W)

    def fn(f):
        f.attrs["schema_version"] = 2
        del f["lives"]
        f.create_dataset("lives", data=np.zeros(7, np.int16))
    _corrupt(path, fn)
    out = _validate(tmp_path).stdout
    assert "schema_version" in out and "lives" in out


def test_validator_catches_duplicate_uids(tmp_path):
    path = make_fake_shard(tmp_path, 2, [3, 4], H, W)
    (tmp_path / "shard_w001_00000.h5").write_bytes(path.read_bytes())
    res = _validate(tmp_path)
    assert res.returncode != 0 and "episode_uid" in res.stdout


def test_validator_fails_on_empty_dir(tmp_path):
    assert _validate(tmp_path).returncode != 0


# --- speed ----------------------------------------------------------------------------------

@pytest.mark.slow
def test_write_speed_84(tmp_path):
    rng = np.random.default_rng(0)
    # Mostly-black frames with a few bright sprites, like Space Invaders.
    pool = np.zeros((64, 84, 84, 3), np.uint8)
    for f in pool:
        ys, xs = rng.integers(0, 80, 40), rng.integers(0, 80, 40)
        for y, x in zip(ys, xs):
            f[y:y + 3, x:x + 4] = rng.integers(50, 256, 3)
    ram = rng.integers(0, 256, (64, RAM_SIZE), dtype=np.uint8)
    n_eps, steps = 20, 999  # 20,000 rows
    t0 = time.perf_counter()
    with ShardWriter(tmp_path, 0, (84, 84, 3), {**ATTRS, "resolution": 84}) as w:
        for e in range(n_eps):
            w.begin_episode(pool[0], ram[0], 3, e, 0, SOURCE_AGENT, 0.2)
            for t in range(steps):
                w.add_step(t % NUM_ACTIONS, 0.0, pool[(t + 1) % 64], ram[(t + 1) % 64], 3)
            w.end_episode(END_TRUNCATED, False)
    rate = n_eps * (steps + 1) / (time.perf_counter() - t0)
    print(f"\nwrite speed at 84x84: {rate:,.0f} rows/s")
    assert rate >= 2_000
