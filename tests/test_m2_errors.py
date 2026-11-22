"""Milestone 2 E9: event detectors, event matching, the nearest-training-frame search, and both script modes."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import make_fake_shard
from wm.data.dataset import is_val_episode
from wm.data.reader import ShardReader, list_shards

REPO = Path(__file__).resolve().parents[1]
H = W = 84
ALIEN, SCORE, EXPLOSION = (134, 134, 29), (50, 132, 50), (162, 134, 56)


def _m2e():
    spec = importlib.util.spec_from_file_location("m2_errors", REPO / "scripts/m2_errors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(n: int) -> np.ndarray:
    return np.zeros((n, H, W, 3), np.uint8)


def test_score_changes_only_in_the_player_one_score_region():
    m = _m2e()
    f = _frames(4)
    f[2:, 3, 10] = SCORE  # player-1 score digit appears at t=2 and stays
    f[3, 3, 70] = SCORE   # right half (player-2 score): ignored
    f[1, 40, 10] = SCORE  # below the score rows: ignored
    assert m.score_changes(f).tolist() == [False, False, True, False]


def test_score_library_snapping_ignores_flicker_but_keeps_real_changes():
    m = _m2e()
    lib_frames = _frames(2)
    lib_frames[0, 3:6, 10:13] = SCORE  # pattern A: one 3x3 digit
    lib_frames[1, 3:6, 20:23] = SCORE  # pattern B: the digit moved
    library = np.unique(m.score_masks(lib_frames), axis=0)
    gen = lib_frames[[0, 0, 0, 1]].copy()
    gen[1, 3, 10] = 0       # flicker: one pixel of A missing
    gen[2, 4, 14] = SCORE   # flicker: one stray pixel next to A
    assert m.score_changes(gen).tolist() == [False, True, True, True]           # exact comparison sees the flicker
    assert m.score_changes(gen, library).tolist() == [False, False, False, True]  # snapped: only the real change


def test_death_onsets_at_the_first_explosion_frame_in_the_player_band():
    m = _m2e()
    f = _frames(6)
    f[2:4, 75, 30] = EXPLOSION  # explosion on frames 2-3
    f[5, 2, 70] = EXPLOSION     # same colour in the player-2 score: ignored
    assert m.death_onsets(f).tolist() == [False, False, True, False, False, False]


def test_alien_count_in_the_alien_band():
    m = _m2e()
    f = _frames(2)
    f[1, 20:22, 10:31] = ALIEN  # 42 pixels = 2 aliens
    f[1, 2, 5] = ALIEN          # score rows: ignored
    assert m.alien_count(f).tolist() == pytest.approx([0.0, 42 / m.PIXELS_PER_ALIEN])


def test_match_events_counts_matched_real_and_unmatched_generated():
    m = _m2e()
    real, gen = np.zeros(30, bool), np.zeros(30, bool)
    real[[3, 10]] = True
    gen[[4, 20]] = True
    assert m.match_events(real, gen, window=2) == (1, 1)
    assert m.match_events(real, gen, window=0) == (0, 2)


def test_nearest_train_frames_matches_brute_force_on_the_train_split(tmp_path):
    m = _m2e()
    make_fake_shard(tmp_path / "train", 6, [9, 7, 12, 5, 10, 8], H, W)
    path = list_shards(tmp_path / "train")[0]
    reader = ShardReader(path)
    frames, eps = reader.read("frames", slice(None)), reader.episodes
    is_train = np.zeros(len(frames), bool)
    for s, n, uid in zip(eps["start"], eps["length"], eps["episode_uid"]):
        is_train[s:s + n] = not is_val_episode(path.name, int(uid), 0.3)
    assert is_train.any() and not is_train.all()
    train_rows, val_rows = np.flatnonzero(is_train), np.flatnonzero(~is_train)
    queries = frames[[train_rows[3], val_rows[0], val_rows[-1]]]

    rmse, where = m.nearest_train_frames(queries, [tmp_path / "train"], 0.3, chunk_rows=7, num_workers=0)
    assert rmse[0] == 0 and where[0] == (str(path), int(train_rows[3]))
    for q in (1, 2):
        d = np.sqrt(((frames[train_rows].astype(np.float64) - queries[q].astype(np.float64)) ** 2).mean(axis=(1, 2, 3)))
        assert rmse[q] == pytest.approx(d.min(), rel=1e-4)
        assert is_train[where[q][1]]  # never a val frame, not even the query itself


def _config(tmp_path: Path, data: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"models: [copy_last]\ndataset_dirs: [{data}]\nn: 2\nstart: 0\nT: 6\nK: 4\nnum_steps: null\ncfg_scale: null\n"
        f"seed: 0\nbatch_size: 2\nmatch_window: 2\nreport_horizons: [1, 6]\nn_examples: 2\n"
        f"nn:\n  train_dirs: [{data}]\n  val_fraction: 0.5\n  horizons: [1, 6]\n  chunk_rows: 5\n  num_workers: 0\n"
        f"out_dir: {tmp_path / 'out'}\n")
    return cfg


def test_events_mode_writes_detector_check_and_per_model_results(tmp_path):
    m = _m2e()
    data = tmp_path / "data"
    make_fake_shard(data, 3, [12, 14, 11], H, W)
    out = m.main(["events", "--config", str(_config(tmp_path, data))])
    result = json.loads((out / "events.json").read_text())
    assert set(result["detectors"]) == {"hits", "deaths", "alien_count_vs_ram_corr"}
    r = result["models"]["copy_last"]
    assert set(r) >= {"hits", "deaths", "alien_abs_error", "player_x_abs_error"}
    assert set(r["alien_abs_error"]) == {"1", "6"}
    assert r["hits"]["real"] >= 0 and r["deaths"]["real"] >= 0


def test_nn_mode_reports_generated_and_real_distances_per_horizon(tmp_path):
    m = _m2e()
    data = tmp_path / "data"
    make_fake_shard(data, 4, [12, 14, 11, 13], H, W)
    out = m.main(["nn", "--config", str(_config(tmp_path, data)), "--models", "copy_last"])
    result = json.loads((out / "nn.json").read_text())
    r = result["models"]["copy_last"]
    assert set(r) == {"1", "6"}
    for h in ("1", "6"):
        for src in ("generated", "real"):
            assert len(r[h][src]["nn_rmse"]) == 2 and min(r[h][src]["nn_rmse"]) >= 0
    assert (out / "nn_examples.png").exists()
