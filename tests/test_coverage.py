import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import make_fake_shard
from wm.data import coverage, ram
from wm.env import make_env

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "scripts" / "coverage_report.py"


def _blank(size: int = 84) -> np.ndarray:
    return np.zeros((1, size, size, 3), np.uint8)


# --- Event counting ------------------------------------------------------------------------


def test_wave_transitions_count_jumps_back_up():
    invaders = np.array([36, 35, 2, 1, 0, 0, 36, 36, 35, 0, 36])
    assert coverage.count_increases(invaders) == 2


def test_life_losses_count_decreases():
    lives = np.array([3, 3, 2, 2, 2, 1, 0])
    assert coverage.count_decreases(lives) == 3


# --- Pixel heuristics on 84x84 model frames ------------------------------------------------


@pytest.mark.parametrize("size", [84, 96])
def test_ufo_visible_only_for_ufo_colour_in_top_band(size):
    frames = np.repeat(_blank(size), 4, axis=0)
    frames[0, 2, 40] = coverage.UFO_RGB
    frames[1, 2, 40] = (142, 142, 142)  # laser grey in the top band is not a UFO
    frames[2, size // 2, 40] = coverage.UFO_RGB  # UFO colour below the band
    frames[3, 2, 40] = (162, 134, 56)  # score digits
    assert coverage.ufo_visible(frames).tolist() == [True, False, False, False]


def test_shield_pixels_counted_in_shield_band_only():
    frames = np.repeat(_blank(), 2, axis=0)
    top, bottom = coverage.shield_band(84)
    frames[:, top:bottom, 10:20] = coverage.SHIELD_RGB
    frames[1, top:bottom, 15:20] = 0  # half eroded
    frames[:, 0:5, 0:5] = coverage.SHIELD_RGB  # outside the band
    counts = coverage.shield_pixels(frames)
    assert counts[0] == (bottom - top) * 10
    assert counts[1] == counts[0] // 2


# --- Report on a fake shard ----------------------------------------------------------------


def test_coverage_report_runs_on_fake_shard(tmp_path):
    make_fake_shard(tmp_path / "ds", 3, [5, 1, 40], 84, 84, encode_rows=False)
    out = tmp_path / "figs"
    result = subprocess.run([sys.executable, str(REPORT), str(tmp_path / "ds"), "--out-dir", str(out)],
                            capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    report = (out / "cp2_coverage_report.md").read_text()
    assert "| Target |" in report
    assert (out / "cp2_coverage_hist.png").exists()
    assert list(out.glob("cp2_coverage_montage_*.png"))


def test_dataset_coverage_counts_on_fake_shard(tmp_path):
    make_fake_shard(tmp_path / "ds", 3, [5, 1, 40], 84, 84, encode_rows=False)
    cov = coverage.dataset_coverage(tmp_path / "ds")
    assert cov.frames["lives"].shape == (46,)
    assert len(cov.episodes["length"]) == 3
    # Fake lives drop by one every 10 steps of an episode: the 40-row episode loses 3.
    assert cov.events["life_losses"] == 3
    assert cov.events["game_overs"] == 2  # make_fake_shard terminates even-numbered episodes


# --- RAM map against the real game ---------------------------------------------------------


@pytest.mark.env
def test_ram_map_matches_game_during_random_rollout():
    env = make_env(0)
    env.reset()
    rng = np.random.default_rng(0)
    total, kills = 0.0, 0
    prev = env.unwrapped.ale.getRAM().copy()
    assert prev[ram.INVADERS_LEFT] == ram.WAVE_SIZE
    for _ in range(400):
        _, reward, terminated, truncated, info = env.step(int(rng.integers(0, 6)))
        r = env.unwrapped.ale.getRAM().copy()
        total += reward
        if terminated or truncated:
            break
        assert r[ram.LIVES] == info["lives"]
        assert ram.score(r) == total
        if 0 < reward <= 30:  # an alien kill (the UFO is worth more)
            kills += 1
            assert int(r[ram.INVADERS_LEFT]) == int(prev[ram.INVADERS_LEFT]) - 1
        prev = r
    assert kills > 0
    env.close()
