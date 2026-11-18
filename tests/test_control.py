"""CP5-A: pixel detectors, counterfactual metrics, inverse dynamics model, and the eval scripts."""

import numpy as np
import pytest

from wm.controls import ACTION_TO_MOVE_FIRE
from wm.data.schema import ACTION_MEANINGS, NUM_ACTIONS
from wm.eval import detectors

H = W = 84
GROUND_RGB = (80, 89, 22)


def scene(player: float | None = None, laser_col: int | None = None, laser_rows: tuple[int, int] = (64, 72),
          bullet: tuple[int, int] | None = None, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Synthetic 84x84 frame: ground, a 5-px player centred on column `player`, optional laser / bullet."""
    f = np.zeros((H, W, 3), np.uint8)
    f[78:] = GROUND_RGB
    f[62:70, 10:16] = (181, 83, 40)  # a shield, which must not count as player or laser
    if player is not None:
        x = int(round(player))
        f[74:78, x - 2:x + 3] = detectors.PLAYER_RGB
    if laser_col is not None:
        f[laser_rows[0]:laser_rows[1], laser_col:laser_col + 2] = detectors.LASER_RGB
    if bullet is not None:  # an alien bullet: one grey column, 4 rows
        f[bullet[0]:bullet[0] + 4, bullet[1]] = detectors.LASER_RGB
    if noise:  # generated frames are close to, not exactly, the palette
        f = np.clip(f + np.random.default_rng(seed).normal(0, noise, f.shape), 0, 255).astype(np.uint8)
    return f


# --- Detectors ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [8, 19, 42, 75])
def test_player_x_is_the_sprite_centroid(x):
    np.testing.assert_allclose(detectors.player_x(scene(player=x)[None]), [x])


def test_player_x_tolerates_generated_frame_noise_and_ignores_other_colours():
    frames = np.stack([scene(player=x, laser_col=x, bullet=(70, 60), noise=8.0, seed=x) for x in (12, 30, 55)])
    np.testing.assert_allclose(detectors.player_x(frames), [12, 30, 55], atol=0.5)


def test_player_x_is_nan_without_a_player():
    frames = np.stack([scene(), scene(player=40)])
    frames[0, 40:44, 30:35] = detectors.PLAYER_RGB  # player colour above the bottom band
    x = detectors.player_x(frames)
    assert np.isnan(x[0]) and x[1] == 40


def test_laser_present_only_for_a_laser_above_the_player():
    frames = np.stack([
        scene(player=30, laser_col=30),                       # just fired
        scene(player=30, laser_col=30, noise=8.0),            # same, generated-looking
        scene(player=30),                                     # nothing
        scene(player=30, laser_col=60),                       # a streak in another column
        scene(player=30, laser_col=30, laser_rows=(10, 18)),  # high up: an old shot, not a new one
        scene(laser_col=30),                                  # no player to fire from
    ])
    np.testing.assert_array_equal(detectors.laser_present(frames), [True, True, False, False, False, False])


def test_calibration_recovers_a_linear_ram_map():
    ram_x = np.arange(35, 118, 3)
    frames = np.stack([scene(player=0.5 * r + 1.0) for r in ram_x] + [scene()])  # last frame: no player
    ram = np.zeros((len(frames), 128), np.uint8)
    ram[:-1, 28] = ram_x
    cal = detectors.calibrate_player_x(frames, ram)
    assert cal["n"] == len(ram_x)
    assert cal["mae_px"] < 0.5 and abs(cal["slope"] - 0.5) < 0.02


def test_frames_from_model_output_round_trip():
    import torch

    f = scene(player=20, laser_col=20)
    x = torch.from_numpy(f).permute(2, 0, 1).float()[None] / 127.5 - 1
    np.testing.assert_array_equal(detectors.to_uint8_frames(x), f[None])


# --- Action mapping ------------------------------------------------------------------------


def test_action_to_move_fire_covers_every_action_consistently():
    assert sorted(ACTION_TO_MOVE_FIRE) == list(range(NUM_ACTIONS))
    assert len(set(ACTION_TO_MOVE_FIRE.values())) == NUM_ACTIONS  # one action per (move, fire)
    for a, (move, fire) in ACTION_TO_MOVE_FIRE.items():
        name = ACTION_MEANINGS[a]
        assert fire == ("FIRE" in name)
        assert move == (1 if "RIGHT" in name else 2 if "LEFT" in name else 0)
