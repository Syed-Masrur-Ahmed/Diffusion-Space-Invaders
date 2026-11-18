"""Pixel detectors for controllability evaluation on real and generated frames (ROADMAP CP5-A).

Calibrated on stored 84x84 `max`-resized frames (D10), where the player is pure green
(50,132,50) in raw rows 185-194 and the player's laser is a 2-column grey (142,142,142) streak
that starts just above the player and rises ~4 rows per agent step. Alien bullets are the same
grey, so `laser_present` only looks at the launch zone above the player, where a new shot is.
Generated frames are near the palette but not on it, so colours match within a tolerance.
Bands are raw-screen rows scaled to the frame height, so the detectors also work at 96.
"""

import math

import numpy as np
import torch
from torch import Tensor

from wm.data import ram as ram_map

RAW_HEIGHT = 210
PLAYER_RGB = (50, 132, 50)
LASER_RGB = (142, 142, 142)
PLAYER_RAW_ROWS = (185, 195)  # rows 74-77 at 84; the ground starts at raw row 195
LAUNCH_RAW_ROWS = (160, 185)  # rows 64-73 at 84: where a shot is on the step after FIRE
COLOUR_TOL = 30  # per-channel distance; the closest other palette colour (the ground) is 43 away
MIN_PLAYER_PIXELS = 4  # the full sprite is ~17 pixels at 84
LASER_HALF_WIDTH = (2, 3)  # columns x-2 .. x+3 around the player centroid hold its laser


def _rows(raw_rows: tuple[int, int], height: int) -> slice:
    return slice(raw_rows[0] * height // RAW_HEIGHT, math.ceil(raw_rows[1] * height / RAW_HEIGHT))


def _near(pixels: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    return (np.abs(pixels.astype(np.int16) - np.array(rgb, np.int16)) < COLOUR_TOL).all(-1)


def to_uint8_frames(x: Tensor) -> np.ndarray:
    """Model frames (..., 3, H, W) in [-1,1] -> uint8 (..., H, W, 3), the inverse of the 4.5 normalization."""
    return ((x.detach().float().cpu() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).movedim(-3, -1).numpy()


def player_x(frames: np.ndarray) -> np.ndarray:
    """(N,) float column of the player's centroid in (N,H,W,3) uint8 frames; NaN when no player is visible."""
    mask = _near(frames[:, _rows(PLAYER_RAW_ROWS, frames.shape[1])], PLAYER_RGB)
    count = mask.sum(axis=(1, 2))
    cols = (mask.sum(axis=1) * np.arange(frames.shape[2])).sum(axis=1)
    return np.where(count >= MIN_PLAYER_PIXELS, cols / np.maximum(count, 1), np.nan)


def laser_present(frames: np.ndarray) -> np.ndarray:
    """(N,) bool: a laser in the launch zone above the player, i.e. a shot fired in the last few steps."""
    x = player_x(frames)
    zone = _near(frames[:, _rows(LAUNCH_RAW_ROWS, frames.shape[1])], LASER_RGB)
    out = np.zeros(len(frames), bool)
    for i in np.flatnonzero(~np.isnan(x)):
        c = int(round(x[i]))
        out[i] = zone[i, :, max(c - LASER_HALF_WIDTH[0], 0):c + LASER_HALF_WIDTH[1] + 1].any()
    return out


def calibrate_player_x(x: np.ndarray, ram: np.ndarray) -> dict[str, float]:
    """Least-squares map RAM player x -> detected column `x` (from `player_x`), and the detector's MAE (pixels) around it."""
    ok = ~np.isnan(x)
    r = ram[ok, ram_map.PLAYER_X].astype(np.float64)
    slope, intercept = np.polyfit(r, x[ok], 1)
    err = np.abs(x[ok] - (slope * r + intercept))
    return {"slope": float(slope), "intercept": float(intercept), "mae_px": float(err.mean()),
            "max_err_px": float(err.max()), "n": int(ok.sum()), "n_no_player": int((~ok).sum())}
