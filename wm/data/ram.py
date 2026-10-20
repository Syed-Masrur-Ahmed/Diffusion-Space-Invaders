"""Verified Space Invaders RAM addresses, for coverage statistics only; never a model input (D8).

Started from the AtariARI annotations (Anand et al., 2019) and checked against frames, rewards,
and `info["lives"]` with `scripts/coverage_report.py --verify-ram`
(`docs/figures/cp2_ram_verify.png`). AtariARI's `player_score=23` and `enemies_y=24` did not
verify (the score is BCD in 102/104, the alien block row is 16); `missiles_y=9` is dropped
because the player laser and alien bullets share a colour, so no clean frame measure exists.
"""

import numpy as np

INVADERS_LEFT = 17  # 36 at the start of a wave, -1 per alien killed, back to 36 when a new wave starts
LIVES = 73  # equals info["lives"], except on the game-over frame
PLAYER_X = 28  # linear in the player sprite's screen column
ALIENS_X = 26  # linear in the alien block's left column
ALIENS_Y = 16  # linear in the alien block's top row
SCORE_HI, SCORE_LO = 102, 104  # score as 4 BCD digits
UFO_STATE = 76  # >= UFO_ACTIVE_MIN while the UFO crosses the screen
UFO_ACTIVE_MIN = 4

WAVE_SIZE = 36


def _bcd(byte: np.ndarray) -> np.ndarray:
    return (byte >> 4) * 10 + (byte & 0x0F)


def score(ram: np.ndarray) -> np.ndarray:
    """Game score from RAM of shape (..., 128)."""
    ram = ram.astype(np.int64)
    return _bcd(ram[..., SCORE_HI]) * 100 + _bcd(ram[..., SCORE_LO])


def ufo_active(ram: np.ndarray) -> np.ndarray:
    """Whether the UFO is on screen, from RAM of shape (..., 128)."""
    return ram[..., UFO_STATE] >= UFO_ACTIVE_MIN
