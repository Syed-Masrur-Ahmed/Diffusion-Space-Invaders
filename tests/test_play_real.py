import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "play_real.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("play_real", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_display_array_widens_and_transposes():
    play_real = _load_script()
    frame = np.arange(210 * 160 * 3, dtype=np.uint32).reshape(210, 160, 3).astype(np.uint8)
    out = play_real.display_array(frame, scale=3)
    assert out.shape == (160 * 2 * 3, 210 * 3, 3)  # pygame surfarray is (W, H, 3)
    assert np.array_equal(out[::6, ::3].swapaxes(0, 1), frame)
    assert frame.shape == (210, 160, 3)  # env frame untouched


@pytest.mark.env
def test_play_loop_runs_headless():
    # conftest sets SDL_VIDEODRIVER=dummy.
    play_real = _load_script()
    assert play_real.run(seed=0, fps=0, scale=1, max_ticks=10) == 10
