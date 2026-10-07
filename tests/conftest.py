import os

# Must be set before pygame initialises a display anywhere in the test session.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
