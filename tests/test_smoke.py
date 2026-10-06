import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import wm

MODULES = sorted(m.name for m in pkgutil.walk_packages(wm.__path__, "wm."))


def test_expected_modules_exist():
    expected = {
        "wm.config", "wm.controls", "wm.env", "wm.inference",
        "wm.agents.policies", "wm.agents.pretrained",
        "wm.data.schema", "wm.data.writer", "wm.data.reader", "wm.data.dataset",
        "wm.data.ram", "wm.data.coverage",
        "wm.diffusion.edm", "wm.diffusion.sampler",
        "wm.models.embeddings", "wm.models.unet", "wm.models.idm",
        "wm.train.trainer",
        "wm.eval.metrics", "wm.eval.baselines", "wm.eval.visualize", "wm.eval.rollout",
        "wm.eval.plots", "wm.eval.detectors", "wm.eval.counterfactual",
    }
    assert expected <= set(MODULES)


def test_every_module_imports():
    for name in MODULES:
        importlib.import_module(name)


def test_wm_never_imports_pygame():
    # Fresh interpreter, so pygame imported by other tests can't hide a violation.
    code = f"import importlib, sys\nfor m in {MODULES!r}: importlib.import_module(m)\nprint('pygame' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(wm.__file__).parents[1])
    assert out.stdout.strip().splitlines()[-1] == "False"
