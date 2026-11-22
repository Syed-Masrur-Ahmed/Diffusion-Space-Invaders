"""Milestone 2 learning curves: TensorBoard export with binned train loss."""

import importlib.util
import json
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

REPO = Path(__file__).resolve().parents[1]


def _curves():
    spec = importlib.util.spec_from_file_location("m2_curves", REPO / "scripts/m2_curves.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_bins_train_loss_and_keeps_val_loss(tmp_path):
    run = tmp_path / "run"
    w = SummaryWriter(str(run / "tb"))
    for s in range(50, 2001, 50):
        w.add_scalar("train/loss", 1.0 if s <= 1000 else 3.0, s)
    for s in (1000, 2000):
        w.add_scalar("val/loss", s / 1000, s)
    w.close()
    out = _curves().main(["--runs", f"r0={run}", "--out", str(tmp_path / "out"), "--bin-steps", "1000"])
    c = json.loads((out / "r0.json").read_text())
    assert c["train"]["step"] == [1000, 2000]
    assert c["train"]["loss"] == pytest.approx([1.0, 3.0])
    assert c["val"] == {"step": [1000, 2000], "loss": pytest.approx([1.0, 2.0])}
