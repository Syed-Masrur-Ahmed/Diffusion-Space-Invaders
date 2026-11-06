"""scripts/memorize.py: fine-tune a checkpoint on n train windows and score it on the same windows."""

import importlib.util
import json
from pathlib import Path

import torch

from tests.fixtures import make_fake_shard
from wm.inference import FRAME_SIZE
from wm.train.trainer import load_run_config, run

REPO = Path(__file__).resolve().parents[1]
K = 4


def _memorize():
    spec = importlib.util.spec_from_file_location("memorize", REPO / "scripts" / "memorize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_checkpoint(tmp_path: Path) -> Path:
    make_fake_shard(tmp_path / "data/fake", 3, [12, 14, 10], FRAME_SIZE, FRAME_SIZE)
    sets = {"data.root": tmp_path / "data", "data.dirs": "[fake]", "data.K": K, "data.batch_size": 2,
            "data.num_workers": 0, "data.val_fraction": 0.3, "model.arch": "tiny", "train.steps": 2,
            "train.runs_dir": tmp_path / "runs", "train.warmup_steps": 1, "train.val_windows": 2,
            "train.n_samples": 1}
    assert run(load_run_config(REPO / "configs/train/debug.yaml", [f"{k}={v}" for k, v in sets.items()])) == 0
    (path,) = (tmp_path / "runs").glob("*_debug/checkpoints/ckpt_last.pt")
    return path


def _eval_cfg(tmp_path: Path) -> dict:
    return {"data": {"dataset_dirs": [str(tmp_path / "data/fake")], "val_fraction": 0.3, "batch_size": 8,
                     "num_workers": 0},
            "metrics": ["psnr", "ssim"], "seed": 0, "num_steps": None, "cfg_scale": None}


def test_fine_tunes_on_the_windows_it_scores(tmp_path: Path):
    memorize = _memorize()
    ckpt = _tiny_checkpoint(tmp_path)
    before = ckpt.read_bytes()
    out = memorize.run(str(ckpt), _eval_cfg(tmp_path), n=5, steps=40, lr=3e-3, device=torch.device("cpu"))
    assert set(out["metrics"]) == {"copy_last", "before", "after"}
    assert all(set(m) == {"psnr", "ssim"} and m["psnr"]["n"] == 5 for m in out["metrics"].values())
    assert len(out["losses"]) == 40 and all(torch.isfinite(torch.tensor(out["losses"])))
    assert sum(out["losses"][-5:]) < sum(out["losses"][:5])  # it fits the 5 windows
    assert out["metrics"]["after"]["psnr"]["mean"] > out["metrics"]["before"]["psnr"]["mean"]
    assert out["grid"]["target"].shape == (5, 3, FRAME_SIZE, FRAME_SIZE)  # all 5 windows (< GRID_ROWS)
    assert ckpt.read_bytes() == before  # the checkpoint file is never modified


def test_main_writes_json_and_grid(tmp_path: Path):
    memorize = _memorize()
    ckpt = _tiny_checkpoint(tmp_path)
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(json.dumps(_eval_cfg(tmp_path)))  # JSON is valid YAML
    out_dir = tmp_path / "out"
    memorize.main(["--config", str(cfg_path), "--checkpoint", str(ckpt), "--n", "3", "--steps", "2",
                   "--out", str(out_dir)])
    result = json.loads((out_dir / "metrics.json").read_text())
    assert result["n"] == 3 and result["steps"] == 2 and len(result["losses"]) == 2
    assert (out_dir / "grid.png").stat().st_size > 0
