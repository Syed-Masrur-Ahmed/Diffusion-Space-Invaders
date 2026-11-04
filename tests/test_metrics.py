import math

import pytest
import torch

from tests.fixtures import make_fake_shard
from wm.data.dataset import WindowDataset
from wm.eval import metrics
from wm.eval.baselines import CopyLastFrame


def _images(n: int = 4, C: int = 3, H: int = 84, W: int = 84, seed: int = 0) -> torch.Tensor:
    return torch.rand(n, C, H, W, generator=torch.Generator().manual_seed(seed))


# --- PSNR and SSIM ------------------------------------------------------------------------


def test_psnr_identical_is_capped_at_100():
    x = _images()
    out = metrics.psnr(x, x.clone())
    assert out.shape == (4,)
    assert torch.equal(out, torch.full((4,), 100.0))


def test_psnr_constant_offset_of_point_one_is_20_db():
    target = _images() * 0.8
    out = metrics.psnr(target + 0.1, target)
    assert torch.allclose(out, torch.full((4,), 20.0), atol=1e-4)


def test_psnr_is_per_image():
    target = _images(2) * 0.5
    pred = target.clone()
    pred[1] += 0.1
    out = metrics.psnr(pred, target)
    assert out[0] == 100.0 and math.isclose(out[1].item(), 20.0, abs_tol=1e-4)


def test_ssim_identical_is_one():
    x = _images()
    out = metrics.ssim(x, x.clone())
    assert out.shape == (4,)
    assert torch.allclose(out, torch.ones(4), atol=1e-6)


def test_ssim_drops_for_noise():
    x = _images()
    noisy = (x + 0.3 * torch.randn(x.shape, generator=torch.Generator().manual_seed(1))).clamp(0, 1)
    assert (metrics.ssim(noisy, x) < 0.9).all()


def test_greyscale_inputs_work_for_psnr_and_ssim():
    x = _images(C=1)
    assert torch.equal(metrics.psnr(x, x), torch.full((4,), 100.0))
    assert torch.allclose(metrics.ssim(x, x), torch.ones(4), atol=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.psnr(_images(H=84), _images(H=42))


# --- LPIPS ------------------------------------------------------------------------------


@pytest.mark.network
def test_lpips_identical_is_about_zero_and_model_is_cached():
    x = _images()
    out = metrics.lpips(x, x.clone())
    assert out.shape == (4,)
    assert (out.abs() < 1e-4).all()
    assert metrics.lpips(_images(seed=1), x).min() > 0.01
    assert metrics._lpips_model(torch.device("cpu")) is metrics._lpips_model(torch.device("cpu"))


@pytest.mark.network
def test_lpips_accepts_greyscale():
    x = _images(C=1)
    assert (metrics.lpips(x, x).abs() < 1e-4).all()


# --- CopyLastFrame -------------------------------------------------------------------------


def test_copy_last_returns_last_context_frame():
    model = CopyLastFrame(K=4, C=3, H=8, W=8)
    assert (model.K, model.C, model.H, model.W) == (4, 3, 8, 8)
    context = torch.rand(2, 4, 3, 8, 8) * 2 - 1
    out = model.predict_next(context, torch.zeros(2, 4, dtype=torch.long))
    assert out.shape == (2, 3, 8, 8)
    assert torch.equal(out, context[:, -1])
    out += 1  # a copy, not a view into the context
    assert not torch.equal(out, context[:, -1])


def test_copy_last_psnr_on_fake_shard_matches_consecutive_encoded_frames(tmp_path):
    # Rows stay < 256, so consecutive encoded frames differ by exactly 1 in R and 0 in G, B.
    make_fake_shard(tmp_path, 3, [30, 40, 50], 8, 8)
    ds = WindowDataset([tmp_path], K=4, split="train", val_fraction=0.0)
    batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
    model = CopyLastFrame(K=4, C=3, H=8, W=8)
    pred = model.predict_next(batch["context"], batch["actions"])
    out = metrics.psnr((pred + 1) / 2, (batch["target"] + 1) / 2)
    expected = 10 * math.log10(3 * 255 ** 2)  # mse = (1/255)^2 / 3
    assert len(out) == 120 - 3 * 4
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-3)


# --- Sample grid ---------------------------------------------------------------------------


def test_save_grid_layout_and_diff_column(tmp_path):
    import imageio.v3 as iio

    from wm.eval.visualize import GAP, SCALE, save_grid

    B, K, H, W = 2, 4, 6, 5
    g = torch.Generator().manual_seed(0)
    to_unit = lambda u: u.float() / 127.5 - 1  # noqa: E731
    context = to_unit(torch.randint(0, 256, (B, K, 3, H, W), generator=g))
    target = to_unit(torch.randint(0, 256, (B, 3, H, W), generator=g))
    pred = to_unit(torch.randint(0, 256, (B, 3, H, W), generator=g))
    path = tmp_path / "grid.png"
    save_grid(path, context, target, pred)

    img = iio.imread(path)
    th, tw = H * SCALE, W * SCALE
    assert img.shape == (B * (th + GAP) - GAP, 4 * (tw + GAP) - GAP, 3) and img.dtype == "uint8"

    def tile(r: int, c: int):
        y, x = r * (th + GAP), c * (tw + GAP)
        return torch.from_numpy(img[y:y + th:SCALE, x:x + tw:SCALE].copy()).permute(2, 0, 1)

    as_u8 = lambda x: ((x + 1) * 127.5).round().to(torch.uint8)  # noqa: E731
    for r in range(B):
        assert torch.equal(tile(r, 0), as_u8(context[r, -1]))
        assert torch.equal(tile(r, 1), as_u8(target[r]))
        assert torch.equal(tile(r, 2), as_u8(pred[r]))
        diff = ((as_u8(target[r]).int() - as_u8(pred[r]).int()).abs() * 4).clamp(max=255)
        assert torch.equal(tile(r, 3), diff.to(torch.uint8))


def test_save_grid_greyscale(tmp_path):
    import imageio.v3 as iio

    from wm.eval.visualize import save_grid

    x = torch.zeros(1, 1, 4, 4)
    save_grid(tmp_path / "g.png", x[:, None].expand(1, 2, 1, 4, 4), x, x)
    assert iio.imread(tmp_path / "g.png").shape[-1] == 3


# --- eval_single_step.py --------------------------------------------------------------------


def _load_eval_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_single_step.py"
    spec = importlib.util.spec_from_file_location("eval_single_step", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pick_indices_spread_and_capped():
    ess = _load_eval_script()
    assert ess.pick_indices(10, 20).tolist() == list(range(10))
    idx = ess.pick_indices(1000, 5)
    assert idx.tolist() == [0, 250, 500, 749, 999]


def test_evaluate_copy_last_on_fake_shard(tmp_path):
    ess = _load_eval_script()
    make_fake_shard(tmp_path, 3, [30, 40, 50], 8, 8)
    ds = WindowDataset([tmp_path], K=4, split="train", val_fraction=0.0)
    out = ess.evaluate(CopyLastFrame(4, 3, 8, 8), ds, ess.pick_indices(len(ds), 50),
                       ["psnr", "ssim"], batch_size=16, device=torch.device("cpu"))
    assert set(out["values"]) == {"psnr", "ssim"} and len(out["values"]["psnr"]) == 50
    assert torch.allclose(out["values"]["psnr"], torch.full((50,), 10 * math.log10(3 * 255 ** 2)), atol=1e-3)
    assert out["grid"]["context"].shape == (8, 4, 3, 8, 8)
    s = ess.summarize(out["values"]["psnr"])
    assert s["n"] == 50 and s["ci95"] < 1e-3


def test_eval_single_step_main_writes_json_and_grid(tmp_path):
    import json

    ess = _load_eval_script()
    data = tmp_path / "data"
    make_fake_shard(data, 4, [30, 30, 30, 30], 8, 8)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"model: copy_last\nn: 16\ndata:\n  dataset_dirs: [{data}]\n  K: 4\n  grayscale: false\n"
                   f"  val_fraction: 1.0\n  batch_size: 8\n  num_workers: 0\n"
                   f"metrics: [psnr, ssim]\nout_dir: {tmp_path / 'out'}\nseed: 0\n"
                   f"num_steps: null\ncfg_scale: null\n")
    out_dir = ess.main(["--config", str(cfg), "--set", "n=12"])
    result = json.loads((out_dir / "metrics.json").read_text())
    assert result["model"] == "copy_last" and result["split"] == "val" and result["n"] == 12
    assert set(result["metrics"]) == {"psnr", "ssim"}
    assert (out_dir / "grid.png").exists()
