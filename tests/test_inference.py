"""WorldModel tests (4.10): fake checkpoints built from TinyDenoiserNet, plus one from the real trainer."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.bench_sampling import main as bench_main
from scripts.bench_sampling import propose_d11
from tests.fixtures import TinyDenoiserNet, make_fake_shard
from wm.config import load_config
from wm.data.schema import NULL_ACTION
from wm.diffusion.edm import EDMConfig
from wm.diffusion.sampler import SamplerConfig
from wm.inference import FRAME_SIZE, WorldModel
from wm.models.unet import UNet, UNetConfig
from wm.train.trainer import DataSection, ModelSection, RunConfig, TrainSection, load_run_config, run

REPO = Path(__file__).resolve().parents[1]
K, C, B = 4, 3, 3
H = W = FRAME_SIZE


def _random_net(seed: int) -> TinyDenoiserNet:
    """Tiny net with a non-zero output layer, so predictions depend on context and actions."""
    torch.manual_seed(seed)
    net = TinyDenoiserNet(C=C, K=K, hidden=8)
    torch.nn.init.normal_(net.out.weight, std=0.1)
    return net


def run_config(arch: str = "tiny", **model) -> RunConfig:
    return RunConfig(name="test", seed=0, data=DataSection(K=K), model=ModelSection(arch=arch, tiny_hidden=8, **model),
                     edm=EDMConfig(), sampler=SamplerConfig(num_steps=2), train=TrainSection())


def write_ckpt(path: Path, cfg: RunConfig, model: torch.nn.Module, ema: torch.nn.Module) -> Path:
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": {}, "scheduler": {},
                "step": 7, "config": dataclasses.asdict(cfg), "rng_states": {}}, path)
    return path


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    return write_ckpt(tmp_path / "ckpt.pt", run_config(), _random_net(0), _random_net(1))


def inputs(seed: int = 0, n: int = B) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    context = torch.rand((n, K, C, H, W), generator=g) * 2 - 1
    actions = torch.randint(0, NULL_ACTION, (n, K), generator=g)
    return context, actions


def test_shapes_range_and_attributes(ckpt: Path):
    wm = WorldModel.from_checkpoint(ckpt, device="cpu")
    assert (wm.K, wm.C, wm.H, wm.W) == (K, C, H, W)
    context, actions = inputs()
    out = wm.predict_next(context, actions, generator=torch.Generator().manual_seed(0))
    assert out.shape == (B, C, H, W) and out.dtype == torch.float32
    assert out.min() >= -1 and out.max() <= 1
    assert not wm.denoiser.training


def test_deterministic_with_generator(ckpt: Path):
    wm = WorldModel.from_checkpoint(ckpt, device="cpu")
    context, actions = inputs()
    a, b, c = (wm.predict_next(context, actions, generator=torch.Generator().manual_seed(s)) for s in (5, 5, 6))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_overrides_reach_the_sampler(ckpt: Path):
    wm = WorldModel.from_checkpoint(ckpt, device="cpu")
    calls = []
    wm.denoiser.net.register_forward_pre_hook(lambda m, args: calls.append((args[0].shape[0], args[4].clone())))
    context, actions = inputs()
    wm.predict_next(context, actions)
    assert len(calls) == 2  # sampler.num_steps from the checkpoint config
    assert all(lvl.eq(0).all() for _, lvl in calls)
    calls.clear()
    wm.predict_next(context, actions, num_steps=4, ctx_level=3)
    assert len(calls) == 4 and all(n == B and lvl.eq(3).all() for n, lvl in calls)
    calls.clear()
    wm.predict_next(context, actions, num_steps=1, cfg_scale=2.0)
    assert [n for n, _ in calls] == [2 * B]  # CFG doubles the batch in one pass


def test_cfg_scale_one_is_the_plain_path(ckpt: Path):
    wm = WorldModel.from_checkpoint(ckpt, device="cpu")
    batch_sizes = []
    wm.denoiser.net.register_forward_pre_hook(lambda m, args: batch_sizes.append(args[0].shape[0]))
    context, actions = inputs()
    plain = wm.predict_next(context, actions, generator=torch.Generator().manual_seed(0))
    one = wm.predict_next(context, actions, cfg_scale=1.0, generator=torch.Generator().manual_seed(0))
    assert torch.equal(plain, one)
    assert batch_sizes == [B] * 4  # 2 steps per call, never a doubled batch
    guided = wm.predict_next(context, actions, cfg_scale=3.0, generator=torch.Generator().manual_seed(0))
    assert not torch.equal(plain, guided)


def test_batched_matches_per_item(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    wm = WorldModel.from_checkpoint(ckpt, device="cpu")
    noise = torch.randn((1, C, H, W), generator=torch.Generator().manual_seed(0))
    real_randn = torch.randn

    def same_noise_per_item(size, *args, **kwargs):  # so batch size does not change any item's noise
        return noise.expand(size[0], -1, -1, -1).clone() if len(size) == 4 else real_randn(size, *args, **kwargs)

    monkeypatch.setattr(torch, "randn", same_noise_per_item)
    context, actions = inputs(n=4)
    batched = wm.predict_next(context, actions, ctx_level=2)
    single = torch.cat([wm.predict_next(context[i:i + 1], actions[i:i + 1], ctx_level=2) for i in range(4)])
    assert torch.allclose(batched, single, atol=1e-5)
    assert not torch.allclose(batched[0], batched[1])  # items really differ


def test_use_ema_picks_the_ema_weights(tmp_path: Path):
    model, ema = _random_net(0), _random_net(1)
    path = write_ckpt(tmp_path / "ckpt.pt", run_config(), model, ema)
    for use_ema, want in ((True, ema), (False, model)):
        got = WorldModel.from_checkpoint(path, device="cpu", use_ema=use_ema).denoiser.net.state_dict()
        assert all(torch.equal(got[k], v) for k, v in want.state_dict().items())
    context, actions = inputs()
    preds = [WorldModel.from_checkpoint(path, device="cpu", use_ema=e).predict_next(
        context, actions, generator=torch.Generator().manual_seed(0)) for e in (True, False)]
    assert not torch.equal(*preds)


def test_unet_checkpoint_and_grayscale(tmp_path: Path):
    cfg = run_config("unet", ch=[8, 16, 16], n_res=1, emb_dim=32)
    cfg.data.grayscale = True
    torch.manual_seed(0)
    net = UNet(UNetConfig(C=1, K=K, H=H, ch=[8, 16, 16], n_res=1, emb_dim=32))
    path = write_ckpt(tmp_path / "ckpt.pt", cfg, net, net)
    wm = WorldModel.from_checkpoint(path, device="cpu")
    assert isinstance(wm.denoiser.net, UNet) and wm.C == 1
    out = wm.predict_next(torch.zeros((2, K, 1, H, W)), torch.zeros((2, K), dtype=torch.long), num_steps=1)
    assert out.shape == (2, 1, H, W)


def test_strict_loading(tmp_path: Path):
    bad = _random_net(0).state_dict()
    bad = {f"net.{k}": v for k, v in bad.items()}  # an EDMDenoiser state dict, not the network's
    path = tmp_path / "ckpt.pt"
    torch.save({"model": bad, "ema": bad, "optimizer": {}, "scheduler": {}, "step": 0,
                "config": dataclasses.asdict(run_config()), "rng_states": {}}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        WorldModel.from_checkpoint(path, device="cpu")


def test_loads_a_checkpoint_written_by_the_trainer(tmp_path: Path):
    make_fake_shard(tmp_path / "data/fake", 3, [10, 12, 9], H, W)
    sets = {"data.root": tmp_path / "data", "data.dirs": "[fake]", "data.K": K, "data.batch_size": 2,
            "data.num_workers": 0, "data.val_fraction": 0.3, "model.arch": "tiny", "train.steps": 3,
            "train.runs_dir": tmp_path / "runs", "train.warmup_steps": 1, "train.val_windows": 2,
            "train.n_samples": 1, "train.ema_decay": 0.5}
    cfg = load_run_config(REPO / "configs/train/debug.yaml", [f"{k}={v}" for k, v in sets.items()])
    assert run(cfg) == 0
    (path,) = (tmp_path / "runs").glob("*_debug/checkpoints/ckpt_last.pt")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    wm = WorldModel.from_checkpoint(path, device="cpu")
    assert all(torch.equal(wm.denoiser.net.state_dict()[k], v) for k, v in saved["ema"].items())
    context, actions = inputs(n=2)
    assert wm.predict_next(context, actions).shape == (2, C, H, W)


def test_frozen_from_checkpoint_positional_args(ckpt: Path):
    wm = WorldModel.from_checkpoint(ckpt, "cpu", True, False)
    context, actions = inputs(n=1)
    out = wm.predict_next(context, actions, num_steps=1, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, C, H, W)


def _formats(net: torch.nn.Module) -> list[tuple[bool, bool]]:
    seen = []
    net.register_forward_pre_hook(lambda m, args: seen.append((
        args[0].is_contiguous(memory_format=torch.channels_last), args[0].is_contiguous(),
        args[2].is_contiguous(memory_format=torch.channels_last))))
    return seen


def test_channels_last_is_the_default(ckpt: Path):
    wm = WorldModel.from_checkpoint(ckpt, device="cpu")
    assert wm.channels_last
    weight = wm.denoiser.net.conv_in.weight
    assert weight.is_contiguous(memory_format=torch.channels_last) and not weight.is_contiguous()
    seen = _formats(wm.denoiser.net)
    context, actions = inputs(n=1)
    wm.predict_next(context, actions, num_steps=1, generator=torch.Generator().manual_seed(0))
    assert seen and all(cl_x and not nchw and cl_ctx for cl_x, nchw, cl_ctx in seen)


def test_channels_last_can_be_disabled(ckpt: Path):
    wm = WorldModel.from_checkpoint(ckpt, device="cpu", channels_last=False)
    assert not wm.channels_last
    assert wm.denoiser.net.conv_in.weight.is_contiguous()
    seen = _formats(wm.denoiser.net)
    context, actions = inputs(n=1)
    wm.predict_next(context, actions, num_steps=1, generator=torch.Generator().manual_seed(0))
    assert seen and all(nchw and not cl_x for cl_x, nchw, _ in seen)


def test_channels_last_matches_contiguous(ckpt: Path):
    context, actions = inputs()
    kwargs = dict(generator=torch.Generator().manual_seed(0))
    cl = WorldModel.from_checkpoint(ckpt, device="cpu", channels_last=True).predict_next(context, actions, **kwargs)
    kwargs["generator"].manual_seed(0)
    plain = WorldModel.from_checkpoint(ckpt, device="cpu", channels_last=False).predict_next(context, actions, **kwargs)
    # Memory format changes conv accumulation order by a fraction of a ulp on this net.
    assert torch.allclose(cl, plain, atol=1e-5, rtol=1e-5)


def test_channels_last_matches_on_a_unet(tmp_path: Path):
    cfg = run_config("unet", ch=[8, 16, 16], n_res=1, emb_dim=32)
    torch.manual_seed(0)
    net = UNet(UNetConfig(C=C, K=K, H=H, ch=[8, 16, 16], n_res=1, emb_dim=32))
    g = torch.Generator().manual_seed(1)  # a zero output conv would make both paths identical
    with torch.no_grad():
        for p in net.parameters():
            if not torch.any(p):
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    path = write_ckpt(tmp_path / "ckpt.pt", cfg, net, net)
    context, actions = inputs(n=1)
    preds = []
    for channels_last in (True, False):
        wm = WorldModel.from_checkpoint(path, device="cpu", channels_last=channels_last)
        preds.append(wm.predict_next(context, actions, num_steps=1, generator=torch.Generator().manual_seed(0)))
    assert torch.allclose(*preds, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("amp", [torch.bfloat16, torch.float16])
def test_autocast_stays_inside_predict_and_is_deterministic(ckpt: Path, amp: torch.dtype):
    wm = WorldModel.from_checkpoint(ckpt, device="cpu", amp_dtype=amp)
    dtypes = []
    wm.denoiser.net.register_forward_hook(lambda m, inp, out: dtypes.append(out.dtype))
    context, actions = inputs()
    out = wm.predict_next(context, actions, generator=torch.Generator().manual_seed(0))
    again = wm.predict_next(context, actions, generator=torch.Generator().manual_seed(0))
    assert dtypes and set(dtypes) == {amp}
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    assert out.min() >= -1 and out.max() <= 1
    assert torch.equal(out, again)


def test_amp_dtype_must_be_fp16_or_bf16(ckpt: Path):
    with pytest.raises(ValueError, match="amp_dtype"):
        WorldModel.from_checkpoint(ckpt, device="cpu", amp_dtype=torch.float32)


def test_propose_d11_picks_the_fewest_steps_within_tolerance():
    rows = [
        {"device": "cuda", "method": "euler", "num_steps": 10, "steps_per_s": 4.0, "psnr": 30.0},
        {"device": "cuda", "method": "euler", "num_steps": 5, "steps_per_s": 16.0, "psnr": 29.8},
        {"device": "cuda", "method": "euler", "num_steps": 3, "steps_per_s": 40.0, "psnr": 29.6},
        {"device": "cuda", "method": "euler", "num_steps": 1, "steps_per_s": 100.0, "psnr": 20.0},
        {"device": "cuda", "method": "heun", "num_steps": 2, "steps_per_s": 20.0, "psnr": 30.0},
        {"device": "cpu", "method": "euler", "num_steps": 1, "steps_per_s": 80.0, "psnr": 30.0},
    ]
    got = propose_d11(rows, reference_steps=10, method="euler", tol_db=0.3, min_steps_per_s=15, device="cuda")
    assert got["num_steps"] == 5 and got["psnr"] == 29.8 and got["reference_psnr"] == 30.0
    assert got["reason"] is None


def test_propose_d11_keeps_ten_steps_when_fewer_steps_lose_too_much():
    rows = [
        {"device": "cuda", "method": "euler", "num_steps": 10, "steps_per_s": 20.0, "psnr": 30.0},
        {"device": "cuda", "method": "euler", "num_steps": 5, "steps_per_s": 40.0, "psnr": 29.0},
    ]
    got = propose_d11(rows, reference_steps=10, method="euler", tol_db=0.3, min_steps_per_s=15, device="cuda")
    assert got["num_steps"] == 10


def test_propose_d11_explains_when_it_cannot_choose():
    missing_device = propose_d11(
        [{"device": "cpu", "method": "euler", "num_steps": 10, "steps_per_s": 100.0, "psnr": 30.0}], device="cuda")
    assert missing_device["num_steps"] is None and "cuda" in missing_device["reason"]
    too_slow = propose_d11([
        {"device": "cuda", "method": "euler", "num_steps": 10, "steps_per_s": 4.0, "psnr": 30.0},
        {"device": "cuda", "method": "euler", "num_steps": 3, "steps_per_s": 40.0, "psnr": 28.0},
    ], device="cuda")
    assert too_slow["num_steps"] is None and too_slow["reference_psnr"] == 30.0
    no_reference = propose_d11(
        [{"device": "cuda", "method": "euler", "num_steps": 3, "steps_per_s": 40.0, "psnr": 30.0}], device="cuda")
    assert no_reference["num_steps"] is None and "10" in no_reference["reason"]


def test_bench_config_lists_the_roadmap_sweep():
    cfg = load_config(REPO / "configs/eval/bench_sampling.yaml")
    assert cfg["num_steps"] == [1, 2, 3, 5, 10]
    assert cfg["methods"] == ["euler", "heun"]
    assert cfg["quality_n"] == 512 and cfg["metrics"] == ["psnr", "lpips"]
    assert cfg["psnr_tol_db"] == 0.3 and cfg["min_steps_per_s"] == 15
    assert cfg["reference_steps"] == 10 and cfg["proposal_method"] == "euler"
    assert cfg["proposal_device"] == "cuda" and cfg["devices"] == ["cpu", "cuda"]


def test_bench_sampling_writes_a_table_and_a_plot(tmp_path: Path, ckpt: Path):
    make_fake_shard(tmp_path / "data", 2, [8, 8], H, W)
    cfg = {"model": str(ckpt), "out_dir": str(tmp_path / "out"), "seed": 0,
           "data": {"dataset_dirs": [str(tmp_path / "data")], "val_fraction": 1.0, "batch_size": 4, "num_workers": 0},
           "quality_n": 4, "metrics": ["psnr"], "num_steps": [1, 2], "methods": ["euler", "heun"],
           "devices": ["cpu"], "cfg_scale": None, "ctx_level": 0, "channels_last": True, "amp_dtype": None,
           "compile": False, "use_ema": True, "speed_warmup": 0, "speed_repeats": 1, "reference_steps": 2,
           "psnr_tol_db": 0.3, "min_steps_per_s": 0, "proposal_device": "cpu", "proposal_method": "euler"}
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = bench_main(["--config", str(path), "--set", "quality_n=2"])
    result = json.loads((out / "results.json").read_text())
    assert result["quality_n"] == 2 and result["n"] == 2
    assert {(r["method"], r["num_steps"]) for r in result["rows"]} == {("euler", 1), ("euler", 2), ("heun", 1), ("heun", 2)}
    assert all(r["device"] == "cpu" and r["steps_per_s"] > 0 and r["psnr"] is not None for r in result["rows"])
    by_key = {(r["method"], r["num_steps"]): r["psnr"] for r in result["rows"]}
    assert by_key[("euler", 2)] != by_key[("heun", 2)]  # N=2 is the first schedule where Heun corrects
    assert result["proposal"]["num_steps"] in (1, 2)
    text = (out / "results.csv").read_text()
    assert text.startswith("device,method,num_steps,steps_per_s,psnr")
    assert (out / "quality_vs_speed.png").is_file()
