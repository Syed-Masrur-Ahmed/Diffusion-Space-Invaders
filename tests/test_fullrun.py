"""CP5-B: full dataset config, final training config, and the chained Slurm submitter."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.test_collect import _load_collect
from tests.test_noise_aug import _flat
from tests.test_trainer import DEBUG_YAML
from wm.config import load_config
from wm.data.schema import EPISODE_UID_STRIDE, EVAL_SEED_BASE
from wm.train.trainer import build_net, load_run_config

REPO = Path(__file__).resolve().parents[1]
FULL_YAML = REPO / "configs/data/full.yaml"
DEBUG_DATA_YAML = REPO / "configs/data/debug.yaml"
FINAL_YAML = REPO / "configs/train/final.yaml"
CHAIN = REPO / "scripts/slurm/final_chain.sh"
MAX_WORKERS = 16  # most collector workers the progress note tells Syed to use


# --- configs/data/full.yaml -------------------------------------------------------------------


def test_full_plan_is_valid_and_sized():
    cfg = load_config(FULL_YAML)
    assert cfg["out_dir"] == "data/full" and (cfg["resolution"], cfg["resize_method"]) == (84, "max")
    assert 2_000_000 <= cfg["total_steps"] <= 5_000_000 and cfg["total_episodes"] is None
    for workers in (1, 8, MAX_WORKERS):
        parts = _load_collect().plan_parts(cfg, workers)
        assert sum(p["budget"] for p in parts) == cfg["total_steps"]


def test_full_adds_low_epsilon_agent_and_keeps_random():
    jobs = load_config(FULL_YAML)["jobs"]
    eps = [j["epsilon"][1] for j in jobs if j["policy"] == "agent"]
    assert min(eps) <= 0.05  # late-wave coverage (CP2-A, D17)
    assert sum(j["fraction"] for j in jobs if j["policy"] == "random") >= 0.15  # action diversity


@pytest.mark.parametrize("debug_workers", [1, 4, 10])
def test_full_seeds_are_training_seeds_disjoint_from_debug(debug_workers):
    full, debug = load_config(FULL_YAML), load_config(DEBUG_DATA_YAML)
    lo = full["base_seed"]
    hi = lo + len(full["jobs"]) * MAX_WORKERS * EPISODE_UID_STRIDE
    debug_hi = debug["base_seed"] + len(debug["jobs"]) * debug_workers * EPISODE_UID_STRIDE
    assert hi <= EVAL_SEED_BASE
    assert lo >= debug_hi  # new episodes, so debug-trained models are never scored on them


# --- configs/train/final.yaml -----------------------------------------------------------------


def test_final_config_has_debug_keys_and_roadmap_settings():
    base, cfg = _flat(load_run_config(DEBUG_YAML)), _flat(load_run_config(FINAL_YAML))
    assert set(cfg) == set(base)
    assert cfg["name"] == "final"
    assert cfg["data.dirs"] == ["full", "human"] and cfg["data.K"] == 4
    assert 64 <= cfg["data.batch_size"] <= 128
    assert (cfg["train.aug_max"], cfg["train.ctx_noise_mode"], cfg["train.p_action_drop"]) == (0.7, "shared", 0.1)
    assert cfg["train.save_every"] == 10_000 and cfg["model.arch"] == "unet"
    assert cfg["sampler.num_steps"] == 2  # D11
    for k in base:  # EDM and sampler settings are the ones every earlier model used
        if k.startswith(("edm.", "sampler.")):
            assert cfg[k] == base[k], k


def test_final_model_is_base_unet():
    cfg = load_run_config(FINAL_YAML)
    assert list(cfg.model.ch) == load_config(REPO / "configs/model/base.yaml")["ch"][84]
    n = sum(p.numel() for p in build_net(cfg.model, C=3, K=cfg.data.K, H=84).parameters())
    assert 34_000_000 < n < 35_000_000  # 34.35M (ROADMAP 1.5, CP3)


# --- scripts/slurm/final_chain.sh -------------------------------------------------------------


def _chain(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(CHAIN), *args], cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "DRY_RUN": "1", **env})


def test_chain_dry_run_prints_afterany_chain():
    out = _chain({"N_JOBS": "3", "DATA_ROOT": "/scratch/data"}, "--set", "train.steps=10")
    assert out.returncode == 0, out.stderr
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("CONFIG=")]
    assert len(lines) == 3
    assert "--dependency" not in lines[0]
    assert "--dependency=afterany:<job 1>" in lines[1]
    assert "--dependency=afterany:<job 2>" in lines[2]
    for ln in lines:
        assert "CONFIG=configs/train/final.yaml" in ln and "scripts/slurm/train.sbatch" in ln
        assert ln.endswith("--set data.root=/scratch/data --set train.steps=10")
        assert "--partition=gpuq" in ln and "--time=" in ln


def test_chain_requires_data_root():
    out = _chain({"DATA_ROOT": ""})
    assert out.returncode != 0 and "DATA_ROOT" in out.stderr
