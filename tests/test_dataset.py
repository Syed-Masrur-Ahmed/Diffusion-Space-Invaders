"""Tests for WindowDataset and make_dataloader (ROADMAP 4.5, CP2-B)."""

import pickle
import zlib

import numpy as np
import pytest
import torch

from tests.fixtures import decode_frame, make_fake_shard
from wm.data.dataset import DataConfig, WindowDataset, make_dataloader
from wm.data.reader import ShardReader
from wm.data.schema import NULL_ACTION

H, W = 6, 5  # non-square, to catch H/W swaps
LENGTHS = [5, 1, 40, 3, 4, 300, 12, 2, 60, 9]  # rows; includes episodes shorter than K


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds")
    make_fake_shard(d, len(LENGTHS), LENGTHS, H, W)
    return d


@pytest.fixture(scope="module")
def two_dirs(tmp_path_factory):
    a, b = tmp_path_factory.mktemp("a"), tmp_path_factory.mktemp("b")
    make_fake_shard(a, 3, [10, 20, 30], H, W)
    make_fake_shard(b, 2, [50, 7], H, W)
    return a, b


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """Undo the RGB normalization: (..., 3, H, W) in [-1,1] -> (..., H, W, 3) uint8."""
    return np.rint((x.numpy() + 1) * 127.5).astype(np.uint8).swapaxes(-3, -2).swapaxes(-2, -1)


def both_splits(dirs, K, **kw):
    return [WindowDataset(dirs, K, s, val_fraction=0.3, **kw) for s in ("train", "val")]


def all_items(ds):
    return [ds[j] for j in range(len(ds))]


# ---- alignment -------------------------------------------------------------------------

@pytest.mark.parametrize("K", [1, 4, 8])
def test_exact_alignment(shard_dir, K):
    """Context rows are i-K..i-1 (oldest first), actions[i-K..i-1], target row i, same episode."""
    reader = ShardReader(next(shard_dir.glob("*.h5")))
    actions = reader.read("actions", slice(None))
    episode_idx = reader.read("episode_idx", slice(None))
    for ds in both_splits([shard_dir], K):
        for item in all_items(ds):
            i = int(item["row"])
            target_row, target_ep = decode_frame(to_uint8(item["target"]))
            assert target_row == i and target_ep == episode_idx[i]
            ctx = [decode_frame(f) for f in to_uint8(item["context"])]
            assert [r for r, _ in ctx] == list(range(i - K, i))
            assert all(e == target_ep for _, e in ctx)
            assert item["actions"].tolist() == actions[i - K:i].tolist()


def test_actions_never_minus_one_or_null(shard_dir):
    """No window contains the last-row -1 action or the CFG NULL_ACTION (CP5-C relies on this)."""
    for K in (1, 4, 8):
        for ds in both_splits([shard_dir], K):
            acts = torch.stack([it["actions"] for it in all_items(ds)])
            assert acts.min() >= 0 and acts.max() < NULL_ACTION


# ---- window counts and episode boundaries ---------------------------------------------

@pytest.mark.parametrize("K", [1, 4, 8])
def test_window_count_and_boundaries(shard_dir, K):
    """Train + val cover exactly sum(max(0, length - K)) targets, none crossing an episode start."""
    reader = ShardReader(next(shard_dir.glob("*.h5")))
    starts, lengths = reader.episodes["start"], reader.episodes["length"]
    expected = {int(s) + j for s, n in zip(starts, lengths) for j in range(K, int(n))}
    train, val = both_splits([shard_dir], K)
    rows = [int(it["row"]) for it in all_items(train) + all_items(val)]
    assert len(train) + len(val) == sum(max(0, int(n) - K) for n in lengths)
    assert sorted(rows) == sorted(expected)


def test_k_longer_than_all_episodes(tmp_path):
    make_fake_shard(tmp_path, 2, [3, 4], H, W)
    assert len(WindowDataset([tmp_path], 4, "train", val_fraction=0.0)) == 0


# ---- split ------------------------------------------------------------------------------

def test_split_disjoint_and_hash_rule(shard_dir):
    """Train and val share no episode, and val membership follows the crc32 rule from 4.5."""
    path = next(shard_dir.glob("*.h5"))
    eps = ShardReader(path).episodes
    episode_idx = ShardReader(path).read("episode_idx", slice(None))
    train, val = both_splits([shard_dir], 1)
    train_eps = {int(episode_idx[int(it["row"])]) for it in all_items(train)}
    val_eps = {int(episode_idx[int(it["row"])]) for it in all_items(val)}
    assert train_eps and val_eps and not train_eps & val_eps
    for e, uid in enumerate(eps["episode_uid"]):
        is_val = zlib.crc32(f"{path.name}:{int(uid)}".encode()) % 10_000 < 0.3 * 10_000
        if eps["length"][e] > 1:
            assert (e in val_eps) == is_val


def test_split_deterministic(shard_dir):
    a = WindowDataset([shard_dir], 4, "val", val_fraction=0.3)
    b = WindowDataset([shard_dir], 4, "val", val_fraction=0.3, seed=123)
    assert [int(a[j]["row"]) for j in range(len(a))] == [int(b[j]["row"]) for j in range(len(b))]


def test_val_fraction_extremes(shard_dir):
    total = sum(max(0, n - 4) for n in LENGTHS)
    assert len(WindowDataset([shard_dir], 4, "train", val_fraction=0.0)) == total
    assert len(WindowDataset([shard_dir], 4, "val", val_fraction=0.0)) == 0
    assert len(WindowDataset([shard_dir], 4, "val", val_fraction=1.0)) == total


def test_bad_split_raises(shard_dir):
    with pytest.raises(ValueError):
        WindowDataset([shard_dir], 4, "test")


# ---- multiple dataset dirs --------------------------------------------------------------

def test_multiple_dirs_shard_field(two_dirs):
    """`shard` indexes the sorted global shard list; rows decode within that shard."""
    ds = WindowDataset(list(two_dirs), 4, "train", val_fraction=0.0)
    assert len(ds) == (6 + 16 + 26) + (46 + 3)
    paths = ds.shard_paths
    assert paths == sorted(paths) and len(paths) == 2
    seen = set()
    for item in all_items(ds):
        s, i = int(item["shard"]), int(item["row"])
        seen.add(s)
        ep = int(ShardReader(paths[s]).read("episode_idx", slice(i, i + 1))[0])
        assert decode_frame(to_uint8(item["target"])) == (i, ep)
    assert seen == {0, 1}


def test_mismatched_frame_shapes_raise(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_fake_shard(a, 1, [10], H, W)
    make_fake_shard(b, 1, [10], H + 1, W)
    with pytest.raises(ValueError):
        WindowDataset([a, b], 4, "train")


def test_no_shards_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        WindowDataset([tmp_path], 4, "train")


# ---- output format ----------------------------------------------------------------------

@pytest.mark.parametrize("grayscale", [False, True])
def test_dtypes_shapes_range(tmp_path, grayscale):
    make_fake_shard(tmp_path, 3, [30, 20, 25], H, W, encode_rows=False)  # random full-range pixels
    ds = WindowDataset([tmp_path], 4, "train", grayscale=grayscale, val_fraction=0.0)
    C = 1 if grayscale else 3
    lo, hi = 1.0, -1.0
    for item in all_items(ds):
        assert item["context"].shape == (4, C, H, W) and item["context"].dtype == torch.float32
        assert item["target"].shape == (C, H, W) and item["target"].dtype == torch.float32
        assert item["actions"].shape == (4,) and item["actions"].dtype == torch.int64
        assert item["row"].dtype == torch.int64 and item["row"].shape == ()
        assert item["shard"].dtype == torch.int64 and item["shard"].shape == ()
        x = torch.cat([item["context"].flatten(), item["target"].flatten()])
        lo, hi = min(lo, x.min().item()), max(hi, x.max().item())
    assert -1.0 <= lo < -0.9 and 0.9 < hi <= 1.0


def test_rgb_normalization_exact(shard_dir):
    ds = WindowDataset([shard_dir], 1, "train", val_fraction=0.0)
    item = ds[0]
    raw = ShardReader(ds.shard_paths[0]).read("frames", slice(int(item["row"]), int(item["row"]) + 1))[0]
    expected = torch.from_numpy(raw).permute(2, 0, 1).float() / 127.5 - 1
    assert torch.equal(item["target"], expected)


def test_grayscale_luma(shard_dir):
    """C=1 is 0.299R + 0.587G + 0.114B on the uint8 values, then /127.5 - 1 (no rounding)."""
    ds = WindowDataset([shard_dir], 2, "train", grayscale=True, val_fraction=0.0)
    for j in (0, len(ds) // 2, len(ds) - 1):
        item = ds[j]
        i = int(item["row"])
        raw = ShardReader(ds.shard_paths[0]).read("frames", slice(i - 2, i + 1)).astype(np.float32)
        luma = 0.299 * raw[..., 0] + 0.587 * raw[..., 1] + 0.114 * raw[..., 2]
        expected = torch.from_numpy(luma / 127.5 - 1)[:, None]
        assert torch.allclose(item["context"], expected[:2], atol=1e-6)
        assert torch.allclose(item["target"], expected[2], atol=1e-6)


# ---- in-memory, pickling, loaders -------------------------------------------------------

@pytest.mark.parametrize("grayscale", [False, True])
def test_in_memory_matches_lazy(shard_dir, grayscale):
    for split in ("train", "val"):
        lazy = WindowDataset([shard_dir], 4, split, grayscale=grayscale, val_fraction=0.3)
        mem = WindowDataset([shard_dir], 4, split, grayscale=grayscale, val_fraction=0.3, in_memory=True)
        assert len(lazy) == len(mem)
        for j in range(len(lazy)):
            a, b = lazy[j], mem[j]
            assert a.keys() == b.keys()
            assert all(torch.equal(a[k], b[k]) for k in a)


def test_no_open_handle_after_init(shard_dir):
    """Building the index must not leave an HDF5 handle open in the parent (pitfall 5)."""
    ds = WindowDataset([shard_dir], 4, "train")
    assert all(r._handle is None for r in ds._readers)
    ds[0]
    clone = pickle.loads(pickle.dumps(ds))
    assert all(r._handle is None for r in clone._readers)
    assert torch.equal(clone[0]["target"], ds[0]["target"])


def make_cfg(dirs, **kw) -> DataConfig:
    base = dict(dataset_dirs=[str(d) for d in dirs], K=4, batch_size=8, num_workers=0,
                val_fraction=0.3, seed=0)
    return DataConfig(**{**base, **kw})


def test_train_loader_shuffles_and_drops_last(shard_dir):
    cfg = make_cfg([shard_dir])
    loader = make_dataloader(cfg, "train")
    n = len(loader.dataset)
    batches = list(loader)
    assert len(batches) == n // cfg.batch_size
    assert all(b["context"].shape == (8, 4, 3, H, W) for b in batches)
    rows = torch.cat([b["row"] for b in batches]).tolist()
    assert rows != sorted(rows)
    # Same seed -> same order.
    again = torch.cat([b["row"] for b in make_dataloader(cfg, "train")]).tolist()
    assert again == rows


def test_val_loader_in_order_keeps_last(shard_dir):
    cfg = make_cfg([shard_dir])
    loader = make_dataloader(cfg, "val")
    rows = torch.cat([b["row"] for b in loader]).tolist()
    assert rows == [int(loader.dataset[j]["row"]) for j in range(len(loader.dataset))]


def test_loader_num_workers_2(shard_dir):
    """Two spawn workers (macOS default) give the same set of samples as the main process."""
    cfg = make_cfg([shard_dir], num_workers=2)
    loader = make_dataloader(cfg, "val")
    assert loader.persistent_workers
    batches = list(loader)
    rows = torch.cat([b["row"] for b in batches]).tolist()
    assert rows == [int(loader.dataset[j]["row"]) for j in range(len(loader.dataset))]
    ref = make_dataloader(make_cfg([shard_dir]), "val")
    for a, b in zip(batches, ref):
        assert torch.equal(a["context"], b["context"]) and torch.equal(a["actions"], b["actions"])


def test_loader_num_workers_2_in_memory(shard_dir):
    cfg = make_cfg([shard_dir], num_workers=2, in_memory=True)
    rows = torch.cat([b["row"] for b in make_dataloader(cfg, "train")]).tolist()
    assert len(rows) == len(set(rows)) > 0
