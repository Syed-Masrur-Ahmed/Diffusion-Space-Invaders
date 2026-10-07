from dataclasses import dataclass, field

import pytest

from wm.config import from_dict, load_config


def _write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text)
    return path


def test_nested_numeric_override(tmp_path):
    path = _write(tmp_path, "name: run\ntrain:\n  lr: 1.0e-4\n  steps: 10\n")
    cfg = load_config(path, ["train.lr=2e-4", "train.steps=20", "name=other"])
    assert cfg == {"name": "other", "train": {"lr": 2e-4, "steps": 20}}


def test_override_parses_bools_and_lists(tmp_path):
    path = _write(tmp_path, "a: {flag: false, ch: [1, 2]}\n")
    cfg = load_config(path, ["a.flag=true", "a.ch=[64,128,256]"])
    assert cfg["a"] == {"flag": True, "ch": [64, 128, 256]}


def test_unknown_override_key_raises(tmp_path):
    path = _write(tmp_path, "train: {lr: 1.0}\n")
    with pytest.raises(KeyError, match="train.lrr"):
        load_config(path, ["train.lrr=2"])


def test_malformed_override_raises(tmp_path):
    path = _write(tmp_path, "a: 1\n")
    with pytest.raises(ValueError, match="key=value"):
        load_config(path, ["a"])


@dataclass
class Train:
    lr: float = 1e-4
    steps: int = 10


@dataclass
class Root:
    name: str
    train: Train = field(default_factory=Train)


def test_from_dict_builds_nested_dataclasses():
    cfg = from_dict(Root, {"name": "x", "train": {"steps": 5}})
    assert cfg == Root(name="x", train=Train(lr=1e-4, steps=5))


def test_from_dict_unknown_key_raises():
    with pytest.raises(KeyError, match="train.bogus"):
        from_dict(Root, {"name": "x", "train": {"bogus": 1}})
