"""YAML config loading with `--set key=value` overrides, and conversion to dataclasses (ROADMAP 4.12)."""

import dataclasses
import re
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads floats like 2e-4, which YAML 1.1 treats as strings."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d[\d_]*)?(?:\.[\d_]*)?[eE][-+]?\d+$"),
    list("-+0123456789."),
)


def _parse(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


def load_config(path: str | Path, overrides: list[str] = ()) -> dict[str, Any]:
    """Load a YAML file as a dict and apply dotted overrides like "train.lr=2e-4"."""
    cfg = _parse(Path(path).read_text()) or {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key=value")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"override key {key!r} not found in {path}")
        node[leaf] = _parse(value)  # so numbers, bools, and lists get their YAML types
    return cfg


def from_dict(cls: type[T], data: dict[str, Any], _prefix: str = "") -> T:
    """Build dataclass `cls` from a nested dict, raising KeyError on keys the dataclass lacks."""
    names = {f.name for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    unknown = sorted(set(data) - names)
    if unknown:
        raise KeyError(f"unknown config keys: {', '.join(_prefix + k for k in unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = from_dict(hint, value, f"{_prefix}{name}.")
        kwargs[name] = value
    return cls(**kwargs)
