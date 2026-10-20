"""YAML config loading with `inherit:` composition and dotted CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`; override wins on conflicts."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML config; parents listed in `inherit` (relative to this file) apply first."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Cyclic config inheritance at {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}

    merged: dict[str, Any] = {}
    for parent in cfg.pop("inherit", []) or []:
        merged = deep_merge(merged, load_config(path.parent / parent, _seen | {path}))
    return deep_merge(merged, cfg)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides; values are parsed as YAML scalars (int, float, bool, null)."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = out
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise TypeError(f"Cannot override {dotted!r}: {key!r} is not a mapping")
        node[leaf] = yaml.safe_load(raw)
    return out


def flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config to dotted keys (for MLflow params)."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
