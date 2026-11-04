"""Минимальный загрузчик YAML-конфигов с переопределением из командной строки.

cfg = load_config("configs/reach.yaml", overrides=["env.max_steps=200", "seed=3"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _set_nested(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_nested(cfg, key.strip(), val.strip())
    return cfg
