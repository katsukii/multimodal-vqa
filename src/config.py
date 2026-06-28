"""Minimal config loading: read a YAML file into a nested dict accessible by attribute."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access (cfg.train.epochs)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
