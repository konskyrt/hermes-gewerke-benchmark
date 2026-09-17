"""Paths and YAML config loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_DIR / "config"
OUT_DIR = PACKAGE_DIR / "out"
REPO_ROOT = PACKAGE_DIR.parent

# The 17 gewerk codes and ~490 component codes are shared with the BIM pipeline.
TAXONOMY_PATH = REPO_ROOT / "backend" / "scripts" / "data" / "label_object.json"


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def schedule_root(cfg: dict[str, Any]) -> Path:
    """Folder holding the source workbooks.

    SCHEDULE_BENCHMARK_ROOT wins over the path baked into sources.yaml so the
    same config works on another machine.
    """
    override = os.environ.get("SCHEDULE_BENCHMARK_ROOT")
    root = Path(override) if override else Path(cfg["root"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"Schedule folder not found: {root}. "
            "Set SCHEDULE_BENCHMARK_ROOT or fix `root` in config/sources.yaml."
        )
    return root


def enabled_sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in cfg["sources"] if s.get("enabled", True)]


def all_sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cfg["sources"])


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
