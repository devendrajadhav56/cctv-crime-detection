"""Load dataset config from YAML and resolve paths against the repo root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"


@dataclass(frozen=True)
class ClipConfig:
    length_sec: float
    stride_sec: float


@dataclass(frozen=True)
class SplitConfig:
    val_ratio: float
    seed: int


@dataclass(frozen=True)
class DataConfig:
    sources: dict[str, Path]
    clip: ClipConfig
    split: SplitConfig
    manifests_dir: Path
    repo_root: Path


def _as_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def load_config(path: Path | None = None) -> DataConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    sources = {
        label: _as_path(str(source_path), REPO_ROOT)
        for label, source_path in raw["sources"].items()
    }
    clip = ClipConfig(
        length_sec=float(raw["clip"]["length_sec"]),
        stride_sec=float(raw["clip"]["stride_sec"]),
    )
    split = SplitConfig(
        val_ratio=float(raw["split"]["val_ratio"]),
        seed=int(raw["split"]["seed"]),
    )
    manifests_dir = _as_path(str(raw["output"]["manifests_dir"]), REPO_ROOT)
    return DataConfig(
        sources=sources,
        clip=clip,
        split=split,
        manifests_dir=manifests_dir,
        repo_root=REPO_ROOT,
    )
