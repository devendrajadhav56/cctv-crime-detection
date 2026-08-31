"""Load dataset config from YAML and resolve paths against the repo root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"
DEFAULT_INFER_CONFIG = REPO_ROOT / "configs" / "infer.yaml"


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


@dataclass(frozen=True)
class InferConfig:
    model_name: str
    frames_per_clip: int
    prompts: dict[str, list[str]]
    labels: tuple[str, ...]
    clip: ClipConfig
    backend: str = "xclip"
    vadclip_checkpoint: Path | None = None


def _as_prompt_list(value: Any) -> list[str]:
    if isinstance(value, str):
        texts = [value]
    elif isinstance(value, list):
        texts = [str(item) for item in value]
    else:
        raise ValueError(f"prompt must be a string or list of strings, got {type(value)}")
    texts = [text.strip() for text in texts if str(text).strip()]
    if not texts:
        raise ValueError("each label needs at least one non-empty prompt")
    return texts


def load_infer_config(path: Path | None = None) -> InferConfig:
    config_path = Path(path) if path is not None else DEFAULT_INFER_CONFIG
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prompts = {str(key): _as_prompt_list(value) for key, value in raw["prompts"].items()}
    labels = tuple(prompts.keys())
    if not labels:
        raise ValueError("infer.yaml prompts must include at least one label")
    backend = str(raw.get("backend", "xclip"))
    vadclip_raw = raw.get("vadclip") or {}
    vadclip_checkpoint = (
        _as_path(str(vadclip_raw["checkpoint_path"]), REPO_ROOT) if "checkpoint_path" in vadclip_raw else None
    )
    if backend == "vadclip" and vadclip_checkpoint is None:
        raise ValueError("backend: vadclip requires vadclip.checkpoint_path in infer.yaml")
    return InferConfig(
        model_name=str(raw["model"]["name"]),
        frames_per_clip=int(raw["model"]["frames_per_clip"]),
        prompts=prompts,
        labels=labels,
        clip=ClipConfig(
            length_sec=float(raw["clip"]["length_sec"]),
            stride_sec=float(raw["clip"]["stride_sec"]),
        ),
        backend=backend,
        vadclip_checkpoint=vadclip_checkpoint,
    )
