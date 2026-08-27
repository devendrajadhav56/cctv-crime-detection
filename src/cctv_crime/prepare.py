"""Scan source folders, split at video level, and write fight/normal manifests."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cctv_crime.config import DataConfig
from cctv_crime.probe import probe_video
from cctv_crime.windows import clip_windows

VIDEO_EXTS = {".mp4"}


def scan_videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Source directory not found: {directory}")
    videos = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        raise FileNotFoundError(f"No .mp4 files in {directory}")
    return videos


def assign_splits(
    labels: list[str],
    val_ratio: float,
    seed: int,
) -> list[str]:
    """Stratified video-level train/val split. Clips inherit this later."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)

    splits = ["train"] * len(labels)
    rng = random.Random(seed)
    for indices in by_label.values():
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * val_ratio)) if len(indices) > 1 else 0
        for index in indices[:n_val]:
            splits[index] = "val"
    return splits


def prepare_dataset(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    to_probe: list[tuple[str, Path]] = []
    for label, directory in config.sources.items():
        for path in scan_videos(directory):
            to_probe.append((label, path))

    video_rows: list[dict[str, object]] = []
    for label, path in tqdm(to_probe, desc="Probing videos"):
        info = probe_video(path)
        video_rows.append(
            {
                "video_id": path.stem,
                "path": str(path.resolve()),
                "filename": path.name,
                "label": label,
                "duration_sec": round(info.duration_sec, 4),
                "fps": round(info.fps, 4),
                "n_frames": info.n_frames,
            }
        )

    labels = [str(row["label"]) for row in video_rows]
    splits = assign_splits(labels, config.split.val_ratio, config.split.seed)
    for row, split in zip(video_rows, splits, strict=True):
        row["split"] = split

    videos = pd.DataFrame(video_rows).sort_values(["label", "video_id"]).reset_index(drop=True)

    clip_rows: list[dict[str, object]] = []
    clip_id = 0
    for row in videos.itertuples(index=False):
        for start_sec, end_sec in clip_windows(
            duration_sec=float(row.duration_sec),
            length_sec=config.clip.length_sec,
            stride_sec=config.clip.stride_sec,
        ):
            clip_rows.append(
                {
                    "clip_id": f"clip_{clip_id:06d}",
                    "video_id": row.video_id,
                    "path": row.path,
                    "start_sec": round(start_sec, 4),
                    "end_sec": round(end_sec, 4),
                    "label": row.label,
                    "split": row.split,
                }
            )
            clip_id += 1

    clips = pd.DataFrame(clip_rows)
    return videos, clips


def write_manifests(videos: pd.DataFrame, clips: pd.DataFrame, manifests_dir: Path) -> tuple[Path, Path]:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    videos_path = manifests_dir / "videos.csv"
    clips_path = manifests_dir / "clips.csv"
    videos.to_csv(videos_path, index=False)
    clips.to_csv(clips_path, index=False)
    return videos_path, clips_path


def print_summary(videos: pd.DataFrame, clips: pd.DataFrame) -> None:
    hours = videos["duration_sec"].sum() / 3600.0
    print("Video counts by label:")
    print(videos.groupby("label").size().to_string())
    print()
    print("Video counts by label x split:")
    print(videos.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())
    print()
    print(f"Total videos: {len(videos)}")
    print(f"Total duration: {hours:.2f} hours")
    print(f"Total clips ({len(clips)}):")
    if clips.empty:
        print("  (none — videos shorter than clip length?)")
        return
    print(clips.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())
