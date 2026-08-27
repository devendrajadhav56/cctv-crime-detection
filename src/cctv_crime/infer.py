"""Slide windows over a video and optionally score them with X-CLIP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cctv_crime.config import InferConfig
from cctv_crime.probe import probe_video
from cctv_crime.windows import clip_windows


@dataclass(frozen=True)
class WindowResult:
    start_sec: float
    end_sec: float
    label: str | None
    confidence: float | None


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_row(row: WindowResult) -> str:
    span = f"{format_timestamp(row.start_sec)}-{format_timestamp(row.end_sec)}"
    if row.label is None or row.confidence is None:
        return f"{span}  (dry-run)"
    return f"{span}  {row.label:<6}  {row.confidence:.2f}"


def infer_video(
    video_path: Path,
    config: InferConfig,
    *,
    dry_run: bool = False,
) -> list[WindowResult]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = probe_video(video_path)
    windows = clip_windows(info.duration_sec, config.clip.length_sec, config.clip.stride_sec)
    if not windows:
        return []

    if dry_run:
        return [
            WindowResult(start_sec=start, end_sec=end, label=None, confidence=None)
            for start, end in windows
        ]

    from cctv_crime.frames import read_window_frames
    from cctv_crime.model import ZeroShotClipClassifier

    classifier = ZeroShotClipClassifier(config)
    results: list[WindowResult] = []
    for start, end in tqdm(windows, desc="Scoring clips"):
        frames = read_window_frames(
            path=video_path,
            start_sec=start,
            end_sec=end,
            fps=info.fps,
            n_video_frames=info.n_frames,
            n_samples=config.frames_per_clip,
        )
        prediction = classifier.predict(frames)
        results.append(
            WindowResult(
                start_sec=start,
                end_sec=end,
                label=prediction.label,
                confidence=prediction.confidence,
            )
        )
    return results


def results_to_frame(results: list[WindowResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start_sec": row.start_sec,
                "end_sec": row.end_sec,
                "label": row.label or "",
                "confidence": row.confidence if row.confidence is not None else "",
            }
            for row in results
        ]
    )
