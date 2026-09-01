"""Slide windows over a video and optionally score them with the configured backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from cctv_crime.config import InferConfig
from cctv_crime.probe import probe_video
from cctv_crime.windows import clip_windows

# X-CLIP's binary label kept its established display word; every other label
# (VadCLIP's specific classes, "normal") just uppercases.
DISPLAY_LABELS = {"fight": "CRIME"}


@dataclass(frozen=True)
class WindowResult:
    start_sec: float
    end_sec: float
    label: str | None
    confidence: float | None
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def display_label(self) -> str | None:
        if self.label is None:
            return None
        return DISPLAY_LABELS.get(self.label, self.label.upper())

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sec": round(self.start_sec, 4),
            "end_sec": round(self.end_sec, 4),
            "label": self.label,
            "display_label": self.display_label,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "probabilities": {key: round(value, 4) for key, value in self.probabilities.items()},
        }


def format_timestamp(seconds: float) -> str:
    total = int(max(seconds, 0.0))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_row(row: WindowResult) -> str:
    span = f"{format_timestamp(row.start_sec)}-{format_timestamp(row.end_sec)}"
    if row.label is None or row.confidence is None:
        return f"{span}  (dry-run)"
    shown = row.display_label or row.label
    return f"{span}  {shown:<6}  {row.confidence:.2f}"


def infer_video(
    video_path: Path,
    config: InferConfig,
    *,
    dry_run: bool = False,
    classifier: Any | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
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

    if classifier is None:
        from cctv_crime.model import build_classifier

        classifier = build_classifier(config)

    from cctv_crime.frames import read_window_frames

    results: list[WindowResult] = []
    total = len(windows)
    for index, (start, end) in enumerate(tqdm(windows, desc="Scoring clips")):
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
                probabilities=dict(prediction.probabilities),
            )
        )
        if progress_callback is not None:
            progress_callback(index + 1, total)
    return results


def results_to_frame(results: list[WindowResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start_sec": row.start_sec,
                "end_sec": row.end_sec,
                "label": row.label or "",
                "display_label": row.display_label or "",
                "confidence": row.confidence if row.confidence is not None else "",
            }
            for row in results
        ]
    )
