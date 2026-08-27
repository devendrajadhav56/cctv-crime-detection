"""Read duration / fps / frame count from a video without decoding every frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    fps: float
    n_frames: int
    duration_sec: float


def probe_video(path: Path) -> VideoProbe:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or n_frames <= 0:
            raise RuntimeError(
                f"Could not read fps/frame count from {path} "
                f"(fps={fps}, n_frames={n_frames})"
            )
        duration_sec = n_frames / fps
        return VideoProbe(
            path=path,
            fps=fps,
            n_frames=n_frames,
            duration_sec=duration_sec,
        )
    finally:
        capture.release()
