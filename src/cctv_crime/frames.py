"""Sample a fixed number of RGB frames from a time window."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def window_frame_indices(
    start_sec: float,
    end_sec: float,
    fps: float,
    n_video_frames: int,
    n_samples: int,
) -> list[int]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if fps <= 0 or n_video_frames <= 0:
        raise ValueError("fps and n_video_frames must be positive")
    duration = end_sec - start_sec
    timestamps = [start_sec + duration * i / n_samples for i in range(n_samples)]
    last = n_video_frames - 1
    return [min(max(int(timestamp * fps), 0), last) for timestamp in timestamps]


def read_rgb_frames(path: Path, indices: list[int]) -> list[Image.Image]:
    """Seek to each index and return RGB PIL frames (BGR converted)."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frames: list[Image.Image] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Could not read frame {index} from {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(np.ascontiguousarray(rgb)))
        return frames
    finally:
        capture.release()


def read_window_frames(
    path: Path,
    start_sec: float,
    end_sec: float,
    fps: float,
    n_video_frames: int,
    n_samples: int,
) -> list[Image.Image]:
    indices = window_frame_indices(start_sec, end_sec, fps, n_video_frames, n_samples)
    return read_rgb_frames(path, indices)
