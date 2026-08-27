"""Sliding time windows over a video."""

from __future__ import annotations


def clip_windows(duration_sec: float, length_sec: float, stride_sec: float) -> list[tuple[float, float]]:
    """Non-overlapping-tail sliding windows. Drop the last window if it is shorter than length."""
    if length_sec <= 0 or stride_sec <= 0:
        raise ValueError("clip length and stride must be positive")
    windows: list[tuple[float, float]] = []
    start = 0.0
    # Allow a tiny float tolerance so duration == length still yields one clip.
    while start + length_sec <= duration_sec + 1e-9:
        windows.append((start, start + length_sec))
        start += stride_sec
    return windows
