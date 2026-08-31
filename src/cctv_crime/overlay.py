"""Burn CRIME / NORMAL labels onto a video from sliding-window scores."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from cctv_crime.infer import WindowResult, format_timestamp
from cctv_crime.probe import probe_video

CRIME_BGR = (70, 55, 255)
NORMAL_BGR = (170, 210, 90)
HUD_BG = (18, 16, 14)


def fight_probability(row: WindowResult) -> float:
    if row.probabilities and "fight" in row.probabilities:
        return float(row.probabilities["fight"])
    if row.label == "fight":
        return float(row.confidence or 0.0)
    if row.label == "normal":
        return 1.0 - float(row.confidence or 0.0)
    return 0.0


def label_at_time(time_sec: float, results: list[WindowResult]) -> tuple[str, float]:
    covering = [row for row in results if row.start_sec <= time_sec < row.end_sec]
    if not covering:
        return "NORMAL", 0.0
    crime_score = max(fight_probability(row) for row in covering)
    if crime_score >= 0.5:
        return "CRIME", crime_score
    return "NORMAL", 1.0 - crime_score


def _scale(width: int, height: int) -> float:
    return max(min(width, height) / 360.0, 0.55)


def draw_hud(frame: np.ndarray, display_label: str, confidence: float, time_sec: float) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    scale = _scale(width, height)
    is_crime = display_label == "CRIME"
    accent = CRIME_BGR if is_crime else NORMAL_BGR

    thickness = max(int(round(4 * scale)), 2)
    cv2.rectangle(out, (0, 0), (width - 1, height - 1), accent, thickness)

    bar_h = max(int(round(36 * scale)), 22)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42 * scale * 1.35
    text_thick = max(int(round(scale)), 1)
    pad = max(int(round(8 * scale)), 5)
    label_text = f"{display_label}  {confidence * 100:4.0f}%"
    time_text = format_timestamp(time_sec)
    cv2.putText(out, label_text, (pad, int(bar_h * 0.72)), font, font_scale, accent, text_thick, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(time_text, font, font_scale, text_thick)
    cv2.putText(
        out,
        time_text,
        (width - tw - pad, int(bar_h * 0.72)),
        font,
        font_scale,
        (220, 220, 220),
        text_thick,
        cv2.LINE_AA,
    )
    return out


def _encode_h264(source: Path, dest: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dest),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    return completed.returncode == 0 and dest.is_file()


def render_labelled_video(
    source: Path,
    dest: Path,
    results: list[WindowResult],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    info = probe_video(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    raw_path = dest.with_name(dest.stem + "_raw.mp4")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, info.fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write video: {raw_path}")

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            time_sec = frame_index / info.fps
            display_label, confidence = label_at_time(time_sec, results)
            writer.write(draw_hud(frame, display_label, confidence, time_sec))
            frame_index += 1
            if progress_callback is not None and info.n_frames:
                progress_callback(frame_index, info.n_frames)
    finally:
        writer.release()
        capture.release()

    if _encode_h264(raw_path, dest):
        raw_path.unlink(missing_ok=True)
        return dest

    raw_path.replace(dest)
    return dest
