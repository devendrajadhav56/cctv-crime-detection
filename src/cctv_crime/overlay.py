"""Burn the predicted class (or NORMAL) onto a video from sliding-window scores."""

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


def anomaly_score(row: WindowResult) -> float:
    """Probability this window is anomalous (any non-"normal" label), regardless
    of which backend or which specific class produced it."""
    if row.label is None or row.confidence is None:
        return 0.0
    if row.label == "normal":
        return 1.0 - float(row.confidence)
    return float(row.confidence)


def label_at_time(time_sec: float, results: list[WindowResult]) -> tuple[str, float]:
    covering = [row for row in results if row.start_sec <= time_sec < row.end_sec]
    if not covering:
        return "NORMAL", 0.0
    best = max(covering, key=anomaly_score)
    score = anomaly_score(best)
    if score >= 0.5:
        return (best.display_label or best.label or "NORMAL"), score
    return "NORMAL", 1.0 - score


def _scale(width: int, height: int) -> float:
    return max(min(width, height) / 360.0, 0.55)


def draw_hud(frame: np.ndarray, display_label: str, confidence: float, time_sec: float) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    scale = _scale(width, height)
    is_anomaly = display_label != "NORMAL"
    accent = CRIME_BGR if is_anomaly else NORMAL_BGR

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


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _ffmpeg_exe() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def render_labelled_video(
    source: Path,
    dest: Path,
    results: list[WindowResult],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Write an H.264 MP4 Chrome can play. OpenCV mp4v is not a browser codec."""
    info = probe_video(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    ok, first = capture.read()
    if not ok or first is None:
        capture.release()
        raise RuntimeError(f"Could not read frames from {source}")

    height, width = first.shape[:2]
    width, height = _even(width), _even(height)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        capture.release()
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg on the GB10 box or pip install imageio-ffmpeg "
            "so labelled videos are H.264 (browsers cannot play OpenCV mp4v)."
        )

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(info.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None

    def push(frame: np.ndarray, frame_index: int) -> None:
        time_sec = frame_index / info.fps
        display_label, confidence = label_at_time(time_sec, results)
        labelled = _resize(draw_hud(frame, display_label, confidence, time_sec), width, height)
        process.stdin.write(np.ascontiguousarray(labelled).tobytes())
        if progress_callback is not None and info.n_frames:
            progress_callback(frame_index + 1, info.n_frames)

    try:
        push(first, 0)
        frame_index = 1
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            push(frame, frame_index)
            frame_index += 1
        process.stdin.close()
        stderr = process.communicate(timeout=600)[1]
    except Exception:
        process.kill()
        capture.release()
        raise
    finally:
        capture.release()

    if process.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        detail = (stderr or b"").decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"H.264 encode failed (need libx264). ffmpeg said:\n{detail}")
    return dest
