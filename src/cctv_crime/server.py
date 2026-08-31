"""GB10 inference API: upload a video, score windows, return a labelled mp4."""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from cctv_crime.config import InferConfig, REPO_ROOT, load_infer_config
from cctv_crime.infer import WindowResult, infer_video
from cctv_crime.overlay import render_labelled_video
from cctv_crime.probe import probe_video
from cctv_crime.windows import clip_windows

WEB_DIR = REPO_ROOT / "web"
JOBS_DIR = REPO_ROOT / "data" / "jobs"
JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    job_id: str
    status: JobStatus = "queued"
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    source_path: Path | None = None
    output_path: Path | None = None
    windows: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12])
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def snapshot(self, job: Job) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "stage": job.stage,
            "progress": round(job.progress, 3),
            "error": job.error,
            "windows": job.windows,
            "video_url": f"/v1/jobs/{job.job_id}/video" if job.status == "done" else None,
        }


def demo_window_results(windows: list[tuple[float, float]]) -> list[WindowResult]:
    """Visible CRIME/NORMAL alternation so the UI can be tested without a GPU."""
    results: list[WindowResult] = []
    for start, end in windows:
        is_crime = int(start) % 16 >= 8
        fight_p = 0.86 if is_crime else 0.14
        label = "fight" if is_crime else "normal"
        results.append(
            WindowResult(
                start_sec=start,
                end_sec=end,
                label=label,
                confidence=max(fight_p, 1.0 - fight_p),
                probabilities={"fight": fight_p, "normal": 1.0 - fight_p},
            )
        )
    return results


def create_app(*, dry_run: bool = False, config: InferConfig | None = None) -> FastAPI:
    infer_config = config or load_infer_config()
    store = JobStore()
    worker_lock = threading.Lock()
    state: dict[str, Any] = {"classifier": None, "dry_run": dry_run}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            from cctv_crime.model import ZeroShotClipClassifier

            state["classifier"] = ZeroShotClipClassifier(infer_config)
        yield

    app = FastAPI(title="CCTV Crime Detection", version="0.3.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def run_job(job: Job) -> None:
        assert job.source_path is not None
        work_dir = JOBS_DIR / job.job_id
        output_path = work_dir / "labelled.mp4"
        try:
            job.status = "running"
            job.stage = "scoring"
            job.progress = 0.05
            if dry_run:
                info = probe_video(job.source_path)
                windows = clip_windows(
                    info.duration_sec,
                    infer_config.clip.length_sec,
                    infer_config.clip.stride_sec,
                )
                results = demo_window_results(windows)
            else:

                def on_score(done: int, total: int) -> None:
                    job.progress = 0.05 + 0.7 * (done / max(total, 1))

                results = infer_video(
                    job.source_path,
                    infer_config,
                    classifier=state["classifier"],
                    progress_callback=on_score,
                )
            if not results:
                raise RuntimeError("Video is shorter than the clip length.")

            job.windows = [row.to_dict() for row in results]
            job.stage = "rendering"
            job.progress = 0.78

            def on_render(done: int, total: int) -> None:
                job.progress = 0.78 + 0.2 * (done / max(total, 1))

            render_labelled_video(
                job.source_path,
                output_path,
                results,
                progress_callback=on_render,
            )
            job.output_path = output_path
            job.stage = "done"
            job.progress = 1.0
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 — surface any worker failure to the UI
            job.status = "error"
            job.stage = "error"
            job.error = str(exc)

    def start_worker(job: Job) -> None:
        def _run() -> None:
            with worker_lock:
                run_job(job)

        threading.Thread(target=_run, daemon=True).start()

    @app.get("/health")
    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        import torch

        return {
            "ok": True,
            "dry_run": dry_run,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "model": infer_config.model_name,
            "model_loaded": state["classifier"] is not None or dry_run,
        }

    @app.post("/v1/jobs")
    async def create_job(video: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
        if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise HTTPException(status_code=400, detail="Upload an mp4/avi/mov/mkv video.")
        job = store.create()
        work_dir = JOBS_DIR / job.job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        source = work_dir / f"source{suffix}"
        source.write_bytes(await video.read())
        if source.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Empty upload.")
        job.source_path = source
        start_worker(job)
        return store.snapshot(job)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            job = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job.") from exc
        return store.snapshot(job)

    @app.get("/v1/jobs/{job_id}/video")
    def get_video(job_id: str) -> FileResponse:
        try:
            job = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job.") from exc
        if job.status != "done" or job.output_path is None or not job.output_path.is_file():
            raise HTTPException(status_code=409, detail="Labelled video is not ready.")
        return FileResponse(
            job.output_path,
            media_type="video/mp4",
            filename=f"{job_id}-labelled.mp4",
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/styles.css")
    def styles() -> FileResponse:
        return FileResponse(WEB_DIR / "styles.css", media_type="text/css")

    @app.get("/app.js")
    def script() -> FileResponse:
        return FileResponse(WEB_DIR / "app.js", media_type="text/javascript")

    return app


app = create_app(dry_run=False)
