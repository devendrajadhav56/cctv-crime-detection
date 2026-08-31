# CCTV crime detection

Fight vs normal from a UCF-Crime subset. Phase 1 indexes videos. Phase 2 scores overlapping clips with a pretrained X-CLIP model (zero-shot, not fine-tuned). The web app sends a clip to the GB10 GPU box and returns a video with **CRIME** / **NORMAL** burned in.

## Phase 1 — Dataset index

Indexes the Fighting and test-normal videos already on disk and writes:

- `data/manifests/videos.csv` — one row per video (`path`, `label`, duration, fps, frames, train/val split)
- `data/manifests/clips.csv` — 4-second windows with a 2-second stride (`start_sec`, `end_sec`). Clips shorter than 4 seconds at the end of a video are dropped.

Videos stay in their original folders. The repo stores paths and timestamps only.

Fight clips inherit the **video-level** label. UCF-Crime Fighting files are weakly labeled: a “fight” video can contain long stretches of normal activity. That is expected in Phase 1.

### Local data (Windows)

| Label | Folder | Count |
| --- | --- | --- |
| fight | `C:\Users\devendra.jadhav\Downloads\Anomaly-Videos-Part-2\Anomaly-Videos-Part-2\Fighting` | 50 |
| normal | `C:\Users\devendra.jadhav\Downloads\Testing_Normal_Videos\Testing_Normal_Videos_Anomaly` | 150 |

Burglary, Explosion, and Part-1 classes are ignored.

**Gap:** UCF-Crime training normals (`Training-Normal-Videos-*`) are not on disk. Phase 1 uses the **test** normal set as the Normal class. That is enough to organize the binary dataset; later training will be weaker until those archives are added.

Edit `configs/data.yaml` if the folders move.

## Phase 2 — Offline classifier

```bash
python scripts/detect_event.py path/to/video.mp4
python scripts/detect_event.py path/to/video.mp4 --dry-run
python scripts/detect_event.py path/to/video.mp4 --csv out.csv
```

Each 4-second window (2-second stride, same as Phase 1) is scored as `fight` or `normal`:

```text
00:00-00:04  normal  0.97
00:02-00:06  normal  0.94
00:04-00:08  fight   0.73
```

The scorer is HuggingFace **X-CLIP** (`microsoft/xclip-base-patch16-16-frames`): 16 frames per clip, scored against the prompt lists in `configs/infer.yaml` (logits averaged per class). This is zero-shot, not trained on UCF-Crime. Accuracy on CCTV fights will be limited until Phase 3 fine-tunes.

`--dry-run` prints window timestamps without loading the model (use this on local Windows).

## Setup (local Windows)

Python 3.10–3.12. CPU PyTorch is enough here. On this machine use 3.12 (`py -3.12`); 3.14 may not have PyTorch wheels yet.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python scripts/check_env.py
python scripts/prepare_dataset.py
python scripts/detect_event.py "C:\Users\devendra.jadhav\Downloads\Anomaly-Videos-Part-2\Anomaly-Videos-Part-2\Fighting\Fighting002_x264.mp4" --dry-run
```

Expected: 50 fight + 150 normal videos, an 80/20 **video-level** split, a non-empty `clips.csv`, and ~43 dry-run windows on Fighting002 (~89.6s).

## GB10 (CUDA inference)

The GB10 box is aarch64 with CUDA 13.0. Do not copy a Windows/x86 PyTorch wheel onto it.

1. Copy this repo and at least one `.mp4` (for example `Fighting002_x264.mp4`) to the box.
2. Install PyTorch from the official CUDA index for that platform, then `pip install -r requirements.txt`.
3. First run downloads X-CLIP weights (needs network, or pre-seed `HF_HOME`).
4. Run:

```bash
python scripts/detect_event.py /path/to/Fighting002_x264.mp4
```

Uses CUDA if `torch.cuda.is_available()`, otherwise CPU. Point `configs/data.yaml` at copied Fighting / Testing_Normal folders when you prepare manifests on the box.

## App — labelled video UI

The GB10 box runs the model and a small HTTP API. Open the UI from your local browser and upload a clip; it comes back with **CRIME** / **NORMAL** burned into every frame.

On GB10:

```bash
pip install -r requirements.txt
python scripts/serve.py --host 0.0.0.0 --port 8000
```

From this Windows machine, open `http://<gb10-ip>:8000`. Leave **GB10 API** empty (same origin). If you open `web/index.html` as a local file instead, paste `http://<gb10-ip>:8000` into that field.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/health` | CUDA / model status |
| POST | `/v1/jobs` | `multipart/form-data` field `video` |
| GET | `/v1/jobs/{id}` | status, progress, window JSON |
| GET | `/v1/jobs/{id}/video` | labelled H.264 MP4 |

Fight windows are shown as **CRIME**. Local UI testing without a GPU:

```bash
python scripts/serve.py --dry-run
```

`--dry-run` skips X-CLIP and paints alternating CRIME/NORMAL bands so overlay and upload can be checked.

## Layout

```text
configs/data.yaml
configs/infer.yaml
src/cctv_crime/          # config, probe, prepare, windows, frames, model, infer, overlay, server
scripts/prepare_dataset.py
scripts/detect_event.py
scripts/serve.py
scripts/check_env.py
web/                     # UI (upload + labelled player)
data/manifests/          # generated CSVs (gitignored)
```
