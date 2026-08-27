# CCTV crime detection — Phase 1

Fight vs normal dataset indexing from a UCF-Crime subset. No training, no live CCTV, no alerts yet.

## What this phase does

Indexes the Fighting and test-normal videos already on disk and writes:

- `data/manifests/videos.csv` — one row per video (`path`, `label`, duration, fps, frames, train/val split)
- `data/manifests/clips.csv` — 4-second windows with a 2-second stride (`start_sec`, `end_sec`). Clips shorter than 4 seconds at the end of a video are dropped.

Videos stay in their original folders. The repo stores paths and timestamps only.

Fight clips inherit the **video-level** label. UCF-Crime Fighting files are weakly labeled: a “fight” video can contain long stretches of normal activity. That is expected in Phase 1.

## Local data (Windows)

| Label | Folder | Count |
| --- | --- | --- |
| fight | `C:\Users\devendra.jadhav\Downloads\Anomaly-Videos-Part-2\Anomaly-Videos-Part-2\Fighting` | 50 |
| normal | `C:\Users\devendra.jadhav\Downloads\Testing_Normal_Videos\Testing_Normal_Videos_Anomaly` | 150 |

Burglary, Explosion, and Part-1 classes are ignored.

**Gap:** UCF-Crime training normals (`Training-Normal-Videos-*`) are not on disk. Phase 1 uses the **test** normal set as the Normal class. That is enough to organize the binary dataset; later training will be weaker until those archives are added.

Edit `configs/data.yaml` if the folders move.

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
```

Expected: 50 fight + 150 normal videos, an 80/20 **video-level** split (same video never appears in both train and val), and a non-empty `clips.csv`.

## GB10 (later phases)

The GB10 box is aarch64 with CUDA 13.0. Do not copy a Windows/x86 PyTorch wheel onto it. Install PyTorch from the official CUDA index for that platform, then point `configs/data.yaml` at the copied Fighting / Testing_Normal folders. Phase 1 itself is meant to run locally against the Downloads paths.

## Layout

```text
configs/data.yaml
src/cctv_crime/          # config, probe, prepare
scripts/prepare_dataset.py
scripts/check_env.py
data/manifests/          # generated CSVs (gitignored)
```
