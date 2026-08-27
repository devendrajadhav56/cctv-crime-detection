"""Score overlapping 4-second windows of a video as fight vs normal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cctv_crime.config import load_infer_config  # noqa: E402
from cctv_crime.infer import format_row, infer_video, results_to_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a .mp4 file")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to infer.yaml (default: configs/infer.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print window timestamps without loading X-CLIP",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write the same rows as CSV",
    )
    args = parser.parse_args()

    config = load_infer_config(args.config)
    results = infer_video(args.video, config, dry_run=args.dry_run)
    if not results:
        print("No windows: video is shorter than the clip length.", file=sys.stderr)
        sys.exit(1)

    for row in results:
        print(format_row(row))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        results_to_frame(results).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
