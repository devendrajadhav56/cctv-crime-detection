"""Write fight-vs-normal video and clip manifests from configs/data.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cctv_crime.config import load_config  # noqa: E402
from cctv_crime.prepare import prepare_dataset, print_summary, write_manifests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to data.yaml (default: configs/data.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    videos, clips = prepare_dataset(config)
    videos_path, clips_path = write_manifests(videos, clips, config.manifests_dir)
    print_summary(videos, clips)
    print()
    print(f"Wrote {videos_path}")
    print(f"Wrote {clips_path}")


if __name__ == "__main__":
    main()
