"""Serve the inference API and web UI (run this on the GB10 box)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cctv_crime.server import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (0.0.0.0 so other machines can connect)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip X-CLIP and overlay demo CRIME/NORMAL bands (local UI testing)",
    )
    args = parser.parse_args()
    uvicorn.run(create_app(dry_run=args.dry_run), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
