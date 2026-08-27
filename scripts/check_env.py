"""Print Python / PyTorch / CUDA environment info."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python: {sys.version.split()[0]} ({platform.system()} {platform.machine()})")

    try:
        import cv2

        print(f"opencv: {cv2.__version__}")
    except ImportError:
        print("opencv: not installed")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"cuda_available: {cuda}")
        if cuda:
            print(f"cuda_version: {torch.version.cuda}")
            print(f"gpu_count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                print(f"gpu_{index}: {torch.cuda.get_device_name(index)}")
        else:
            print("gpu: none (CPU is expected on local Windows for Phase 1)")
    except ImportError:
        print("torch: not installed")


if __name__ == "__main__":
    main()
