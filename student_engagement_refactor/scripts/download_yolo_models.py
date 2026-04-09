"""
download_yolo_models.py
=======================
Downloads YOLOv8s, YOLOv9s, and YOLO11s weights via the Ultralytics API.
Ultralytics caches models in ~/.cache/ultralytics/ and also accepts bare
model names like 'yolov9s.pt' at runtime — this script just pre-fetches them
so your first benchmark run is not delayed by download.

Usage
-----
  python scripts/download_yolo_models.py
  python scripts/download_yolo_models.py --models yolov9s yolo11s
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS = {
    "yolov8s": "yolov8s.pt",
    "yolov9s": "yolov9s.pt",
    "yolo11s": "yolo11s.pt",
}


def download(model_keys: list):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run:  pip install ultralytics")
        sys.exit(1)

    for key in model_keys:
        weights = MODELS[key]
        print(f"  Downloading / verifying {key} ({weights}) …", end="", flush=True)
        try:
            YOLO(weights)          # triggers download to ~/.cache/ultralytics
            print(" OK")
        except Exception as e:
            print(f" FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pre-download YOLO model weights.")
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Models to download. Default: all three."
    )
    args = parser.parse_args()
    print(f"\nDownloading {len(args.models)} model(s) …\n")
    download(args.models)
    print("\nDone. Run the benchmark with:\n")
    print("  python scripts/compare_yolo_models.py --source 0 --duration 30\n")


if __name__ == "__main__":
    main()
