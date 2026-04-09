"""
compare_yolo_models.py
======================
Benchmarks YOLOv8s vs YOLOv9s vs YOLO11s on a video file or your webcam.

What it measures per model
--------------------------
  - Average inference latency (ms)          → how fast it is
  - Frames Per Second (FPS)                 → real-time viability
  - Average detections per frame            → recall proxy
  - Average confidence per detection        → precision proxy
  - Person-only detection count (class 0)   → accuracy for our use-case
  - RAM usage delta (MB)                    → memory footprint

Usage
-----
  # Compare all three on webcam (30 seconds each)
  python scripts/compare_yolo_models.py --source 0 --duration 30

  # Compare on a video file
  python scripts/compare_yolo_models.py --source path/to/classroom.mp4

  # Compare on a folder of images
  python scripts/compare_yolo_models.py --source path/to/images/ --mode images

  # Compare specific models only
  python scripts/compare_yolo_models.py --source 0 --models yolov8s yolo11s

  # Save results CSV
  python scripts/compare_yolo_models.py --source 0 --output results/model_comparison.csv
"""

import sys
import time
import argparse
import os
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from src.detection.yolov_wrapper import YoloDetector

# ─────────────────────────────────────────────
# Model registry: short name → weights filename
# Ultralytics auto-downloads if not present.
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "yolov8s":  "yolov8s.pt",
    "yolov9s":  "yolov9s.pt",
    "yolo11s":  "yolo11s.pt",
    # You can add larger variants here:
    # "yolov8m":  "yolov8m.pt",
    # "yolo11m":  "yolo11m.pt",
}

PERSON_CLASS_ID = 0   # COCO class 0 = person


# ─────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────

def _ram_mb() -> float:
    """Return current process RSS in MB, or 0 if psutil not available."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 / 1024
    return 0.0


def run_benchmark_on_frames(detector: YoloDetector, frames: list, warmup: int = 5) -> dict:
    """
    Run the detector over a list of frames and collect metrics.

    Args:
        detector:  Instantiated YoloDetector.
        frames:    List of BGR numpy arrays.
        warmup:    Number of warm-up frames (excluded from statistics).

    Returns:
        dict with benchmark statistics.
    """
    latencies = []
    det_counts = []
    person_counts = []
    confidences = []

    ram_before = _ram_mb()

    for i, frame in enumerate(frames):
        dets, lat = detector.detect_timed(frame)

        if i < warmup:
            # skip warm-up frames
            continue

        latencies.append(lat)
        det_counts.append(len(dets))

        persons = [d for d in dets if d["cls"] == PERSON_CLASS_ID]
        person_counts.append(len(persons))

        if persons:
            confidences.extend([p["conf"] for p in persons])

    ram_after = _ram_mb()

    if not latencies:
        return {}

    avg_lat = float(np.mean(latencies))
    return {
        "model":                detector.model_family,
        "weights":              detector.weights_path,
        "frames_benchmarked":   len(latencies),
        "avg_latency_ms":       round(avg_lat, 2),
        "p50_latency_ms":       round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms":       round(float(np.percentile(latencies, 95)), 2),
        "avg_fps":              round(1000.0 / avg_lat, 1),
        "avg_detections":       round(float(np.mean(det_counts)), 2),
        "avg_persons":          round(float(np.mean(person_counts)), 2),
        "avg_person_conf":      round(float(np.mean(confidences)), 3) if confidences else 0.0,
        "ram_delta_mb":         round(ram_after - ram_before, 1),
    }


def collect_frames_from_source(source, duration_sec: int, max_frames: int = 2000) -> list:
    """
    Collect a list of BGR frames from a webcam index or video file.
    Caps at max_frames to avoid OOM.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_wanted = min(int(duration_sec * fps), max_frames)

    frames = []
    print(f"  Collecting {total_wanted} frames from source …", end="", flush=True)
    while len(frames) < total_wanted:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f" got {len(frames)} frames.")
    return frames


def collect_frames_from_images(folder: str) -> list:
    """Collect all image frames from a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    print(f"  Loaded {len(frames)} images from {folder}")
    return frames


# ─────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────

def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table to the terminal."""
    if not results:
        print("No results to display.")
        return

    metrics = [
        ("Model",           "model",            "s"),
        ("Avg Latency(ms)", "avg_latency_ms",   ".2f"),
        ("P95 Latency(ms)", "p95_latency_ms",   ".2f"),
        ("Avg FPS",         "avg_fps",          ".1f"),
        ("Avg Persons/frm", "avg_persons",       ".2f"),
        ("Avg Conf",        "avg_person_conf",  ".3f"),
        ("RAM Δ (MB)",      "ram_delta_mb",     ".1f"),
        ("Frames",          "frames_benchmarked","d"),
    ]

    print()
    print("=" * 85)
    print("  YOLO MODEL COMPARISON RESULTS")
    print("=" * 85)

    header = "  {:<18}".format("Metric")
    for r in results:
        header += f"  {r['model']:>16}"
    print(header)
    print("-" * 85)

    for label, key, fmt in metrics:
        row = f"  {label:<18}"
        for r in results:
            val = r.get(key, "-")
            if isinstance(val, str):
                row += f"  {val:>16}"
            else:
                row += f"  {format(val, fmt):>16}"
        print(row)

    print("=" * 85)

    # Winner banner
    if len(results) > 1:
        fastest = min(results, key=lambda r: r["avg_latency_ms"])
        most_persons = max(results, key=lambda r: r["avg_persons"])
        best_conf = max(results, key=lambda r: r["avg_person_conf"])
        print()
        print("  SUMMARY:")
        print(f"    Fastest model          : {fastest['model']}  ({fastest['avg_fps']} FPS)")
        print(f"    Most detections/frame  : {most_persons['model']}  ({most_persons['avg_persons']:.2f} persons avg)")
        print(f"    Highest avg confidence : {best_conf['model']}  ({best_conf['avg_person_conf']:.3f})")
        print()


# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────

def save_results_csv(results: list[dict], output_path: str):
    if not results:
        return
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved → {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and compare YOLOv8 / YOLOv9 / YOLO11 for person detection."
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: webcam index (0), video file path, or image folder path."
    )
    parser.add_argument(
        "--mode", choices=["video", "images"], default="video",
        help="'video' for webcam/video file, 'images' for a folder of images."
    )
    parser.add_argument(
        "--duration", type=int, default=30,
        help="Seconds of video to collect for benchmarking (video/webcam mode only)."
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Which models to compare. Default: all three."
    )
    parser.add_argument(
        "--conf", type=float, default=0.20,
        help="Detection confidence threshold (same for all models, default 0.20)."
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold (default 0.45)."
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: cpu)."
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warm-up frames per model (excluded from statistics)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save results as CSV, e.g. results/comparison.csv"
    )
    args = parser.parse_args()

    # ── Resolve source ──────────────────────────────────────
    source = args.source
    try:
        source = int(source)       # webcam index
    except ValueError:
        pass                       # keep as string (file path)

    # ── Collect frames once (shared across all models) ──────
    print("\n[1/3] Collecting benchmark frames …")
    if args.mode == "images":
        frames = collect_frames_from_images(str(source))
    else:
        frames = collect_frames_from_source(source, duration_sec=args.duration)

    if not frames:
        print("ERROR: No frames collected. Check your source.")
        sys.exit(1)

    # ── Run each model ───────────────────────────────────────
    print(f"\n[2/3] Running inference for {len(args.models)} model(s) …")
    results = []
    for model_key in args.models:
        weights = MODEL_REGISTRY[model_key]
        print(f"\n  [{model_key}]  weights={weights}")
        print( "  Loading model (will auto-download if not cached) …")

        try:
            detector = YoloDetector(
                weights_path=weights,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
            )
        except Exception as e:
            print(f"  ERROR loading {model_key}: {e}")
            continue

        print(f"  Benchmarking on {len(frames)} frames (warmup={args.warmup}) …")
        stats = run_benchmark_on_frames(detector, frames, warmup=args.warmup)
        if stats:
            results.append(stats)
            print(f"  → {stats['avg_fps']} FPS  |  "
                  f"{stats['avg_persons']:.2f} persons/frame  |  "
                  f"conf {stats['avg_person_conf']:.3f}")

        # Explicitly delete to free memory before loading next model
        del detector

    # ── Display results ──────────────────────────────────────
    print("\n[3/3] Comparison results:")
    print_comparison_table(results)

    if args.output:
        save_results_csv(results, args.output)

    return results


if __name__ == "__main__":
    main()
