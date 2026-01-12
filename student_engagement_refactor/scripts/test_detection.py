"""
Test detection performance on a video to diagnose why students are missed.
Shows frame-by-frame detection counts and visualizes results.
"""
import os
import sys
import cv2
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_logger
from src.detection.yolov_wrapper import YoloDetector

logger = get_logger("TestDetection")


def test_detection(video_path: str, output_dir: str = "results/detection_test", 
                  conf_threshold: float = 0.25, sample_frames: int = 10):
    """
    Test detection on sample frames and show results.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for annotated frames
        conf_threshold: Confidence threshold for detection
        sample_frames: Number of frames to sample evenly
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info("="*60)
    logger.info("Video Information")
    logger.info("="*60)
    logger.info(f"Path: {video_path}")
    logger.info(f"Resolution: {width}x{height}")
    logger.info(f"FPS: {fps:.1f}")
    logger.info(f"Total Frames: {total_frames}")
    logger.info(f"Duration: {total_frames/fps:.1f} seconds")
    logger.info("="*60)
    
    # Initialize detector with lower threshold
    logger.info(f"\nInitializing detector (conf={conf_threshold})...")
    detector = YoloDetector(
        weights_path="models/weights/yolov8s.pt",
        device="cpu",
        conf=conf_threshold,
        iou=0.45
    )
    
    # Sample frames evenly throughout video
    frame_indices = [int(i * total_frames / sample_frames) for i in range(sample_frames)]
    
    detection_counts = []
    
    logger.info(f"\nTesting detection on {sample_frames} sample frames...")
    logger.info("="*60)
    
    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {frame_num}")
            continue
        
        # Detect people
        detections = detector.detect(frame)
        detection_counts.append(len(detections))
        
        # Draw detections
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['conf']
            
            # Color based on confidence
            if conf >= 0.5:
                color = (0, 255, 0)  # High confidence - green
            elif conf >= 0.35:
                color = (0, 255, 255)  # Medium - yellow
            else:
                color = (0, 165, 255)  # Low - orange
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{conf:.2f}"
            cv2.putText(annotated, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add info overlay
        info = f"Frame {frame_num}/{total_frames} | Detected: {len(detections)} people"
        cv2.putText(annotated, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Save annotated frame
        output_path = os.path.join(output_dir, f'frame_{frame_num:04d}_det{len(detections)}.jpg')
        cv2.imwrite(output_path, annotated)
        
        logger.info(f"Frame {frame_num:4d}: {len(detections):2d} detections (conf ≥ {conf_threshold}) → {output_path}")
    
    cap.release()
    
    # Summary statistics
    logger.info("="*60)
    logger.info("Detection Summary")
    logger.info("="*60)
    logger.info(f"Frames tested: {len(detection_counts)}")
    logger.info(f"Min detections: {min(detection_counts) if detection_counts else 0}")
    logger.info(f"Max detections: {max(detection_counts) if detection_counts else 0}")
    logger.info(f"Avg detections: {sum(detection_counts)/len(detection_counts):.1f}" if detection_counts else "N/A")
    logger.info("="*60)
    logger.info(f"\n✓ Annotated frames saved to: {output_dir}")
    logger.info("\nRECOMMENDATIONS:")
    
    avg_det = sum(detection_counts)/len(detection_counts) if detection_counts else 0
    
    if avg_det < 5:
        logger.warning("⚠ Very low detection rate!")
        logger.info("  • Try lowering confidence threshold: --conf 0.15")
        logger.info("  • Check if video is too low resolution")
        logger.info("  • Verify students are clearly visible")
    elif avg_det < 10:
        logger.warning("⚠ Moderate detection rate")
        logger.info("  • Consider lowering confidence threshold: --conf 0.2")
        logger.info("  • Check tracking settings in config.yaml")
    else:
        logger.info("✓ Good detection rate")
        logger.info("  • Current settings seem appropriate")
    
    logger.info("\nTo improve detection in your main project:")
    logger.info("  1. Edit config.yaml → detection.conf = 0.25 (lower)")
    logger.info("  2. Edit config.yaml → tracking.min_hits = 3 (lower)")
    logger.info("  3. Edit config.yaml → processing.process_every_n_frames = 1 (process every frame)")


def main():
    parser = argparse.ArgumentParser(
        description='Test detection performance on video',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output-dir', type=str, 
                       default='results/detection_test',
                       help='Output directory for annotated frames')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25, lower = more detections)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of sample frames to test')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    test_detection(args.video, args.output_dir, args.conf, args.samples)


if __name__ == "__main__":
    main()
