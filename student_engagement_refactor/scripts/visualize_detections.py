"""
Visualize detected students from video with their IDs.
Generates screenshots showing all detected people with bounding boxes and IDs.
"""
import os
import sys
import yaml
import cv2
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_logger
from src.detection.yolov_wrapper import YoloDetector
from src.tracking.sort_tracker import Sort

logger = get_logger("VisualizeDetections")


def load_engagement_data(csv_path: str) -> pd.DataFrame:
    """Load engagement data CSV."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from {csv_path}")
    logger.info(f"Unique students detected: {df['student_id'].nunique()}")
    return df


def find_best_frames_per_student(df: pd.DataFrame, min_engagement: float = 0.5):
    """
    Find the best frame for each student (highest engagement score).
    
    Args:
        df: Engagement dataframe
        min_engagement: Minimum engagement threshold
    
    Returns:
        Dictionary mapping student_id to frame number
    """
    best_frames = {}
    
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id]
        
        # Find frame with highest engagement score
        best_row = student_data.loc[student_data['engagement_score'].idxmax()]
        best_frames[student_id] = {
            'frame': int(best_row['frame']),
            'engagement': float(best_row['engagement_score']),
            'bbox': [
                int(best_row['bbox_x']),
                int(best_row['bbox_y']),
                int(best_row['bbox_w']),
                int(best_row['bbox_h'])
            ]
        }
    
    return best_frames


def create_detection_summary(video_path: str, csv_path: str, output_dir: str, 
                             mode: str = 'grid', max_students: int = None):
    """
    Create visual summary of all detected students.
    
    Args:
        video_path: Path to video file
        csv_path: Path to engagement CSV
        output_dir: Output directory for images
        mode: 'grid' for collage, 'frames' for individual frames, 'both'
        max_students: Maximum number of students to show (None = all)
    """
    # Load engagement data
    df = load_engagement_data(csv_path)
    if df is None:
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {total_frames} frames @ {fps:.1f} FPS")
    
    # Find best frames for each student
    best_frames = find_best_frames_per_student(df)
    student_ids = sorted(best_frames.keys())
    
    if max_students:
        student_ids = student_ids[:max_students]
    
    logger.info(f"Processing {len(student_ids)} unique students")
    
    # Extract crops for each student
    student_crops = {}
    
    for student_id in student_ids:
        frame_info = best_frames[student_id]
        frame_num = frame_info['frame']
        bbox = frame_info['bbox']
        engagement = frame_info['engagement']
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {frame_num} for student {student_id}")
            continue
        
        # Extract crop with padding
        x, y, w, h = bbox
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        crop = frame[y1:y2, x1:x2].copy()
        
        # Draw bbox on crop (adjusted coordinates)
        cv2.rectangle(crop, 
                     (padding, padding), 
                     (padding + w, padding + h), 
                     (0, 255, 0), 2)
        
        # Add student ID and engagement score
        label = f"ID: {student_id}"
        score_label = f"Eng: {engagement:.2f}"
        
        cv2.putText(crop, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(crop, score_label, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        student_crops[student_id] = crop
        
        # Save individual frame if requested
        if mode in ['frames', 'both']:
            individual_path = os.path.join(output_dir, f'student_{student_id:03d}.jpg')
            cv2.imwrite(individual_path, crop)
            logger.info(f"Saved student {student_id} to {individual_path}")
    
    cap.release()
    
    # Create grid collage if requested
    if mode in ['grid', 'both'] and student_crops:
        create_student_grid(student_crops, output_dir, df)


def create_student_grid(student_crops: dict, output_dir: str, df: pd.DataFrame):
    """Create a grid collage of all detected students."""
    n_students = len(student_crops)
    
    # Calculate grid dimensions
    cols = min(5, n_students)
    rows = (n_students + cols - 1) // cols
    
    # Find max dimensions
    max_h = max(crop.shape[0] for crop in student_crops.values())
    max_w = max(crop.shape[1] for crop in student_crops.values())
    
    # Create canvas
    canvas_h = rows * (max_h + 20) + 100
    canvas_w = cols * (max_w + 20) + 40
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # Add title
    title = f"Detected Students: {n_students} unique IDs"
    cv2.putText(canvas, title, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    stats = f"Total frames: {len(df)} | Avg engagement: {df['engagement_score'].mean():.3f}"
    cv2.putText(canvas, stats, (20, 75), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    # Place crops in grid
    for idx, (student_id, crop) in enumerate(sorted(student_crops.items())):
        row = idx // cols
        col = idx % cols
        
        y_offset = 100 + row * (max_h + 20) + 10
        x_offset = 20 + col * (max_w + 20) + 10
        
        # Resize crop if needed
        if crop.shape[0] != max_h or crop.shape[1] != max_w:
            # Maintain aspect ratio
            scale = min(max_w / crop.shape[1], max_h / crop.shape[0])
            new_w = int(crop.shape[1] * scale)
            new_h = int(crop.shape[0] * scale)
            crop = cv2.resize(crop, (new_w, new_h))
        
        # Center crop in cell
        y_center = y_offset + (max_h - crop.shape[0]) // 2
        x_center = x_offset + (max_w - crop.shape[1]) // 2
        
        canvas[y_center:y_center+crop.shape[0], 
               x_center:x_center+crop.shape[1]] = crop
    
    # Save grid
    grid_path = os.path.join(output_dir, 'all_detected_students_grid.jpg')
    cv2.imwrite(grid_path, canvas)
    logger.info(f"✓ Saved student grid to {grid_path}")
    logger.info(f"  Grid size: {cols} columns × {rows} rows")


def create_annotated_frame(video_path: str, csv_path: str, output_dir: str, 
                          frame_num: int = None, sample_interval: int = 30):
    """
    Create annotated frame(s) showing all detected students at once.
    
    Args:
        video_path: Path to video file
        csv_path: Path to engagement CSV
        output_dir: Output directory
        frame_num: Specific frame to annotate (None = auto-select)
        sample_interval: Sample every N frames to find frame with most students
    """
    df = load_engagement_data(csv_path)
    if df is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find frame with most students detected if not specified
    if frame_num is None:
        frame_counts = df.groupby('frame')['student_id'].count()
        frame_num = frame_counts.idxmax()
        logger.info(f"Auto-selected frame {frame_num} with {frame_counts.max()} students")
    
    # Get all students in that frame
    frame_data = df[df['frame'] == frame_num]
    
    # Open video and read frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error(f"Failed to read frame {frame_num}")
        return
    
    # Draw all detections
    annotated = frame.copy()
    
    for _, row in frame_data.iterrows():
        student_id = int(row['student_id'])
        x, y, w, h = int(row['bbox_x']), int(row['bbox_y']), int(row['bbox_w']), int(row['bbox_h'])
        engagement = float(row['engagement_score'])
        
        # Color based on engagement
        if engagement >= 0.7:
            color = (0, 255, 0)  # Green - highly engaged
        elif engagement >= 0.4:
            color = (0, 255, 255)  # Yellow - moderately engaged
        else:
            color = (0, 0, 255)  # Red - low engagement
        
        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 3)
        
        # Draw label background
        label = f"ID:{student_id} | {engagement:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x, y-label_h-10), (x+label_w+10, y), color, -1)
        
        # Draw label text
        cv2.putText(annotated, label, (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add frame info
    info_text = f"Frame {frame_num} | Students: {len(frame_data)} | Avg Engagement: {frame_data['engagement_score'].mean():.3f}"
    cv2.putText(annotated, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # Save annotated frame
    output_path = os.path.join(output_dir, f'annotated_frame_{frame_num}.jpg')
    cv2.imwrite(output_path, annotated)
    logger.info(f"✓ Saved annotated frame to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize detected students from video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create grid collage of all detected students
  python scripts/visualize_detections.py --video lecture.mp4 --csv data/labels/engagement_data.csv
  
  # Save individual crops for each student
  python scripts/visualize_detections.py --video lecture.mp4 --csv data/labels/engagement_data.csv --mode frames
  
  # Create annotated frame showing all students at once
  python scripts/visualize_detections.py --video lecture.mp4 --csv data/labels/engagement_data.csv --annotated-frame
        """
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--csv', type=str, 
                       default='data/labels/engagement_data.csv',
                       help='Path to engagement CSV file')
    parser.add_argument('--output-dir', type=str, 
                       default='results/detected_students',
                       help='Output directory for images')
    parser.add_argument('--mode', type=str, 
                       choices=['grid', 'frames', 'both'],
                       default='both',
                       help='Output mode: grid collage, individual frames, or both')
    parser.add_argument('--max-students', type=int,
                       help='Maximum number of students to show')
    parser.add_argument('--annotated-frame', action='store_true',
                       help='Also create annotated frame showing all students at once')
    parser.add_argument('--frame', type=int,
                       help='Specific frame number for annotated frame (auto-selects if not specified)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.csv):
        logger.error(f"CSV file not found: {args.csv}")
        return
    
    logger.info("="*60)
    logger.info("Student Detection Visualization")
    logger.info("="*60)
    logger.info(f"Video: {args.video}")
    logger.info(f"CSV: {args.csv}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*60)
    
    # Create detection summary
    create_detection_summary(args.video, args.csv, args.output_dir, 
                           args.mode, args.max_students)
    
    # Create annotated frame if requested
    if args.annotated_frame:
        create_annotated_frame(args.video, args.csv, args.output_dir, args.frame)
    
    logger.info("="*60)
    logger.info("✓ Visualization complete!")
    logger.info(f"Check output directory: {args.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
