"""
Batch processing script for analyzing multiple videos or images.
Useful for processing datasets for evaluation.
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_logger
from src.main import run_video_mode, run_image_mode

logger = get_logger("BatchProcess")


def process_video_batch(video_dir: str, output_dir: str, max_frames_per_video: int = None):
    """
    Process multiple videos in batch.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save results
        max_frames_per_video: Maximum frames to process per video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
    
    logger.info(f"Found {len(video_files)} video files")
    
    results = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        logger.info(f"Processing: {video_path.name}")
        
        try:
            # Set output CSV for this video
            output_csv = os.path.join(output_dir, f"{video_path.stem}_engagement.csv")
            
            # Process video (without display)
            run_video_mode(
                source=str(video_path),
                display=False,
                max_frames=max_frames_per_video
            )
            
            results.append({
                'video': video_path.name,
                'status': 'success',
                'output': output_csv
            })
            
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            results.append({
                'video': video_path.name,
                'status': 'error',
                'error': str(e)
            })
    
    # Save processing summary
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, 'batch_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Batch processing complete. Summary saved to {summary_path}")


def process_image_batch(image_dir: str, output_dir: str):
    """
    Process multiple images in batch.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
    
    logger.info(f"Found {len(image_files)} image files")
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            run_image_mode(str(image_path))
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
    
    logger.info("Batch image processing complete")


def main():
    parser = argparse.ArgumentParser(description='Batch process videos or images')
    parser.add_argument('input_dir', type=str, help='Input directory with videos/images')
    parser.add_argument('--output-dir', type=str, default='results/batch',
                       help='Output directory for results')
    parser.add_argument('--type', choices=['video', 'image'], default='video',
                       help='Type of files to process')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames per video')
    
    args = parser.parse_args()
    
    if args.type == 'video':
        process_video_batch(args.input_dir, args.output_dir, args.max_frames)
    else:
        process_image_batch(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
