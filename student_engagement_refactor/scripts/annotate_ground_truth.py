"""
Annotation tool for creating ground truth labels.
Allows manual labeling of engagement levels for validation.
"""
import os
import sys
import cv2
import pandas as pd
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_logger

logger = get_logger("Annotate")


class AnnotationTool:
    """Interactive annotation tool for engagement labeling."""
    
    def __init__(self, video_path: str, output_csv: str):
        self.video_path = video_path
        self.output_csv = output_csv
        self.cap = cv2.VideoCapture(video_path)
        self.annotations = []
        self.current_frame = 0
        self.paused = False
        self.engagement_level = 0.5  # Default middle engagement
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        logger.info(f"Loaded video: {video_path}")
        logger.info("Controls: Space=pause, 0-9=set engagement (0=low, 9=high), Q=quit and save")
    
    def run(self):
        """Run the annotation interface."""
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                frame = self.current_frame_img.copy()
            
            # Store current frame for paused mode
            self.current_frame_img = frame.copy()
            
            # Draw UI
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Engagement bar
            bar_height = 40
            bar_y = h - 60
            cv2.rectangle(display, (10, bar_y), (w - 10, bar_y + bar_height), (50, 50, 50), -1)
            
            # Fill engagement level
            fill_width = int((w - 20) * self.engagement_level)
            color = (0, int(255 * self.engagement_level), int(255 * (1 - self.engagement_level)))
            cv2.rectangle(display, (10, bar_y), (10 + fill_width, bar_y + bar_height), color, -1)
            
            # Text overlay
            cv2.putText(display, f"Frame: {self.current_frame}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Engagement: {self.engagement_level:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Space: Pause | 0-9: Set Level | Q: Save & Quit", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow("Annotation Tool", display)
            
            key = cv2.waitKey(30 if not self.paused else 0) & 0xFF
            
            if key == ord('q'):
                logger.info("Quitting annotation tool")
                break
            elif key == ord(' '):
                self.paused = not self.paused
                logger.info(f"{'Paused' if self.paused else 'Resumed'}")
            elif ord('0') <= key <= ord('9'):
                # Set engagement level 0-9 maps to 0.0-1.0
                self.engagement_level = (key - ord('0')) / 9.0
                logger.info(f"Engagement set to {self.engagement_level:.2f}")
                
                # Record annotation
                self.annotations.append({
                    'frame': self.current_frame,
                    'timestamp': self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                    'ground_truth_engagement': self.engagement_level
                })
        
        self.save_annotations()
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_annotations(self):
        """Save annotations to CSV."""
        if not self.annotations:
            logger.warning("No annotations to save")
            return
        
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df = pd.DataFrame(self.annotations)
        df.to_csv(self.output_csv, index=False)
        logger.info(f"Saved {len(self.annotations)} annotations to {self.output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Annotation tool for ground truth labels')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default='data/labels/ground_truth.csv',
                       help='Output CSV file for annotations')
    
    args = parser.parse_args()
    
    tool = AnnotationTool(args.video, args.output)
    tool.run()


if __name__ == "__main__":
    main()
