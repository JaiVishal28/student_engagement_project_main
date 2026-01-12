"""
Annotation tool for creating ground truth labels.
Allows manual labeling of engagement levels for validation.
Video is auto-resized to fit screen while preserving aspect ratio.
"""

import os
import sys
import cv2
import pandas as pd
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_logger

logger = get_logger("Annotate")


def resize_to_screen(frame, max_width=1280, max_height=720):
    """
    Resize frame to fit within max_width x max_height
    while preserving aspect ratio (no cropping).
    """
    h, w = frame.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    if scale < 1.0:
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return frame


class AnnotationTool:
    """Interactive annotation tool for engagement labeling."""

    def __init__(self, video_path: str, output_csv: str):
        self.video_path = video_path
        self.output_csv = output_csv
        self.cap = cv2.VideoCapture(video_path)

        self.annotations = []
        self.current_frame = 0
        self.paused = False
        self.engagement_level = 0.5  # Default medium engagement
        self.current_frame_img = None

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        logger.info(f"Loaded video: {video_path}")
        logger.info(
            "Controls: Space = Pause/Resume | 0-9 = Set engagement | Q = Save & Quit"
        )

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

            # Resize frame to fit screen
            display = resize_to_screen(frame)

            # Store resized frame for pause mode
            self.current_frame_img = display.copy()

            # UI overlay
            h, w = display.shape[:2]

            # Engagement bar
            bar_height = 35
            bar_y = h - 60

            cv2.rectangle(
                display,
                (10, bar_y),
                (w - 10, bar_y + bar_height),
                (40, 40, 40),
                -1,
            )

            fill_width = int((w - 20) * self.engagement_level)
            bar_color = (
                0,
                int(255 * self.engagement_level),
                int(255 * (1 - self.engagement_level)),
            )

            cv2.rectangle(
                display,
                (10, bar_y),
                (10 + fill_width, bar_y + bar_height),
                bar_color,
                -1,
            )

            # Text overlays
            cv2.putText(
                display,
                f"Frame: {self.current_frame}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"Engagement: {self.engagement_level:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                "SPACE: Pause | 0-9: Label | Q: Save & Quit",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            cv2.imshow("Annotation Tool", display)

            key = cv2.waitKey(30 if not self.paused else 0) & 0xFF

            if key == ord("q"):
                logger.info("User requested quit")
                break

            elif key == ord(" "):
                self.paused = not self.paused
                logger.info("Paused" if self.paused else "Resumed")

            elif ord("0") <= key <= ord("9"):
                self.engagement_level = (key - ord("0")) / 9.0
                logger.info(f"Engagement set to {self.engagement_level:.2f}")

                self.annotations.append(
                    {
                        "frame": self.current_frame,
                        "timestamp": self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                        "ground_truth_engagement": self.engagement_level,
                    }
                )

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

        logger.info(
            f"Saved {len(self.annotations)} annotations to {self.output_csv}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Annotation tool for ground truth labels"
    )
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument(
        "--output",
        type=str,
        default="data/labels/ground_truth.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    tool = AnnotationTool(args.video, args.output)
    tool.run()


if __name__ == "__main__":
    main()
