import os
import csv
import datetime
import numpy as np
from typing import List, Dict, Any
from src.logging_utils import get_logger

logger = get_logger("DataLogger")


class DataLogger:
    """
    Logger for student engagement data with CSV output.
    Supports logging features, scores, and metadata.
    """
    
    def __init__(self, csv_path: str = "data/labels/engagement_data.csv"):
        """
        Initialize data logger.
        
        Args:
            csv_path: Path to output CSV file
        """
        

        self.csv_path = csv_path
        logger.info(f"CSV absolute path: {os.path.abspath(self.csv_path)}")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._ensure_header()
        self.log_count = 0
        self.student_ids = set()
        self.engagement_sum = 0.0 # For mean calculation

    def _ensure_header(self):
        """Create CSV file with header if it doesn't exist."""
        # Define exact columns (store as class variable)
        self.csv_columns = [
            'timestamp', 'student_id', 'engagement_score',
            'gaze', 'mouth_open', 'eye_openness', 'head_pitch', 'movement',
            'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
            'audio_energy', 'speech_probability', 'speaker_count',
            'background_noise_level', 'audio_engagement_score'
        ]
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_columns)
            logger.info(f"Created CSV header at {self.csv_path} with {len(self.csv_columns)} columns")

    def log(self, rows: List[Dict[str, Any]]):
        """
        Log engagement data rows to CSV.

        Args:
            rows: List of dictionaries with engagement data
        """
        if not rows:
            return

        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)

                for r in rows:
                    timestamp = datetime.datetime.now().isoformat()
                    self.student_ids.add(r.get('student_id'))
                    self.engagement_sum += float(r.get('engagement_score', 0.0))

                    # Build row matching EXACTLY the header columns
                    row = [
                        timestamp,
                        r.get('student_id', 'unknown'),
                        r.get('engagement_score', 0.0),
                        r.get('gaze', 'unknown'),
                        r.get('mouth_open', 0.0),
                        r.get('eye_openness', 0.0),
                        r.get('head_pitch', 0.0),
                        r.get('movement', 0.0),
                        r.get('bbox_xmin', 0),
                        r.get('bbox_ymin', 0),
                        r.get('bbox_xmax', 0),
                        r.get('bbox_ymax', 0),
                        r.get('audio_energy', 0.0),
                        r.get('speech_probability', 0.0),
                        r.get('speaker_count', 0),
                        r.get('background_noise_level', 0.0),  # Changed to numeric to avoid string alignment issues
                        r.get('audio_engagement_score', 0.0),
                    ]
                    
                    # Verify row length matches header
                    if len(row) != len(self.csv_columns):
                        logger.error(f"Row length mismatch: {len(row)} != {len(self.csv_columns)}")
                        continue
                    
                    writer.writerow(row)

                # 🔥 CRITICAL FOR WINDOWS: force write to disk
                f.flush()
                os.fsync(f.fileno())

            self.log_count += len(rows)
            logger.debug(f"Logged {len(rows)} rows (total: {self.log_count})")

        except Exception as e:
            logger.error(f"Error logging data: {e}", exc_info=True)

    def get_summary(self):
        mean_engagement = (
            self.engagement_sum / self.log_count
            if self.log_count > 0 else 0.0
        )

        return {
            'total_records': self.log_count,
            'unique_students': len(self.student_ids),
            'mean_engagement': mean_engagement,
            'log_file': self.csv_path
        }

      
    # def get_summary(self):
    #     try:
    #         import pandas as pd
    #         df = pd.read_csv(self.csv_path)

    #         if df.empty or 'engagement_score' not in df.columns:
    #             return {
    #                 'total_records': 0,
    #                 'unique_students': 0,
    #                 'mean_engagement': 0.0,
    #                 'std_engagement': 0.0,
    #                 'log_file': self.csv_path
    #             }

    #         return {
    #             'total_records': len(df),
    #             'unique_students': df['student_id'].nunique(),
    #             'mean_engagement': float(df['engagement_score'].mean()),
    #             'std_engagement': float(df['engagement_score'].std()),
    #             'log_file': self.csv_path
    #         }
    #     except Exception as e:
    #         logger.error(f"Error getting summary: {e}")
    #         return {
    #             'total_records': 0,
    #             'unique_students': 0,
    #             'mean_engagement': 0.0,
    #             'std_engagement': 0.0,
    #             'log_file': self.csv_path
    #         }


    # def get_summary(self) -> Dict[str, Any]:
    #     """
    #     Get summary statistics of logged data.
        
    #     Returns:
    #         Dictionary with summary statistics
    #     """
    #     try:
    #         import pandas as pd
    #         df = pd.read_csv(self.csv_path)
            
    #         return {
    #             'total_records': len(df),
    #             'unique_students': df['student_id'].nunique(),
    #             'mean_engagement': df['engagement_score'].mean(),
    #             'std_engagement': df['engagement_score'].std(),
    #             'log_file': self.csv_path
    #         }
    #     except Exception as e:
    #         logger.error(f"Error getting summary: {e}")
    #         return {'error': str(e)}
