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
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._ensure_header()
        self.log_count = 0

    def _ensure_header(self):
        """Create CSV file with header if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            header = [
                'timestamp', 'student_id', 'engagement_score',
                'gaze', 'mouth_open', 'eye_openness', 'head_pitch', 'movement',
                'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax'
            ]
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            logger.info(f"Created CSV header at {self.csv_path}")

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
                timestamp = datetime.datetime.now().isoformat()
                
                for r in rows:
                    writer.writerow([
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
                    ])
            
            self.log_count += len(rows)
            logger.debug(f"Logged {len(rows)} rows (total: {self.log_count})")
        
        except Exception as e:
            logger.error(f"Error logging data: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of logged data.
        
        Returns:
            Dictionary with summary statistics
        """
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            
            return {
                'total_records': len(df),
                'unique_students': df['student_id'].nunique(),
                'mean_engagement': df['engagement_score'].mean(),
                'std_engagement': df['engagement_score'].std(),
                'log_file': self.csv_path
            }
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}
