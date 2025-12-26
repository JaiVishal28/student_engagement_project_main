import os, csv, datetime, numpy as np
from src.logging_utils import get_logger

logger = get_logger("DataLogger")

class DataLogger:
    def __init__(self, csv_path="data/labels/engagement_data.csv"):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.csv_path):
            header = ['timestamp','student_id','gaze','mouth_open','eye_openness','bbox_xmin','bbox_ymin','bbox_xmax','bbox_ymax']
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            logger.info("Created CSV header at %s", self.csv_path)

    def log(self, rows):
        """rows: list of dicts with keys same as header"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    r.get('student_id'),
                    r.get('gaze'),
                    r.get('mouth_open'),
                    r.get('eye_openness'),
                    r.get('bbox_xmin'),
                    r.get('bbox_ymin'),
                    r.get('bbox_xmax'),
                    r.get('bbox_ymax'),
                ])
        logger.debug("Logged %d rows", len(rows))
