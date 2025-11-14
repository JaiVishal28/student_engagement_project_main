import csv
import datetime
import os

class DataLogger:
    """
    A class to handle logging student engagement data to a CSV file.
    It creates a CSV file in the 'data' folder and writes a header
    containing all the features we extract.
    """
    
    def __init__(self, file_name='engagement_data.csv'):
        # Ensure the 'data' directory exists
        self.data_folder = 'data'
        os.makedirs(self.data_folder, exist_ok=True)
        
        self.file_path = os.path.join(self.data_folder, file_name)
        self.file = None
        self.writer = None
        self.header = self._get_header()
        self._initialize_file()

    def _get_header(self):
        """Defines the header for our CSV file."""
        header = ['timestamp', 'student_id', 'gaze', 'mouth_open', 'eye_openness']
        
        # Add headers for 33 pose landmarks (x, y, z, visibility)
        for i in range(33):
            header.extend([
                f'pose_lm_{i}_x', f'pose_lm_{i}_y', 
                f'pose_lm_{i}_z', f'pose_lm_{i}_visibility'
            ])
        return header

    def _initialize_file(self):
        """Opens the CSV file and writes the header if it's a new file."""
        file_exists = os.path.isfile(self.file_path)
        
        # Open in 'append' mode ('a') so we don't overwrite old data
        self.file = open(self.file_path, 'a', newline='')
        self.writer = csv.writer(self.file)
        
        if not file_exists:
            self.writer.writerow(self.header)
            self.file.flush()

    def log_data(self, all_student_data):
        """Writes data for all detected students from a single frame."""
        timestamp = datetime.datetime.now().isoformat()
        
        for data in all_student_data:
            row = [
                timestamp,
                data['student_id'],
                data['gaze'],
                data['mouth_open'],
                data['eye_openness']
            ]
            
            pose_landmarks = data.get('pose_landmarks')
            if pose_landmarks:
                for lm in pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                # If no pose is found, fill with empty values
                row.extend([None] * (33 * 4)) 
                
            self.writer.writerow(row)
        
        # Flush the buffer to ensure data is written to disk
        self.file.flush()

    def close(self):
        """Closes the CSV file."""
        if self.file:
            self.file.close()