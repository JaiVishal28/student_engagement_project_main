"""
Repair malformed engagement CSV files with column misalignment.
Backs up original and creates a clean version with proper alignment.
"""
import os
import sys
import pandas as pd
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_logger

logger = get_logger("RepairCSV")


def repair_csv(input_path: str, output_path: str = None, backup: bool = True):
    """
    Repair malformed CSV with column misalignment.
    
    Args:
        input_path: Path to malformed CSV
        output_path: Path to save repaired CSV (None = overwrite original)
        backup: Whether to backup original file
    """
    if not os.path.exists(input_path):
        logger.error(f"File not found: {input_path}")
        return False
    
    # Backup original if requested
    if backup:
        backup_path = input_path + ".backup"
        import shutil
        shutil.copy2(input_path, backup_path)
        logger.info(f"✓ Backed up original to: {backup_path}")
    
    # Define expected columns
    EXPECTED_COLUMNS = [
        'timestamp', 'student_id', 'engagement_score',
        'gaze', 'mouth_open', 'eye_openness', 'head_pitch', 'movement',
        'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
        'audio_energy', 'speech_probability', 'speaker_count',
        'background_noise_level', 'audio_engagement_score'
    ]
    
    logger.info(f"Reading CSV: {input_path}")
    
    try:
        # Read with no header first to see what we have
        df_raw = pd.read_csv(input_path, header=None, on_bad_lines='skip')
        logger.info(f"Raw data shape: {df_raw.shape}")
        
        # Check if first row looks like header
        first_row = df_raw.iloc[0].tolist()
        has_header = 'timestamp' in str(first_row[0]).lower() or 'student' in str(first_row).lower()
        
        if has_header:
            logger.info("Detected header row, skipping it")
            df_raw = df_raw.iloc[1:]
        
        # If more columns than expected, truncate
        if df_raw.shape[1] > len(EXPECTED_COLUMNS):
            logger.warning(f"Found {df_raw.shape[1]} columns, expected {len(EXPECTED_COLUMNS)}")
            logger.info(f"Truncating to first {len(EXPECTED_COLUMNS)} columns")
            df_raw = df_raw.iloc[:, :len(EXPECTED_COLUMNS)]
        
        # If fewer columns, pad with NaN
        elif df_raw.shape[1] < len(EXPECTED_COLUMNS):
            logger.warning(f"Found {df_raw.shape[1]} columns, expected {len(EXPECTED_COLUMNS)}")
            logger.info(f"Padding with {len(EXPECTED_COLUMNS) - df_raw.shape[1]} empty columns")
            for i in range(len(EXPECTED_COLUMNS) - df_raw.shape[1]):
                df_raw[df_raw.shape[1] + i] = 0
        
        # Assign column names
        df_raw.columns = EXPECTED_COLUMNS
        
        # Clean data types
        logger.info("Cleaning data types...")
        
        # Numeric columns
        numeric_cols = [
            'student_id', 'engagement_score', 'mouth_open', 'eye_openness',
            'head_pitch', 'movement', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
            'audio_energy', 'speech_probability', 'speaker_count',
            'background_noise_level', 'audio_engagement_score'
        ]
        
        for col in numeric_cols:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
        
        # Convert bbox columns to integers
        bbox_cols = ['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']
        for col in bbox_cols:
            df_raw[col] = df_raw[col].astype(int)
        
        # Remove invalid rows (zero bounding boxes)
        valid_mask = (
            (df_raw['bbox_xmax'] > df_raw['bbox_xmin']) &
            (df_raw['bbox_ymax'] > df_raw['bbox_ymin'])
        )
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} rows with invalid bounding boxes")
            df_raw = df_raw[valid_mask]
        
        # Save repaired CSV
        output = output_path if output_path else input_path
        df_raw.to_csv(output, index=False)
        
        logger.info("="*60)
        logger.info("✓ CSV REPAIR COMPLETE")
        logger.info("="*60)
        logger.info(f"Input:  {input_path}")
        logger.info(f"Output: {output}")
        logger.info(f"Rows:   {len(df_raw)}")
        logger.info(f"Columns: {len(df_raw.columns)}")
        logger.info(f"Students: {df_raw['student_id'].nunique()}")
        logger.info(f"Avg Engagement: {df_raw['engagement_score'].mean():.3f}")
        logger.info("="*60)
        
        # Show sample
        logger.info("\nFirst 3 rows:")
        print(df_raw.head(3).to_string())
        
        return True
        
    except Exception as e:
        logger.error(f"Error repairing CSV: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Repair malformed engagement CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Repair CSV in-place (with backup)
  python scripts/repair_csv.py --input data/labels/engagement_data.csv
  
  # Repair and save to new file
  python scripts/repair_csv.py --input data/labels/engagement_data.csv --output data/labels/engagement_data_fixed.csv
  
  # Repair without backup
  python scripts/repair_csv.py --input data/labels/engagement_data.csv --no-backup
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to malformed CSV file')
    parser.add_argument('--output', type=str,
                       help='Path to save repaired CSV (default: overwrite input)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup of original file')
    
    args = parser.parse_args()
    
    success = repair_csv(args.input, args.output, backup=not args.no_backup)
    
    if success:
        logger.info("\n✓ Repair successful! You can now run visualize_detections.py")
    else:
        logger.error("\n✗ Repair failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
