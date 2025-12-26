"""
Evaluation script for student engagement detection system.
Computes metrics on a test dataset with ground truth annotations.
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_logger
from src.evaluation.metrics import compute_metrics, confusion_matrix_metrics

logger = get_logger("Evaluate")


def load_predictions_and_labels(predictions_csv: str, labels_csv: str) -> tuple:
    """
    Load predictions and ground truth labels.
    
    Args:
        predictions_csv: Path to CSV with predicted engagement scores
        labels_csv: Path to CSV with ground truth labels
    
    Returns:
        (y_true, y_pred) numpy arrays
    """
    try:
        pred_df = pd.read_csv(predictions_csv)
        label_df = pd.read_csv(labels_csv)
        
        # Merge on timestamp and student_id
        merged = pd.merge(pred_df, label_df, on=['timestamp', 'student_id'], 
                         suffixes=('_pred', '_true'))
        
        y_pred = merged['engagement_score'].values
        y_true = merged['ground_truth_engagement'].values
        
        return y_true, y_pred
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return np.array([]), np.array([])


def evaluate_from_csv(predictions_csv: str, labels_csv: str, output_path: str = None):
    """
    Evaluate model performance from CSV files.
    
    Args:
        predictions_csv: Path to predictions
        labels_csv: Path to ground truth
        output_path: Optional path to save results
    """
    logger.info("Loading predictions and labels...")
    y_true, y_pred = load_predictions_and_labels(predictions_csv, labels_csv)
    
    if len(y_true) == 0:
        logger.error("No data loaded for evaluation")
        return
    
    logger.info(f"Loaded {len(y_true)} samples")
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Sample Size: {len(y_true)}")
    logger.info(f"\nRegression Metrics:")
    logger.info(f"  MAE:         {metrics['mae']:.4f}")
    logger.info(f"  RMSE:        {metrics['rmse']:.4f}")
    logger.info(f"  Correlation: {metrics['correlation']:.4f}")
    logger.info(f"\nClassification Metrics (threshold=0.5):")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:    {metrics['f1']:.4f}")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    logger.info("="*50 + "\n")
    
    # Save results if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df = pd.DataFrame([metrics])
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Also save detailed comparison
        comparison_df = pd.DataFrame({
            'ground_truth': y_true,
            'predicted': y_pred,
            'error': np.abs(y_true - y_pred)
        })
        comparison_path = output_path.replace('.csv', '_detailed.csv')
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Detailed comparison saved to {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate student engagement model')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV file')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to ground truth labels CSV file')
    parser.add_argument('--output', type=str, default='results/evaluation/metrics.csv',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_from_csv(args.predictions, args.labels, args.output)


if __name__ == "__main__":
    main()
