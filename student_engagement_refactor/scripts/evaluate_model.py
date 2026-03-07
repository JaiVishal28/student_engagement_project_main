"""
Evaluation script for student engagement detection system.
Evaluates predictions against frame-level ground truth annotations.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_logger
from src.evaluation.metrics import compute_metrics

logger = get_logger("Evaluate")


def load_predictions_and_labels(predictions_csv: str, labels_csv: str):
    """
    Load predictions and ground truth labels and align them by frame.
    """
    try:
        # Load CSVs safely
        pred_df = pd.read_csv(
            predictions_csv,
            engine="python",
            on_bad_lines="skip"
        )
        gt_df = pd.read_csv(labels_csv)

        # ---- Normalize column names ----
        if "engagement" in pred_df.columns:
            pred_df = pred_df.rename(columns={"engagement": "engagement_score"})

        if "engagement_score" not in pred_df.columns:
            raise ValueError("Predictions CSV missing 'engagement_score'")

        if "ground_truth_engagement" not in gt_df.columns:
            raise ValueError("Ground truth CSV missing 'ground_truth_engagement'")

        # ---- Ensure frame column exists ----
        if "frame" not in pred_df.columns:
            logger.warning("Predictions CSV has no 'frame' column, using row index as frame")
            pred_df["frame"] = pred_df.index

        if "frame" not in gt_df.columns:
            raise ValueError("Ground truth CSV missing 'frame' column")

        # ---- Aggregate predictions per frame ----
        pred_frame = (
            pred_df
            .groupby("frame")["engagement_score"]
            .mean()
            .reset_index()
        )

        # ---- Merge predictions with ground truth ----
        merged = pd.merge(
            pred_frame,
            gt_df,
            on="frame",
            how="inner"
        )

        if merged.empty:
            logger.warning("No overlapping frames between predictions and ground truth")

        y_pred = merged["engagement_score"].values
        y_true = merged["ground_truth_engagement"].values

        return y_true, y_pred, merged

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return np.array([]), np.array([]), None


def evaluate_from_csv(predictions_csv: str, labels_csv: str, output_path: str = None):
    """
    Evaluate model performance from CSV files.
    """
    logger.info("Loading predictions and labels...")
    y_true, y_pred, merged_df = load_predictions_and_labels(
        predictions_csv, labels_csv
    )

    if len(y_true) == 0:
        logger.error("No data loaded for evaluation")
        return

    logger.info(f"Loaded {len(y_true)} evaluation samples")

    # ---- Compute metrics ----
    metrics = compute_metrics(y_true, y_pred)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS (Frame-Level)")
    logger.info("=" * 60)
    logger.info(f"Sample Size: {len(y_true)}")
    logger.info("\nRegression Metrics:")
    logger.info(f"  MAE:         {metrics['mae']:.4f}")
    logger.info(f"  RMSE:        {metrics['rmse']:.4f}")
    logger.info(f"  Correlation: {metrics['correlation']:.4f}")
    logger.info("\nClassification Metrics (threshold=0.5):")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:    {metrics['f1']:.4f}")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    logger.info("=" * 60 + "\n")

    # ---- Save results ----
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        pd.DataFrame([metrics]).to_csv(output_path, index=False)
        logger.info(f"Metrics saved to {output_path}")

        # Save detailed comparison
        detailed_path = output_path.replace(".csv", "_detailed.csv")
        merged_df["abs_error"] = np.abs(
            merged_df["ground_truth_engagement"] - merged_df["engagement_score"]
        )
        merged_df.to_csv(detailed_path, index=False)
        logger.info(f"Detailed comparison saved to {detailed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate student engagement model (frame-level)"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions CSV file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to ground truth labels CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation/metrics.csv",
        help="Path to save evaluation results",
    )

    args = parser.parse_args()
    evaluate_from_csv(args.predictions, args.labels, args.output)


if __name__ == "__main__":
    main()
