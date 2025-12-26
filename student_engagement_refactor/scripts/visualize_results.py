"""
Visualization script for student engagement analysis.
Creates publication-quality plots and figures.
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_utils import get_logger

logger = get_logger("Visualize")

# Try importing matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")


def plot_engagement_over_time(data_csv: str, output_dir: str):
    """Plot engagement scores over time for all students."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return
    
    df = pd.read_csv(data_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id]
        ax.plot(student_data['timestamp'], student_data['engagement_score'], 
               label=f'Student {student_id}', alpha=0.7)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Engagement Score', fontsize=12)
    ax.set_title('Student Engagement Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'engagement_timeline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved timeline plot to {output_path}")
    plt.close()


def plot_engagement_distribution(data_csv: str, output_dir: str):
    """Plot distribution of engagement scores."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return
    
    df = pd.read_csv(data_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['engagement_score'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Engagement Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Engagement Scores', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by student
    student_data = [df[df['student_id'] == sid]['engagement_score'].values 
                   for sid in df['student_id'].unique()]
    axes[1].boxplot(student_data, labels=df['student_id'].unique())
    axes[1].set_xlabel('Student ID', fontsize=12)
    axes[1].set_ylabel('Engagement Score', fontsize=12)
    axes[1].set_title('Engagement Score by Student', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'engagement_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved distribution plot to {output_path}")
    plt.close()


def plot_feature_correlation(data_csv: str, output_dir: str):
    """Plot correlation heatmap of features."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return
    
    df = pd.read_csv(data_csv)
    
    # Select numeric columns
    numeric_cols = ['engagement_score', 'gaze_forward_ratio', 'eye_openness_mean', 
                   'head_pitch_std', 'movement_mean']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        logger.warning("Not enough numeric columns for correlation plot")
        return
    
    corr_matrix = df[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
               square=True, linewidths=1, ax=ax, fmt='.2f')
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved correlation plot to {output_path}")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Plot confusion matrix."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return
    
    from src.evaluation.metrics import confusion_matrix_metrics
    
    metrics = confusion_matrix_metrics(y_true, y_pred, threshold=0.5)
    cm = np.array([[metrics['TN'], metrics['FP']], 
                   [metrics['FN'], metrics['TP']]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Not Engaged', 'Engaged'],
               yticklabels=['Not Engaged', 'Engaged'], ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved confusion matrix to {output_path}")
    plt.close()


def generate_summary_statistics(data_csv: str, output_dir: str):
    """Generate and save summary statistics."""
    df = pd.read_csv(data_csv)
    
    summary = {
        'Total Samples': len(df),
        'Total Students': df['student_id'].nunique(),
        'Mean Engagement': df['engagement_score'].mean(),
        'Std Engagement': df['engagement_score'].std(),
        'Min Engagement': df['engagement_score'].min(),
        'Max Engagement': df['engagement_score'].max(),
        'Median Engagement': df['engagement_score'].median()
    }
    
    summary_df = pd.DataFrame([summary])
    output_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Saved summary statistics to {output_path}")
    
    # Per-student statistics
    student_stats = df.groupby('student_id').agg({
        'engagement_score': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    student_stats.columns = ['_'.join(col).strip() for col in student_stats.columns.values]
    
    student_output = os.path.join(output_dir, 'per_student_statistics.csv')
    student_stats.to_csv(student_output)
    logger.info(f"Saved per-student statistics to {student_output}")


def main():
    parser = argparse.ArgumentParser(description='Visualize student engagement results')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to engagement data CSV file')
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                       help='Directory to save plots')
    parser.add_argument('--plots', type=str, nargs='+', 
                       default=['timeline', 'distribution', 'correlation', 'summary'],
                       help='Which plots to generate')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Generating visualizations from {args.data}")
    
    if 'timeline' in args.plots:
        plot_engagement_over_time(args.data, args.output_dir)
    
    if 'distribution' in args.plots:
        plot_engagement_distribution(args.data, args.output_dir)
    
    if 'correlation' in args.plots:
        plot_feature_correlation(args.data, args.output_dir)
    
    if 'summary' in args.plots:
        generate_summary_statistics(args.data, args.output_dir)
    
    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
