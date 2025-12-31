"""
Evaluation metrics for student engagement detection system.
Includes precision, recall, F1-score, accuracy, and confusion matrix computation.
"""
import numpy as np
from typing import Dict, List, Tuple
from src.logging_utils import get_logger

logger = get_logger("Metrics")


def confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute confusion matrix-based metrics.
    
    Args:
        y_true: Ground truth labels (0 or 1, or continuous [0,1] if using threshold)
        y_pred: Predicted scores (continuous [0,1])
        threshold: Classification threshold for binary prediction
    
    Returns:
        Dictionary with TP, FP, TN, FN, precision, recall, F1, accuracy
    """
    if len(y_true) == 0:
        logger.warning("Empty arrays provided to confusion_matrix_metrics")
        return {
            'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0
        }
    
    # Convert to binary if needed
    y_true_bin = (np.array(y_true) >= threshold).astype(int)
    y_pred_bin = (np.array(y_pred) >= threshold).astype(int)
    
    TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    FP = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    TN = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    FN = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-10)
    
    return {
        'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth engagement scores [0,1]
        y_pred: Predicted engagement scores [0,1]
    
    Returns:
        Dictionary with MAE, RMSE, correlation, and classification metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty arrays provided to compute_metrics")
        return {
            'mae': 0.0, 'rmse': 0.0, 'correlation': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0
        }
    
    # Regression metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    
    # Classification metrics (using 0.5 threshold)
    cm_metrics = confusion_matrix_metrics(y_true, y_pred, threshold=0.5)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(correlation),
        'precision': cm_metrics['precision'],
        'recall': cm_metrics['recall'],
        'f1': cm_metrics['f1'],
        'accuracy': cm_metrics['accuracy']
    }


def mean_average_precision(detections: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> float:
    """
    Compute mean average precision for detection task.
    
    Args:
        detections: List of detection dicts with 'bbox' and 'conf'
        ground_truths: List of ground truth dicts with 'bbox'
        iou_threshold: IoU threshold for matching
    
    Returns:
        mAP score
    """
    if len(ground_truths) == 0:
        return 0.0
    
    from src.tracking.sort_tracker import iou
    
    # Sort detections by confidence
    sorted_dets = sorted(detections, key=lambda x: x.get('conf', 0), reverse=True)
    
    tp = np.zeros(len(sorted_dets))
    fp = np.zeros(len(sorted_dets))
    matched_gt = set()
    
    for i, det in enumerate(sorted_dets):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truths):
            if j in matched_gt:
                continue
            current_iou = iou(det['bbox'], gt['bbox'])
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return float(ap)
