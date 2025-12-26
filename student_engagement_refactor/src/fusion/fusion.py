import numpy as np
from typing import Dict, Any

def normalize(x, minv, maxv):
    """Normalize value to [0, 1] range."""
    if x is None:
        return 0.0
    return float((x - minv) / (maxv - minv + 1e-6))


def simple_engagement_score(features: Dict[str, Any]) -> float:
    """
    Compute engagement score from visual features using weighted fusion.
    
    Args:
        features: Dictionary containing:
            - gaze: Gaze direction ('Forward', 'Left', 'Right')
            - eye_openness: Eye aspect ratio (higher = more open)
            - mouth_open: Mouth aspect ratio
            - head_pitch: Head pitch angle
            - movement: Movement metric
    
    Returns:
        Engagement score in [0, 1] range
    """
    score = 0.0
    weights = {
        "gaze": 0.4,       # Most important: looking forward
        "eye": 0.25,       # Eye openness indicates alertness
        "mouth": 0.05,     # Talking/yawning detection
        "head": 0.2,       # Head pose alignment
        "movement": 0.1    # Restlessness indicator
    }
    
    # Gaze direction scoring
    gaze = features.get('gaze')
    if gaze == "Forward":
        score += weights['gaze'] * 1.0
    elif gaze in ["Left", "Right"]:
        score += weights['gaze'] * 0.2
    else:
        score += weights['gaze'] * 0.5  # Unknown/partial
    
    # Eye openness (normalized)
    eye = features.get('eye_openness', 0.0)
    if eye is not None and eye > 0:
        score += weights['eye'] * normalize(eye, 0.0, 0.06)
    
    # Mouth openness (penalize yawning/talking excessively)
    mouth = features.get('mouth_open', 0.0)
    if mouth is not None:
        mouth_score = 0.0 if mouth > 0.05 else 1.0
        score += weights['mouth'] * mouth_score
    
    # Head pitch (penalize extreme angles)
    hp = features.get('head_pitch', 0.0)
    if hp is not None:
        head_score = 1 - min(abs(hp), 0.2) / 0.2
        score += weights['head'] * head_score
    
    # Movement (penalize excessive restlessness)
    movement = features.get('movement', 0.0)
    if movement is not None:
        # Low movement = engaged, high movement = distracted
        movement_score = 1.0 - normalize(movement, 0.0, 50.0)
        score += weights['movement'] * max(0.0, movement_score)
    
    return max(0.0, min(1.0, score))


def compute_engagement_score(features: Dict[str, Any], method: str = 'weighted') -> float:
    """
    Compute engagement score using specified method.
    
    Args:
        features: Feature dictionary
        method: 'weighted' (default), 'threshold', or 'ml' (placeholder for ML models)
    
    Returns:
        Engagement score [0, 1]
    """
    if method == 'weighted':
        return simple_engagement_score(features)
    elif method == 'threshold':
        # Binary threshold-based classification
        engaged_count = 0
        total_features = 0
        
        if features.get('gaze') == "Forward":
            engaged_count += 1
        total_features += 1
        
        if features.get('eye_openness', 0) > 0.03:
            engaged_count += 1
        total_features += 1
        
        if features.get('mouth_open', 0) < 0.05:
            engaged_count += 1
        total_features += 1
        
        return engaged_count / total_features if total_features > 0 else 0.5
    else:
        # Default to weighted
        return simple_engagement_score(features)
