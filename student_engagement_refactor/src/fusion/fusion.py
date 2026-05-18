import numpy as np
from typing import Dict, Any, Optional

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
    contributions = {}  # Track individual component contributions
    
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
        gaze_score = 1.0
    elif gaze in ["Left", "Right"]:
        gaze_score = 0.2
    else:
        gaze_score = 0.5  # Unknown/partial
    contributions['gaze'] = weights['gaze'] * gaze_score
    score += contributions['gaze']
    
    # Eye openness (normalized)
    eye = features.get('eye_openness', 0.0)
    if eye is not None and eye > 0:
        eye_score = normalize(eye, 0.0, 0.06)
    else:
        eye_score = 0.0
    contributions['eye'] = weights['eye'] * eye_score
    score += contributions['eye']
    
    # Mouth openness (penalize yawning/talking excessively)
    mouth = features.get('mouth_open', 0.0)
    if mouth is not None:
        mouth_score = 0.0 if mouth > 0.05 else 1.0
    else:
        mouth_score = 0.5
    contributions['mouth'] = weights['mouth'] * mouth_score
    score += contributions['mouth']
    
    # Head pitch (penalize extreme angles)
    hp = features.get('head_pitch', 0.0)
    if hp is not None:
        head_score = 1 - min(abs(hp), 0.2) / 0.2
    else:
        head_score = 0.5
    contributions['head'] = weights['head'] * head_score
    score += contributions['head']
    
    # Movement (penalize excessive restlessness)
    movement = features.get('movement', 0.0)
    if movement is not None:
        # Low movement = engaged, high movement = distracted
        movement_score = 1.0 - normalize(movement, 0.0, 50.0)
    else:
        movement_score = 0.5
    contributions['movement'] = weights['movement'] * max(0.0, movement_score)
    score += contributions['movement']
    
    # Store contributions in features dict for debugging
    features['_score_breakdown'] = contributions
    
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


def multimodal_engagement_score(visual_features: Dict[str, Any], 
                                 audio_features: Optional[Dict[str, Any]] = None,
                                 visual_weight: float = 0.65,
                                 audio_weight: float = 0.35,
                                 verbose: bool = False) -> float:
    """
    Compute multimodal engagement score from visual and audio features.
    
    Args:
        visual_features: Visual feature dictionary (gaze, eyes, etc.)
        audio_features: Audio feature dictionary (noise, speech, etc.)
        visual_weight: Weight for visual features (default 0.65)
        audio_weight: Weight for audio features (default 0.35)
        verbose: Print detailed score breakdown
    
    Returns:
        Combined engagement score [0, 1]
    """
    # Get visual engagement score
    visual_score = simple_engagement_score(visual_features)
    
    # If no audio features, return visual only
    if audio_features is None or not audio_features:
        if verbose:
            breakdown = visual_features.get('_score_breakdown', {})
            print(f"\n📊 Engagement Score Breakdown (Visual Only):")
            print(f"  Gaze (40%):     {breakdown.get('gaze', 0):.3f}")
            print(f"  Eyes (25%):     {breakdown.get('eye', 0):.3f}")
            print(f"  Mouth (5%):     {breakdown.get('mouth', 0):.3f}")
            print(f"  Head (20%):     {breakdown.get('head', 0):.3f}")
            print(f"  Movement (10%): {breakdown.get('movement', 0):.3f}")
            print(f"  ──────────────")
            print(f"  TOTAL:          {visual_score:.3f}")
        return visual_score
    
    # Get audio engagement score
    audio_score = audio_features.get('audio_engagement_score', 0.7)
    
    # Weighted fusion
    combined_score = (visual_weight * visual_score) + (audio_weight * audio_score)
    
    # Track modulation factors
    modulations = []
    
    # Apply audio modulation factors
    # Penalize if multiple speakers detected (side conversations)
    if audio_features.get('multiple_speakers_detected', False):
        combined_score *= 0.85  # Reduce by 15%
        modulations.append("Multiple speakers (-15%)")
    
    # Penalize excessive noise
    if audio_features.get('excessive_noise', False):
        combined_score *= 0.90  # Reduce by 10%
        modulations.append("Excessive noise (-10%)")
    
    # Bonus for attentive environment (low noise, single speaker)
    if audio_features.get('background_noise_level') == 'low' and \
       audio_features.get('speaker_count', 1) == 1:
        combined_score = min(1.0, combined_score * 1.1)  # Boost by 10%
        modulations.append("Quiet environment (+10%)")
    
    if verbose:
        breakdown = visual_features.get('_score_breakdown', {})
        print(f"\n📊 Engagement Score Breakdown (Multimodal):")
        print(f"  VISUAL (65% weight):")
        print(f"    Gaze (40%):     {breakdown.get('gaze', 0):.3f}  ← Forward gaze = highly engaged")
        print(f"    Eyes (25%):     {breakdown.get('eye', 0):.3f}  ← Wide eyes = alert & focused")
        print(f"    Mouth (5%):     {breakdown.get('mouth', 0):.3f}  ← Closed mouth = attentive")
        print(f"    Head (20%):     {breakdown.get('head', 0):.3f}  ← Upright head = good posture")
        print(f"    Movement (10%): {breakdown.get('movement', 0):.3f}  ← Minimal movement = calm")
        print(f"    → Visual Score: {visual_score:.3f}")
        print(f"  AUDIO (35% weight):")
        print(f"    Audio Score:    {audio_score:.3f}  ← High = quiet environment")
        print(f"    Student Noise:  {audio_features.get('student_noise_detected', False)}  ← True = students talking")
        print(f"    Noise Level:    {audio_features.get('student_noise_level', 0):.3f}  ← 0=quiet, 1=loud")
        print(f"    Is Teacher:     {audio_features.get('is_teacher_speaking', False)}  ← Similarity above threshold")
        print(f"    Similarity:     {audio_features.get('teacher_similarity', 0):.3f}  ← z-score metric (threshold={audio_features.get('similarity_threshold', 0.65):.2f}), Speaker Count={audio_features.get('speaker_count', 1)}")
        print(f"  ──────────────")
        print(f"  Base Score:     {(visual_weight * visual_score) + (audio_weight * audio_score):.3f}")
        if modulations:
            for mod in modulations:
                print(f"  {mod}")
        print(f"  FINAL SCORE:    {combined_score:.3f}")
        
        # Add interpretation help
        sim_score = audio_features.get('teacher_similarity', 0)
        spk_count = audio_features.get('speaker_count', 1)
        if spk_count > 1:
            print(f"  🔊 MULTIPLE SPEAKERS DETECTED (count={spk_count}) → STUDENT NOISE")
        elif audio_features.get('is_teacher_speaking', False):
            thr = audio_features.get('similarity_threshold', 0.65)
            print(f"  ℹ️  Audio classified as TEACHER (similarity {sim_score:.2f} > {thr:.2f})")
        else:
            thr = audio_features.get('similarity_threshold', 0.65)
            print(f"  ⚠️  Audio classified as STUDENT/OTHER (similarity {sim_score:.2f} < {thr:.2f})")
    
    return max(0.0, min(1.0, combined_score))
