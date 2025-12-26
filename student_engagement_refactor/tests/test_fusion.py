"""Unit tests for fusion module."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.fusion import simple_engagement_score, compute_engagement_score, normalize


def test_normalize():
    """Test normalization function."""
    assert normalize(5, 0, 10) == 0.5
    assert normalize(0, 0, 10) == 0.0
    assert normalize(10, 0, 10) == 1.0
    assert normalize(None, 0, 10) == 0.0


def test_simple_engagement_score_forward_gaze():
    """Test engagement score with forward gaze."""
    features = {
        'gaze': 'Forward',
        'eye_openness': 0.04,
        'mouth_open': 0.02,
        'head_pitch': 0.0,
        'movement': 5.0
    }
    
    score = simple_engagement_score(features)
    
    assert 0 <= score <= 1
    assert score > 0.5  # Forward gaze should give high score


def test_simple_engagement_score_left_gaze():
    """Test engagement score with left gaze."""
    features = {
        'gaze': 'Left',
        'eye_openness': 0.04,
        'mouth_open': 0.02,
        'head_pitch': 0.0,
        'movement': 5.0
    }
    
    score = simple_engagement_score(features)
    
    assert 0 <= score <= 1
    assert score < 0.7  # Non-forward gaze should give lower score


def test_simple_engagement_score_empty_features():
    """Test engagement score with minimal features."""
    features = {}
    
    score = simple_engagement_score(features)
    
    assert 0 <= score <= 1


def test_simple_engagement_score_all_none():
    """Test engagement score when all features are None."""
    features = {
        'gaze': None,
        'eye_openness': None,
        'mouth_open': None,
        'head_pitch': None,
        'movement': None
    }
    
    score = simple_engagement_score(features)
    
    assert 0 <= score <= 1


def test_simple_engagement_score_high_engagement():
    """Test high engagement scenario."""
    features = {
        'gaze': 'Forward',
        'eye_openness': 0.05,  # Wide open eyes
        'mouth_open': 0.01,    # Mouth closed
        'head_pitch': 0.0,     # Upright
        'movement': 2.0        # Minimal movement
    }
    
    score = simple_engagement_score(features)
    
    assert score > 0.7  # Should be highly engaged


def test_simple_engagement_score_low_engagement():
    """Test low engagement scenario."""
    features = {
        'gaze': 'Left',
        'eye_openness': 0.01,  # Eyes nearly closed
        'mouth_open': 0.08,    # Yawning
        'head_pitch': 0.3,     # Head down
        'movement': 50.0       # High movement
    }
    
    score = simple_engagement_score(features)
    
    assert score < 0.5  # Should be disengaged


def test_compute_engagement_score_weighted():
    """Test compute_engagement_score with weighted method."""
    features = {
        'gaze': 'Forward',
        'eye_openness': 0.04,
        'mouth_open': 0.02,
        'head_pitch': 0.0,
        'movement': 5.0
    }
    
    score = compute_engagement_score(features, method='weighted')
    
    assert 0 <= score <= 1


def test_compute_engagement_score_threshold():
    """Test compute_engagement_score with threshold method."""
    features = {
        'gaze': 'Forward',
        'eye_openness': 0.04,
        'mouth_open': 0.02,
        'head_pitch': 0.0,
        'movement': 5.0
    }
    
    score = compute_engagement_score(features, method='threshold')
    
    assert 0 <= score <= 1


def test_engagement_score_consistency():
    """Test that same features give same score."""
    features = {
        'gaze': 'Forward',
        'eye_openness': 0.04,
        'mouth_open': 0.02,
        'head_pitch': 0.05,
        'movement': 10.0
    }
    
    score1 = simple_engagement_score(features)
    score2 = simple_engagement_score(features)
    
    assert score1 == score2


def test_engagement_score_bounds():
    """Test that scores are always within [0, 1]."""
    test_cases = [
        {'gaze': 'Forward', 'eye_openness': 100, 'mouth_open': 100, 'head_pitch': 100, 'movement': 1000},
        {'gaze': 'Left', 'eye_openness': -100, 'mouth_open': -100, 'head_pitch': -100, 'movement': -1000},
        {'gaze': None, 'eye_openness': None, 'mouth_open': None, 'head_pitch': None, 'movement': None},
    ]
    
    for features in test_cases:
        score = simple_engagement_score(features)
        assert 0 <= score <= 1, f"Score {score} out of bounds for features {features}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
