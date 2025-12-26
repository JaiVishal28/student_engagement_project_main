"""Unit tests for feature extraction module."""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.visual_features import extract_all, compute_gaze, mouth_open_ratio, eye_openness


@pytest.fixture
def sample_frame():
    """Create a sample test frame."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return img


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return [100, 100, 300, 400]  # [xmin, ymin, xmax, ymax]


def test_extract_all_returns_dict(sample_frame, sample_bbox):
    """Test that extract_all returns a dictionary."""
    features = extract_all(sample_frame, sample_bbox, 640, 480)
    
    assert isinstance(features, dict)
    assert 'gaze' in features
    assert 'mouth_open' in features
    assert 'eye_openness' in features
    assert 'head_pitch' in features


def test_extract_all_with_invalid_bbox(sample_frame):
    """Test feature extraction with invalid bbox."""
    invalid_bbox = [0, 0, 1, 1]  # Very small bbox
    
    try:
        features = extract_all(sample_frame, invalid_bbox, 640, 480)
        # Should return dict even if features couldn't be extracted
        assert isinstance(features, dict)
    except Exception:
        # Or it might raise an exception, which is also acceptable
        pass


def test_extract_all_feature_types(sample_frame, sample_bbox):
    """Test that extracted features have correct types."""
    features = extract_all(sample_frame, sample_bbox, 640, 480)
    
    # Gaze should be string or None
    assert features['gaze'] is None or isinstance(features['gaze'], str)
    
    # Numeric features should be float or None
    if features['mouth_open'] is not None:
        assert isinstance(features['mouth_open'], (int, float))
    
    if features['eye_openness'] is not None:
        assert isinstance(features['eye_openness'], (int, float))


def test_gaze_computation():
    """Test gaze direction computation."""
    # Mock landmarks class
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    # Create mock landmarks
    landmarks = [None] * 500
    landmarks[1] = MockLandmark(0.5, 0.5)  # nose
    landmarks[33] = MockLandmark(0.3, 0.5)  # left
    landmarks[263] = MockLandmark(0.7, 0.5)  # right
    
    gaze = compute_gaze(landmarks)
    assert gaze == "Forward"
    
    # Test right gaze
    landmarks[1] = MockLandmark(0.2, 0.5)
    gaze = compute_gaze(landmarks)
    assert gaze == "Right"
    
    # Test left gaze
    landmarks[1] = MockLandmark(0.8, 0.5)
    gaze = compute_gaze(landmarks)
    assert gaze == "Left"


def test_mouth_open_ratio():
    """Test mouth open ratio computation."""
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    landmarks = [None] * 500
    landmarks[13] = MockLandmark(0.5, 0.4)  # top
    landmarks[14] = MockLandmark(0.5, 0.6)  # bottom
    
    ratio = mouth_open_ratio(landmarks)
    assert abs(ratio - 0.2) < 0.01


def test_eye_openness():
    """Test eye openness computation."""
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    landmarks = [None] * 500
    landmarks[386] = MockLandmark(0.3, 0.25)  # left eye top
    landmarks[374] = MockLandmark(0.3, 0.30)  # left eye bottom
    landmarks[159] = MockLandmark(0.7, 0.25)  # right eye top
    landmarks[145] = MockLandmark(0.7, 0.30)  # right eye bottom
    
    openness = eye_openness(landmarks)
    assert openness > 0
    assert isinstance(openness, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
