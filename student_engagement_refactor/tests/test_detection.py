"""Unit tests for detection module."""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.yolov_wrapper import YoloDetector


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return img


@pytest.fixture
def detector():
    """Create detector instance (requires model weights)."""
    try:
        return YoloDetector("models/weights/yolov8s.pt", device="cpu")
    except Exception as e:
        pytest.skip(f"Model weights not available: {e}")


def test_detector_initialization(detector):
    """Test detector initializes correctly."""
    assert detector is not None
    assert detector.model is not None


def test_detection_output_format(detector, sample_image):
    """Test that detection returns correct format."""
    detections = detector.detect(sample_image)
    
    assert isinstance(detections, list)
    
    if len(detections) > 0:
        det = detections[0]
        assert 'xmin' in det
        assert 'ymin' in det
        assert 'xmax' in det
        assert 'ymax' in det
        assert 'conf' in det
        assert 'cls' in det
        
        # Check value ranges
        assert det['xmin'] >= 0
        assert det['ymin'] >= 0
        assert det['xmax'] >= det['xmin']
        assert det['ymax'] >= det['ymin']
        assert 0 <= det['conf'] <= 1


def test_detection_empty_image(detector):
    """Test detection on empty image."""
    empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(empty_img)
    
    # Should return list (possibly empty)
    assert isinstance(detections, list)


def test_detection_bbox_validity(detector, sample_image):
    """Test that bounding boxes are within image bounds."""
    h, w = sample_image.shape[:2]
    detections = detector.detect(sample_image)
    
    for det in detections:
        assert 0 <= det['xmin'] < w
        assert 0 <= det['ymin'] < h
        assert 0 < det['xmax'] <= w
        assert 0 < det['ymax'] <= h


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
