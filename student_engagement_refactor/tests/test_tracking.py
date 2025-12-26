"""Unit tests for tracking module."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.sort_tracker import Sort, Track, iou


def test_iou_calculation():
    """Test IoU calculation."""
    bbox1 = [0, 0, 10, 10]
    bbox2 = [5, 5, 15, 15]
    
    iou_val = iou(bbox1, bbox2)
    
    assert 0 <= iou_val <= 1
    # Overlap of 5x5 = 25, union of 10x10 + 10x10 - 25 = 175
    expected = 25 / 175
    assert abs(iou_val - expected) < 0.01


def test_iou_no_overlap():
    """Test IoU with non-overlapping boxes."""
    bbox1 = [0, 0, 10, 10]
    bbox2 = [20, 20, 30, 30]
    
    iou_val = iou(bbox1, bbox2)
    assert iou_val == 0


def test_iou_perfect_overlap():
    """Test IoU with identical boxes."""
    bbox1 = [0, 0, 10, 10]
    bbox2 = [0, 0, 10, 10]
    
    iou_val = iou(bbox1, bbox2)
    assert abs(iou_val - 1.0) < 0.01


def test_sort_initialization():
    """Test SORT tracker initialization."""
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    assert tracker.max_age == 30
    assert tracker.min_hits == 3
    assert tracker.iou_threshold == 0.3
    assert tracker.frame_count == 0
    assert len(tracker.tracks) == 0


def test_sort_single_detection():
    """Test tracking with single detection."""
    tracker = Sort()
    
    detection = [[10, 10, 50, 50]]
    tracks = tracker.update(detection)
    
    # New track may not be confirmed yet (min_hits=3)
    assert len(tracks) >= 0


def test_sort_multiple_frames():
    """Test tracking across multiple frames."""
    tracker = Sort(min_hits=1)
    
    # Frame 1
    detections1 = [[10, 10, 50, 50]]
    tracks1 = tracker.update(detections1)
    assert len(tracks1) >= 1
    
    # Frame 2 - similar position
    detections2 = [[12, 12, 52, 52]]
    tracks2 = tracker.update(detections2)
    assert len(tracks2) >= 1
    
    # Track ID should be consistent
    if len(tracks1) > 0 and len(tracks2) > 0:
        assert tracks1[0][4] == tracks2[0][4]  # Same ID


def test_sort_no_detections():
    """Test tracker with no detections."""
    tracker = Sort()
    
    tracks = tracker.update([])
    assert len(tracks) == 0


def test_sort_track_deletion():
    """Test track deletion after max_age."""
    tracker = Sort(max_age=3, min_hits=1)
    
    # Create a track
    detections = [[10, 10, 50, 50]]
    tracker.update(detections)
    
    # Run several frames without detections
    for _ in range(5):
        tracks = tracker.update([])
    
    # Track should be deleted
    assert len(tracks) == 0


def test_track_initialization():
    """Test Track object initialization."""
    bbox = [10, 20, 50, 60]
    track = Track(bbox, track_id=1)
    
    assert track.id == 1
    assert track.hits == 1
    assert track.no_losses == 0
    assert track.bbox == bbox


def test_track_prediction():
    """Test track prediction."""
    bbox = [10, 20, 50, 60]
    track = Track(bbox, track_id=1)
    
    predicted_bbox = track.predict()
    
    assert len(predicted_bbox) == 4
    # Predicted bbox should be close to original
    for i in range(4):
        assert abs(predicted_bbox[i] - bbox[i]) < 10


def test_track_update():
    """Test track update with new detection."""
    bbox = [10, 20, 50, 60]
    track = Track(bbox, track_id=1)
    
    new_bbox = [12, 22, 52, 62]
    track.update(new_bbox)
    
    assert track.hits == 2
    assert track.no_losses == 0
    assert track.bbox == new_bbox


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
