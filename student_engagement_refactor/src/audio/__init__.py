"""
Audio processing module for classroom engagement detection.
Includes audio capture, VAD, and feature extraction.
"""

from .audio_capture import AudioCapture
from .vad_detector import VADDetector
from .audio_features import AudioFeatureExtractor
from .speaker_enrollment import SpeakerEnrollment

__all__ = ['AudioCapture', 'VADDetector', 'AudioFeatureExtractor', 'SpeakerEnrollment']
