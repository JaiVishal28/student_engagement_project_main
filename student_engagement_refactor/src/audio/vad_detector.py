import torch
import numpy as np
from src.logging_utils import get_logger

logger = get_logger("VADDetector")


class VADDetector:
    """
    Voice Activity Detection using Silero VAD.
    Efficient, edge-friendly speech detection.
    """
    
    def __init__(self, threshold=0.5, sample_rate=16000):
        """
        Initialize VAD detector.
        
        Args:
            threshold: Speech probability threshold (0-1)
            sample_rate: Audio sample rate (must be 8000 or 16000)
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = None
        self.utils = None
        
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model."""
        try:
            # Load Silero VAD model from torch hub
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Extract utility functions
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = self.utils
            
            self.model.eval()
            logger.info(f"Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            logger.warning("VAD will use fallback energy-based detection")
            self.model = None
    
    def detect_speech(self, audio_chunk, return_confidence=False):
        """
        Detect if audio chunk contains speech.
        
        Args:
            audio_chunk: Audio data as numpy array (float32, [-1, 1])
            return_confidence: If True, return speech probability
            
        Returns:
            Boolean (speech detected) or float (speech probability)
        """
        if self.model is None:
            # Fallback to energy-based detection
            return self._energy_based_vad(audio_chunk, return_confidence)
        
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()
            
            # Get speech probability
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            if return_confidence:
                return speech_prob
            else:
                return speech_prob > self.threshold
                
        except Exception as e:
            logger.debug(f"VAD detection error: {e}")
            return self._energy_based_vad(audio_chunk, return_confidence)
    
    def _energy_based_vad(self, audio_chunk, return_confidence=False):
        """
        Fallback energy-based VAD.
        
        Args:
            audio_chunk: Audio data as numpy array
            return_confidence: If True, return normalized energy
            
        Returns:
            Boolean or float
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Normalize energy (typical speech is 0.01 - 0.3)
        normalized_energy = min(energy / 0.3, 1.0)
        
        if return_confidence:
            return normalized_energy
        else:
            return energy > 0.02  # Threshold for speech
    
    def get_speech_segments(self, audio_data):
        """
        Get timestamps of speech segments in audio.
        
        Args:
            audio_data: Long audio array
            
        Returns:
            List of speech segment timestamps
        """
        if self.model is None:
            return []
        
        try:
            audio_tensor = torch.from_numpy(audio_data).float()
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold
            )
            return speech_timestamps
        except Exception as e:
            logger.debug(f"Error getting speech segments: {e}")
            return []
    
    def count_speakers_estimate(self, audio_chunk):
        """
        Rough estimate of number of speakers based on energy variation.
        
        Args:
            audio_chunk: Audio data
            
        Returns:
            Estimated speaker count (1, 2, or 3+)
        """
        # Split into segments
        segment_size = len(audio_chunk) // 4
        if segment_size < 100:
            return 1
        
        segments = [
            audio_chunk[i:i+segment_size]
            for i in range(0, len(audio_chunk), segment_size)
            if len(audio_chunk[i:i+segment_size]) == segment_size
        ]
        
        # Calculate energy variance across segments
        energies = [np.sqrt(np.mean(seg ** 2)) for seg in segments]
        
        if len(energies) < 2:
            return 1
        
        energy_variance = np.var(energies)
        mean_energy = np.mean(energies)
        max_energy = np.max(energies)
        
        # EXTREMELY SENSITIVE: Any variance or high energy suggests multiple speakers
        if energy_variance > 0.00001 and mean_energy > 0.01:  # 10x more sensitive
            return 2  # Multiple speakers detected
        elif max_energy > 0.05:  # High energy = likely multiple sources
            return 2
        elif mean_energy > 0.008:
            return 1  # Single speaker
        else:
            return 0  # Silence
