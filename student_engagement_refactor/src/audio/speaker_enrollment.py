import numpy as np
import torch
from collections import deque
from src.logging_utils import get_logger

logger = get_logger("SpeakerEnrollment")


class SpeakerEnrollment:
    """
    Teacher voice enrollment system using spectral profiling.
    Records teacher's voice characteristics during enrollment period,
    then filters it out to detect only student voices and noise.
    """
    
    def __init__(self, enrollment_duration=10.0, similarity_threshold=0.75):
        """
        Initialize speaker enrollment system.
        
        Args:
            enrollment_duration: Seconds to record teacher's voice profile
            similarity_threshold: How similar audio must be to teacher (0-1)
        """
        self.enrollment_duration = enrollment_duration
        self.similarity_threshold = similarity_threshold
        
        # Teacher voice profile
        self.teacher_profile = None
        self.teacher_spectral_features = []
        self.teacher_mfcc_profile = None
        self.is_enrolled = False
        
        # Rolling statistics
        self.enrollment_samples = []
        self.max_enrollment_samples = 20  # ~10 seconds at 0.5s chunks
        
        logger.info(f"Speaker enrollment initialized: {enrollment_duration}s enrollment period")
    
    def add_enrollment_sample(self, audio_chunk, sample_rate=16000):
        """
        Add audio sample during teacher enrollment period.
        Should be called only when teacher is speaking alone.
        
        Args:
            audio_chunk: Audio data (numpy array)
            sample_rate: Sample rate of audio
        """
        if self.is_enrolled:
            logger.warning("Already enrolled, ignoring new sample")
            return
        
        # Extract spectral features
        features = self._extract_spectral_features(audio_chunk, sample_rate)
        self.enrollment_samples.append(features)
        
        logger.debug(f"Enrollment sample {len(self.enrollment_samples)}/{self.max_enrollment_samples} added")
        
        # Check if we have enough samples
        if len(self.enrollment_samples) >= self.max_enrollment_samples:
            self._finalize_enrollment()
    
    def _extract_spectral_features(self, audio_chunk, sample_rate):
        """
        Extract spectral features from audio for voice profiling.
        
        Returns:
            Dictionary with spectral characteristics
        """
        # FFT for frequency analysis
        fft = np.fft.rfft(audio_chunk)
        magnitude = np.abs(fft)
        
        # Spectral centroid (where is the "center of mass" of the spectrum)
        freqs = np.fft.rfftfreq(len(audio_chunk), 1/sample_rate)
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        # Spectral flux (change in magnitude spectrum)
        spectral_flux = np.sqrt(np.sum(np.diff(magnitude)**2)) / len(magnitude)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / 2
        zcr = zero_crossings / len(audio_chunk)
        
        # Energy in different frequency bands
        # Low: 0-300Hz, Mid: 300-2000Hz, High: 2000-8000Hz
        low_band = magnitude[(freqs >= 0) & (freqs < 300)]
        mid_band = magnitude[(freqs >= 300) & (freqs < 2000)]
        high_band = magnitude[(freqs >= 2000) & (freqs < 8000)]
        
        low_energy = np.sum(low_band**2)
        mid_energy = np.sum(mid_band**2)
        high_energy = np.sum(high_band**2)
        total_energy = low_energy + mid_energy + high_energy + 1e-10
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flux': spectral_flux,
            'zcr': zcr,
            'low_energy_ratio': low_energy / total_energy,
            'mid_energy_ratio': mid_energy / total_energy,
            'high_energy_ratio': high_energy / total_energy,
            'total_energy': total_energy,
            'magnitude_spectrum': magnitude
        }
    
    def _finalize_enrollment(self):
        """Finalize teacher voice profile from collected samples."""
        if not self.enrollment_samples:
            logger.error("No enrollment samples collected")
            return
        
        # Average all features
        self.teacher_profile = {
            'spectral_centroid': np.mean([s['spectral_centroid'] for s in self.enrollment_samples]),
            'spectral_rolloff': np.mean([s['spectral_rolloff'] for s in self.enrollment_samples]),
            'spectral_flux': np.mean([s['spectral_flux'] for s in self.enrollment_samples]),
            'zcr': np.mean([s['zcr'] for s in self.enrollment_samples]),
            'low_energy_ratio': np.mean([s['low_energy_ratio'] for s in self.enrollment_samples]),
            'mid_energy_ratio': np.mean([s['mid_energy_ratio'] for s in self.enrollment_samples]),
            'high_energy_ratio': np.mean([s['high_energy_ratio'] for s in self.enrollment_samples]),
            'total_energy_mean': np.mean([s['total_energy'] for s in self.enrollment_samples]),
            'total_energy_std': np.std([s['total_energy'] for s in self.enrollment_samples]),
        }
        
        # Calculate standard deviations for tolerance
        self.teacher_profile['spectral_centroid_std'] = np.std([s['spectral_centroid'] for s in self.enrollment_samples])
        self.teacher_profile['spectral_rolloff_std'] = np.std([s['spectral_rolloff'] for s in self.enrollment_samples])
        
        self.is_enrolled = True
        logger.info(f"Teacher enrollment complete! Profile created from {len(self.enrollment_samples)} samples")
        logger.info(f"  Spectral Centroid: {self.teacher_profile['spectral_centroid']:.1f} Hz")
        logger.info(f"  Spectral Rolloff: {self.teacher_profile['spectral_rolloff']:.1f} Hz")
        logger.info(f"  ZCR: {self.teacher_profile['zcr']:.4f}")
    
    def is_teacher_speaking(self, audio_chunk, sample_rate=16000):
        """
        Determine if current audio matches teacher's voice profile.
        
        Args:
            audio_chunk: Audio data to analyze
            sample_rate: Sample rate
            
        Returns:
            (is_teacher, similarity_score)
            - is_teacher: Boolean, True if this sounds like teacher
            - similarity_score: 0-1, how similar to teacher profile
        """
        if not self.is_enrolled:
            # During enrollment, assume it's teacher
            return True, 1.0
        
        # Extract features from current audio
        current_features = self._extract_spectral_features(audio_chunk, sample_rate)
        
        # Calculate similarity score
        similarity = self._calculate_similarity(current_features)
        
        is_teacher = similarity > self.similarity_threshold
        
        return is_teacher, similarity
    
    def _calculate_similarity(self, current_features):
        """
        Calculate similarity between current audio and teacher profile.
        
        Returns:
            Similarity score (0-1), 1 = identical to teacher
        """
        if not self.is_enrolled or not self.teacher_profile:
            return 0.0
        
        # Compare spectral centroid (weighted 30%)
        centroid_diff = abs(current_features['spectral_centroid'] - self.teacher_profile['spectral_centroid'])
        centroid_tolerance = max(self.teacher_profile['spectral_centroid_std'], 100)  # At least 100Hz tolerance
        centroid_similarity = max(0, 1 - (centroid_diff / centroid_tolerance))
        
        # Compare spectral rolloff (weighted 25%)
        rolloff_diff = abs(current_features['spectral_rolloff'] - self.teacher_profile['spectral_rolloff'])
        rolloff_tolerance = max(self.teacher_profile['spectral_rolloff_std'], 200)
        rolloff_similarity = max(0, 1 - (rolloff_diff / rolloff_tolerance))
        
        # Compare ZCR (weighted 15%)
        zcr_diff = abs(current_features['zcr'] - self.teacher_profile['zcr'])
        zcr_similarity = max(0, 1 - (zcr_diff / 0.1))  # 0.1 is typical ZCR range
        
        # Compare frequency band ratios (weighted 30%)
        band_diff = (
            abs(current_features['low_energy_ratio'] - self.teacher_profile['low_energy_ratio']) +
            abs(current_features['mid_energy_ratio'] - self.teacher_profile['mid_energy_ratio']) +
            abs(current_features['high_energy_ratio'] - self.teacher_profile['high_energy_ratio'])
        ) / 3
        band_similarity = max(0, 1 - band_diff * 2)  # Scale to 0-1
        
        # Weighted average
        total_similarity = (
            0.30 * centroid_similarity +
            0.25 * rolloff_similarity +
            0.15 * zcr_similarity +
            0.30 * band_similarity
        )
        
        return total_similarity
    
    def get_student_noise_level(self, audio_chunk, sample_rate=16000):
        """
        Analyze audio to determine student noise level (excluding teacher).
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary with student noise analysis:
            - student_noise_detected: Boolean
            - noise_level: 0-1 scale
            - is_teacher: Boolean
            - confidence: 0-1
        """
        if not self.is_enrolled:
            return {
                'student_noise_detected': False,
                'noise_level': 0.0,
                'is_teacher': True,
                'confidence': 0.0,
                'status': 'enrolling'
            }
        
        # Check if this is teacher speaking
        is_teacher, teacher_similarity = self.is_teacher_speaking(audio_chunk, sample_rate)
        
        # Extract features
        features = self._extract_spectral_features(audio_chunk, sample_rate)
        current_energy = features['total_energy']
        
        # Calculate student noise level
        if is_teacher:
            # This sounds like teacher - minimal student noise
            # But check if energy is HIGHER than expected (students talking over teacher)
            expected_energy = self.teacher_profile['total_energy_mean']
            energy_ratio = current_energy / (expected_energy + 1e-10)
            
            if energy_ratio > 1.5:  # 50% more energy than teacher alone
                # Extra noise on top of teacher
                noise_level = min(1.0, (energy_ratio - 1.0) / 2.0)  # Scale excess energy
                student_noise_detected = noise_level > 0.2
            else:
                noise_level = 0.0
                student_noise_detected = False
        else:
            # This does NOT sound like teacher - likely student voices
            # Map dissimilarity to noise level
            dissimilarity = 1.0 - teacher_similarity
            noise_level = min(1.0, dissimilarity * 1.5)  # Amplify for sensitivity
            student_noise_detected = noise_level > 0.3
        
        return {
            'student_noise_detected': student_noise_detected,
            'noise_level': float(noise_level),
            'is_teacher': is_teacher,
            'teacher_similarity': float(teacher_similarity),
            'confidence': float(abs(teacher_similarity - 0.5) * 2),  # How confident we are
            'status': 'active'
        }
    
    def reset(self):
        """Reset enrollment to start fresh."""
        self.teacher_profile = None
        self.enrollment_samples = []
        self.is_enrolled = False
        logger.info("Speaker enrollment reset")
