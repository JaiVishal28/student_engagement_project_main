import numpy as np
from collections import deque
from src.logging_utils import get_logger

logger = get_logger("AudioFeatures")


class AudioFeatureExtractor:
    """
    Extract audio features for classroom engagement analysis.
    Focuses on detecting student activity vs teacher monologue.
    """
    
    def __init__(self, baseline_duration=5.0, noise_threshold=0.03, sample_rate=16000):
        """
        Initialize audio feature extractor.
        
        Args:
            baseline_duration: Seconds of audio to establish baseline (teacher voice)
            noise_threshold: Energy threshold for background noise
            sample_rate: Audio sample rate (default 16000 Hz)
        """
        self.baseline_duration = baseline_duration
        self.noise_threshold = noise_threshold
        self.sample_rate = sample_rate
        
        # Baseline tracking (first few seconds assumed to be teacher only)
        self.baseline_energy = []
        self.baseline_established = False
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        
        # Rolling window for recent audio analysis
        self.energy_window = deque(maxlen=30)  # Last 30 chunks
        self.speech_window = deque(maxlen=30)
        self.speaker_count_window = deque(maxlen=30)
        
        logger.info("AudioFeatureExtractor initialized")
    
    def update_baseline(self, audio_chunk):
        """
        Update baseline with teacher's voice profile.
        Call this during initial period when only teacher speaks.
        
        Args:
            audio_chunk: Audio data from teacher speaking
        """
        energy = self._calculate_energy(audio_chunk)
        self.baseline_energy.append(energy)
        
        # After enough samples, establish baseline
        if len(self.baseline_energy) >= 10 and not self.baseline_established:
            self.baseline_mean = np.mean(self.baseline_energy)
            self.baseline_std = np.std(self.baseline_energy)
            self.baseline_established = True
            logger.info(f"Baseline established: mean={self.baseline_mean:.4f}, std={self.baseline_std:.4f}")
    
    def extract_features(self, audio_chunk, vad_result=None, speaker_count=1, speaker_enrollment=None):
        """
        Extract engagement features from audio chunk.
        
        Args:
            audio_chunk: Audio data (numpy array, float32)
            vad_result: Speech probability from VAD (0-1)
            speaker_count: Estimated number of speakers
            speaker_enrollment: SpeakerEnrollment object for teacher filtering
            
        Returns:
            Dictionary of audio features
        """
        # Basic audio metrics
        energy = self._calculate_energy(audio_chunk)
        zcr = self._zero_crossing_rate(audio_chunk)
        
        # Update rolling windows
        self.energy_window.append(energy)
        if vad_result is not None:
            self.speech_window.append(vad_result)
        self.speaker_count_window.append(speaker_count)
        
        # Check for student noise using speaker enrollment
        student_noise_info = None
        if speaker_enrollment and speaker_enrollment.is_enrolled:
            student_noise_info = speaker_enrollment.get_student_noise_level(
                audio_chunk, 
                sample_rate=self.sample_rate,
                speaker_count=speaker_count
            )
        
        # Calculate features
        features = {
            # Raw metrics
            'audio_energy': float(energy),
            'zero_crossing_rate': float(zcr),
            'speech_probability': float(vad_result) if vad_result is not None else 0.0,
            'speaker_count': int(speaker_count),
            
            # Student noise detection (NEW - teacher-filtered)
            'student_noise_detected': student_noise_info['student_noise_detected'] if student_noise_info else False,
            'student_noise_level': student_noise_info['noise_level'] if student_noise_info else 0.0,
            'is_teacher_speaking': student_noise_info['is_teacher'] if student_noise_info else True,
            'teacher_similarity': student_noise_info.get('teacher_similarity', 1.0) if student_noise_info else 1.0,
            'similarity_threshold': student_noise_info.get('similarity_threshold', 0.65) if student_noise_info else 0.65,
            
            # Legacy engagement indicators (kept for compatibility)
            'background_noise_level': self._background_noise_level(energy),
            'multiple_speakers_detected': speaker_count > 1,
            'excessive_noise': energy > (self.baseline_mean + 2 * self.baseline_std) if self.baseline_established else False,
            
            # Temporal features (over recent window)
            'avg_energy_recent': float(np.mean(self.energy_window)) if self.energy_window else 0.0,
            'energy_variance': float(np.var(self.energy_window)) if len(self.energy_window) > 1 else 0.0,
            'speech_activity_ratio': float(np.mean([s > 0.5 for s in self.speech_window])) if self.speech_window else 0.0,
        }
        
        # Calculate engagement score from audio (using new method)
        features['audio_engagement_score'] = self._calculate_audio_engagement_v2(features)
        
        return features
    
    def _calculate_energy(self, audio_chunk):
        """Calculate RMS energy of audio."""
        return float(np.sqrt(np.mean(audio_chunk ** 2)))
    
    def _zero_crossing_rate(self, audio_chunk):
        """Calculate zero-crossing rate (indicates pitch/frequency characteristics)."""
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / 2
        return zero_crossings / len(audio_chunk)
    
    def _background_noise_level(self, current_energy):
        """
        Estimate background noise level relative to baseline.
        
        Returns:
            Noise level classification: 'low', 'medium', 'high'
        """
        if not self.baseline_established:
            return 'unknown'
        
        if current_energy < self.baseline_mean - self.baseline_std:
            return 'low'  # Quieter than teacher (good - listening)
        elif current_energy < self.baseline_mean + self.baseline_std:
            return 'medium'  # Similar to teacher (normal)
        else:
            return 'high'  # Louder than teacher (disengagement)
    
    def _calculate_audio_engagement(self, features):
        """
        Calculate engagement score from audio features (LEGACY METHOD).
        Kept for backward compatibility.
        
        Logic:
        - Low noise + single speaker = High engagement (listening)
        - Multiple speakers = Low engagement (side conversations)
        - Excessive noise = Low engagement (disruptions)
        
        Returns:
            Engagement score (0-1)
        """
        score = 0.7  # Baseline neutral
        
        # Penalize multiple speakers (side conversations)
        if features['multiple_speakers_detected']:
            score -= 0.3
        
        # Penalize excessive noise
        if features['excessive_noise']:
            score -= 0.2
        
        # Reward low, attentive noise level
        if features['background_noise_level'] == 'low':
            score += 0.2
        elif features['background_noise_level'] == 'high':
            score -= 0.15
        
        # Consider speech activity ratio
        # Very high ratio with single speaker = teacher talking (neutral)
        # Medium ratio with multiple speakers = discussions (could be disengagement)
        speech_ratio = features['speech_activity_ratio']
        if speech_ratio > 0.7 and features['speaker_count'] == 1:
            score += 0.1  # Consistent single speaker (teacher)
        elif speech_ratio > 0.5 and features['speaker_count'] > 1:
            score -= 0.2  # Multiple people talking
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def _calculate_audio_engagement_v2(self, features):
        """
        NEW: Calculate engagement using teacher-filtered audio analysis.
        This method uses speaker enrollment to accurately detect student noise.
        
        Logic:
        - Teacher speaking alone = HIGH engagement (students listening)
        - Student noise detected = LOWER engagement (disruptions/side talk)
        - Noise level scales the disengagement
        
        Returns:
            Engagement score (0-1)
        """
        base_score = 0.85  # Start optimistic (engaged class)
        
        # If student noise is detected (teacher voice filtered out)
        if features.get('student_noise_detected', False):
            noise_level = features.get('student_noise_level', 0.0)
            
            # Heavy penalty for student noise
            # noise_level: 0.0 = no noise, 1.0 = maximum disruption
            noise_penalty = noise_level * 0.6  # Up to 60% reduction
            base_score -= noise_penalty
            
            logger.debug(f"Student noise detected! Level: {noise_level:.2f}, Penalty: {noise_penalty:.2f}")
        
        # If it's clearly teacher speaking (high similarity)
        elif features.get('is_teacher_speaking', False):
            teacher_sim = features.get('teacher_similarity', 0.0)
            
            # High teacher similarity = likely paying attention
            if teacher_sim > 0.8:
                base_score = 0.90  # Very engaged (listening to teacher)
            else:
                base_score = 0.75  # Moderate engagement
        
        # Check for excessive general noise (backup check)
        if features.get('excessive_noise', False):
            base_score *= 0.85  # 15% reduction
        
        # Clamp to valid range
        return max(0.0, min(1.0, base_score))
    
    def reset_baseline(self):
        """Reset baseline for new session."""
        self.baseline_energy = []
        self.baseline_established = False
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        logger.info("Baseline reset")
    
    def get_summary_stats(self):
        """Get summary statistics of recent audio."""
        if not self.energy_window:
            return {}
        
        return {
            'mean_energy': float(np.mean(self.energy_window)),
            'max_energy': float(np.max(self.energy_window)),
            'min_energy': float(np.min(self.energy_window)),
            'energy_std': float(np.std(self.energy_window)),
            'speech_activity_pct': float(np.mean([s > 0.5 for s in self.speech_window]) * 100) if self.speech_window else 0.0,
            'avg_speaker_count': float(np.mean(self.speaker_count_window)) if self.speaker_count_window else 0.0,
        }
