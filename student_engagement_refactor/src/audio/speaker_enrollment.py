import numpy as np
import torch
from collections import deque
from src.logging_utils import get_logger

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

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
        
        # Teacher MFCC profile (primary identity feature, requires librosa)
        self.teacher_mfcc_profile = None

        # Rolling statistics
        self.enrollment_samples = []
        self.max_enrollment_samples = 20  # ~10 seconds at 0.5s chunks

        # Rolling window for smoothed classification (last 5 chunks = ~2.5 sec).
        # Median is used so one outlier chunk never flips the label.
        self.similarity_window = deque(maxlen=5)

        logger.info(f"Speaker enrollment initialized: {enrollment_duration}s enrollment period")
        if HAS_LIBROSA:
            logger.info("  Using MFCC cosine similarity for speaker identification")
        else:
            logger.warning("  librosa not found — falling back to spectral similarity (less accurate)")
    
    def add_enrollment_sample(self, audio_chunk, sample_rate=16000):
        """
        Add audio sample during teacher enrollment period.
        Should be called only when teacher is speaking alone.
        
        Args:
            audio_chunk: Audio data (numpy array)
            sample_rate: Sample rate of audio
            
        Returns:
            True if enrollment is complete, False otherwise
        """
        if self.is_enrolled:
            logger.warning("Already enrolled, ignoring new sample")
            return True

        # --- Energy gate: only enroll chunks where teacher is actually speaking ---
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < 0.006:
            logger.debug(f"Skipping silent enrollment chunk (rms={rms:.4f})")
            return False

        # Extract spectral features
        features = self._extract_spectral_features(audio_chunk, sample_rate)
        self.enrollment_samples.append(features)
        
        logger.debug(f"Enrollment sample {len(self.enrollment_samples)}/{self.max_enrollment_samples} added")
        
        # Check if we have enough samples
        if len(self.enrollment_samples) >= self.max_enrollment_samples:
            self._finalize_enrollment()
            return True
        
        return False
    
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
        
        result = {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flux': spectral_flux,
            'zcr': zcr,
            'low_energy_ratio': low_energy / total_energy,
            'mid_energy_ratio': mid_energy / total_energy,
            'high_energy_ratio': high_energy / total_energy,
            'total_energy': total_energy,
            'mfcc_mean': None,
        }

        # MFCC — primary speaker identity feature (requires librosa)
        if HAS_LIBROSA:
            try:
                mfcc = librosa.feature.mfcc(
                    y=audio_chunk.astype(np.float32), sr=sample_rate, n_mfcc=13
                )
                result['mfcc_mean'] = np.mean(mfcc, axis=1)  # shape (13,)
            except Exception as e:
                logger.debug(f"MFCC extraction failed: {e}")

        return result
    
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
        
        # Build MFCC profile (primary identity vector)
        if HAS_LIBROSA:
            mfcc_vectors = [
                s['mfcc_mean'] for s in self.enrollment_samples
                if s.get('mfcc_mean') is not None
            ]
            if mfcc_vectors:
                mfcc_array = np.array(mfcc_vectors)  # shape (N, 13)
                self.teacher_mfcc_profile = np.mean(mfcc_array, axis=0)
                # Per-coefficient std dev — used for z-score similarity.
                # Floor raised to 3.0 (was 1.0) so natural phoneme-to-phoneme
                # variation in MFCC_0 (energy) and formant coefficients doesn't
                # push the teacher's own chunks into high z-score territory.
                self.teacher_mfcc_std = np.std(mfcc_array, axis=0) + 3.0
                logger.info(f"  MFCC profile built from {len(mfcc_vectors)} speech chunks")

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

        # Raw per-chunk similarity
        raw_similarity = self._calculate_similarity(current_features)

        # Smooth over last 5 chunks (~2.5 sec) using median.
        # One high-scoring YouTube chunk won't flip to "teacher";
        # one low-scoring teacher chunk won't flip to "student".
        self.similarity_window.append(raw_similarity)
        smoothed_similarity = float(np.median(self.similarity_window))

        is_teacher = smoothed_similarity > self.similarity_threshold

        return is_teacher, smoothed_similarity
    
    def _calculate_similarity(self, current_features):
        """
        Calculate similarity between current audio and teacher profile.

        Primary method: MFCC cosine similarity (requires librosa).
        Fallback: improved spectral feature comparison with wider tolerances.

        Returns:
            Similarity score [0, 1], where 1 = identical to teacher
        """
        if not self.is_enrolled or not self.teacher_profile:
            return 0.0

        # ── Primary: z-score based MFCC matching ────────────────────────────
        # Cosine similarity is unsuitable here — it scores ALL human speech > 0.9
        # because MFCC mean vectors share the same statistical shape for any speaker.
        # Instead we measure how many std-devs each coefficient deviates from the
        # teacher's enrollment distribution.  Same speaker → small z-scores (~0-1).
        # Different speaker → large z-scores (~3-6 for most coefficients).
        if (
            HAS_LIBROSA
            and self.teacher_mfcc_profile is not None
            and hasattr(self, 'teacher_mfcc_std')
            and current_features.get('mfcc_mean') is not None
        ):
            z_scores = np.abs(
                current_features['mfcc_mean'] - self.teacher_mfcc_profile
            ) / self.teacher_mfcc_std  # per-coefficient normalised distance
            mean_z = float(np.mean(z_scores))
            # Convert to [0,1]: z≈0 → 1.0 (identical), large z → near 0
            # Scale factor 3.0: teacher's natural variation (mean_z ~1.0-1.6)
            # scores 0.59-0.72; a genuinely different speaker (mean_z ~3-5)
            # scores 0.19-0.37, well below the 0.45 threshold.
            similarity = float(np.exp(-mean_z / 3.0))
            return similarity

        # ── Fallback: spectral comparison with realistic tolerances ──────────
        # Natural speech for 1 speaker varies ±1000Hz in centroid → use 1500Hz tolerance
        centroid_diff = abs(current_features['spectral_centroid'] - self.teacher_profile['spectral_centroid'])
        centroid_tolerance = max(self.teacher_profile['spectral_centroid_std'] * 3.0, 1500.0)
        centroid_similarity = max(0.0, 1.0 - centroid_diff / centroid_tolerance)

        rolloff_diff = abs(current_features['spectral_rolloff'] - self.teacher_profile['spectral_rolloff'])
        rolloff_tolerance = max(self.teacher_profile['spectral_rolloff_std'] * 3.0, 2000.0)
        rolloff_similarity = max(0.0, 1.0 - rolloff_diff / rolloff_tolerance)

        zcr_diff = abs(current_features['zcr'] - self.teacher_profile['zcr'])
        zcr_similarity = max(0.0, 1.0 - zcr_diff / 0.20)  # 0.20 tolerance

        band_diff = (
            abs(current_features['low_energy_ratio'] - self.teacher_profile['low_energy_ratio']) +
            abs(current_features['mid_energy_ratio'] - self.teacher_profile['mid_energy_ratio']) +
            abs(current_features['high_energy_ratio'] - self.teacher_profile['high_energy_ratio'])
        ) / 3.0
        band_similarity = max(0.0, 1.0 - band_diff * 1.5)

        return (
            0.30 * centroid_similarity +
            0.25 * rolloff_similarity +
            0.15 * zcr_similarity +
            0.30 * band_similarity
        )
    
    def get_student_noise_level(self, audio_chunk, sample_rate=16000, speaker_count=1):
        """
        Analyze audio to determine student noise level (excluding teacher).
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            speaker_count: Number of speakers detected (from VAD)
            
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
        
        # ── Silence gate — no speech means no noise ────────────────────────
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < 0.006:
            return {
                'student_noise_detected': False,
                'noise_level': 0.0,
                'is_teacher': False,
                'teacher_similarity': 0.0,
                'confidence': 1.0,
                'status': 'silence',
                'speaker_count': speaker_count,
            }

        # ── Primary check: does this audio match the teacher's voice? ─────────
        # speaker_count from the VAD fallback is unreliable (not Silero) so we
        # treat it only as soft evidence, not a hard override.
        is_teacher, teacher_similarity = self.is_teacher_speaking(audio_chunk, sample_rate)

        if is_teacher:
            # Teacher's voice — check if energy is unusually high
            # (would indicate students talking loudly over the teacher)
            features = self._extract_spectral_features(audio_chunk, sample_rate)
            expected_energy = self.teacher_profile.get('total_energy_mean', 1.0)
            energy_ratio = features['total_energy'] / (expected_energy + 1e-10)

            if energy_ratio > 2.5:  # >2.5× louder than teacher baseline = extra voices
                noise_level = min(1.0, (energy_ratio - 1.0) / 4.0)
                student_noise_detected = noise_level > 0.35
            else:
                noise_level = 0.0
                student_noise_detected = False
        else:
            # Not teacher's voice — likely a student speaking
            dissimilarity = 1.0 - teacher_similarity
            noise_level = min(1.0, dissimilarity)
            student_noise_detected = noise_level > 0.40

            if student_noise_detected:
                logger.info(
                    f"⚠️  Student noise detected "
                    f"(teacher_similarity={teacher_similarity:.3f} < {self.similarity_threshold})"
                )

        return {
            'student_noise_detected': student_noise_detected,
            'noise_level': float(noise_level),
            'is_teacher': is_teacher,
            'teacher_similarity': float(teacher_similarity),
            'similarity_threshold': float(self.similarity_threshold),
            'confidence': float(teacher_similarity if is_teacher else 1.0 - teacher_similarity),
            'status': 'active',
            'speaker_count': speaker_count,
        }
    
    def reset(self):
        """Reset enrollment to start fresh."""
        self.teacher_profile = None
        self.enrollment_samples = []
        self.similarity_window.clear()
        self.is_enrolled = False
        logger.info("Speaker enrollment reset")
