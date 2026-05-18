"""
Consolidated audio module: capture, VAD, features, speaker enrollment.
"""
import numpy as np
import threading
import queue
import time
import logging
import pyaudio
import torch
from collections import deque

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

log = logging.getLogger("audio")


# ─── Audio Capture ──────────────────────────────────────────────────────────
class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio = None
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False

    def start(self):
        if self.running:
            return
        self.audio = pyaudio.PyAudio()
        dev = self.audio.get_default_input_device_info()
        log.info(f"Audio device: {dev['name']}")
        self.stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
            input=True, frames_per_buffer=self.chunk_size,
            stream_callback=self._callback
        )
        self.running = True
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self.audio_queue.put_nowait({'data': data, 'timestamp': time.time()})
        except queue.Full:
            pass
        return (in_data, pyaudio.paContinue)

    def get_chunk(self, timeout=0.01):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_running(self):
        return self.running and self.stream and self.stream.is_active()

    def cleanup(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
            self.audio = None


# ─── VAD Detector ───────────────────────────────────────────────────────────
class VADDetector:
    def __init__(self, threshold=0.5, sample_rate=16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = None
        try:
            self.model, self.utils = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=False
            )
            self.model.eval()
            log.info("Silero VAD loaded")
        except Exception as e:
            log.warning(f"VAD fallback to energy-based: {e}")

    def detect_speech(self, audio_chunk, return_confidence=False):
        if self.model is None:
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            val = min(energy / 0.3, 1.0)
            return val if return_confidence else energy > 0.02
        try:
            t = torch.from_numpy(audio_chunk).float()
            with torch.no_grad():
                prob = self.model(t, self.sample_rate).item()
            return prob if return_confidence else prob > self.threshold
        except:
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            val = min(energy / 0.3, 1.0)
            return val if return_confidence else energy > 0.02

    def count_speakers_estimate(self, audio_chunk):
        seg_size = len(audio_chunk) // 4
        if seg_size < 100:
            return 1
        segs = [audio_chunk[i:i+seg_size] for i in range(0, len(audio_chunk), seg_size)
                if len(audio_chunk[i:i+seg_size]) == seg_size]
        energies = [np.sqrt(np.mean(s ** 2)) for s in segs]
        if len(energies) < 2:
            return 1
        mean_e = np.mean(energies)
        if mean_e < 0.005:
            return 0
        cv = np.std(energies) / (mean_e + 1e-10)
        return 2 if cv > 1.2 and mean_e > 0.10 else 1


# ─── Speaker Enrollment ────────────────────────────────────────────────────
class SpeakerEnrollment:
    def __init__(self, enrollment_duration=10.0, similarity_threshold=0.75):
        self.enrollment_duration = enrollment_duration
        self.similarity_threshold = similarity_threshold
        self.teacher_profile = None
        self.teacher_mfcc_profile = None
        self.teacher_mfcc_std = None
        self.is_enrolled = False
        self.enrollment_samples = []
        self.max_enrollment_samples = 20
        self.similarity_window = deque(maxlen=5)

    def add_enrollment_sample(self, audio_chunk, sample_rate=16000):
        if self.is_enrolled:
            return True
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < 0.006:
            return False
        features = self._extract_features(audio_chunk, sample_rate)
        self.enrollment_samples.append(features)
        if len(self.enrollment_samples) >= self.max_enrollment_samples:
            self._finalize()
            return True
        return False

    def _extract_features(self, chunk, sr):
        fft = np.fft.rfft(chunk)
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(len(chunk), 1/sr)
        centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-10)
        cumsum = np.cumsum(mag)
        ri = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        rolloff = freqs[ri[0]] if len(ri) > 0 else 0
        flux = np.sqrt(np.sum(np.diff(mag)**2)) / len(mag)
        zcr = np.sum(np.abs(np.diff(np.sign(chunk)))) / 2 / len(chunk)
        low = mag[(freqs >= 0) & (freqs < 300)]
        mid = mag[(freqs >= 300) & (freqs < 2000)]
        high = mag[(freqs >= 2000) & (freqs < 8000)]
        le, me, he = np.sum(low**2), np.sum(mid**2), np.sum(high**2)
        te = le + me + he + 1e-10
        result = {
            'centroid': centroid, 'rolloff': rolloff, 'flux': flux, 'zcr': zcr,
            'low_r': le/te, 'mid_r': me/te, 'high_r': he/te, 'energy': te, 'mfcc': None
        }
        if HAS_LIBROSA:
            try:
                mfcc = librosa.feature.mfcc(y=chunk.astype(np.float32), sr=sr, n_mfcc=13)
                result['mfcc'] = np.mean(mfcc, axis=1)
            except:
                pass
        return result

    def _finalize(self):
        s = self.enrollment_samples
        self.teacher_profile = {
            'centroid': np.mean([x['centroid'] for x in s]),
            'rolloff': np.mean([x['rolloff'] for x in s]),
            'flux': np.mean([x['flux'] for x in s]),
            'zcr': np.mean([x['zcr'] for x in s]),
            'low_r': np.mean([x['low_r'] for x in s]),
            'mid_r': np.mean([x['mid_r'] for x in s]),
            'high_r': np.mean([x['high_r'] for x in s]),
            'energy_mean': np.mean([x['energy'] for x in s]),
            'energy_std': np.std([x['energy'] for x in s]),
            'centroid_std': np.std([x['centroid'] for x in s]),
            'rolloff_std': np.std([x['rolloff'] for x in s]),
        }
        if HAS_LIBROSA:
            vecs = [x['mfcc'] for x in s if x.get('mfcc') is not None]
            if vecs:
                arr = np.array(vecs)
                self.teacher_mfcc_profile = np.mean(arr, axis=0)
                self.teacher_mfcc_std = np.std(arr, axis=0) + 2.0
        self.is_enrolled = True
        log.info(f"Teacher enrolled from {len(s)} samples")

    def is_teacher_speaking(self, audio_chunk, sample_rate=16000):
        if not self.is_enrolled:
            return True, 1.0
        feats = self._extract_features(audio_chunk, sample_rate)
        raw = self._similarity(feats)
        self.similarity_window.append(raw)
        smoothed = float(np.median(self.similarity_window))
        return smoothed > self.similarity_threshold, smoothed

    def _similarity(self, feats):
        if not self.is_enrolled or not self.teacher_profile:
            return 0.0
        if HAS_LIBROSA and self.teacher_mfcc_profile is not None and self.teacher_mfcc_std is not None and feats.get('mfcc') is not None:
            z = np.abs(feats['mfcc'] - self.teacher_mfcc_profile) / self.teacher_mfcc_std
            return float(np.exp(-np.mean(z) / 3.0))
        p = self.teacher_profile
        c_sim = max(0, 1 - abs(feats['centroid'] - p['centroid']) / max(p['centroid_std']*3, 1500))
        r_sim = max(0, 1 - abs(feats['rolloff'] - p['rolloff']) / max(p['rolloff_std']*3, 2000))
        z_sim = max(0, 1 - abs(feats['zcr'] - p['zcr']) / 0.20)
        b_diff = (abs(feats['low_r']-p['low_r']) + abs(feats['mid_r']-p['mid_r']) + abs(feats['high_r']-p['high_r'])) / 3
        b_sim = max(0, 1 - b_diff * 1.5)
        return 0.30*c_sim + 0.25*r_sim + 0.15*z_sim + 0.30*b_sim

    def get_student_noise(self, audio_chunk, sample_rate=16000, speaker_count=1):
        if not self.is_enrolled:
            return {'detected': False, 'level': 0, 'is_teacher': True, 'similarity': 0, 'status': 'enrolling'}
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < 0.006:
            return {'detected': False, 'level': 0, 'is_teacher': False, 'similarity': 0, 'status': 'silence'}
        is_t, sim = self.is_teacher_speaking(audio_chunk, sample_rate)
        if is_t:
            feats = self._extract_features(audio_chunk, sample_rate)
            ratio = feats['energy'] / (self.teacher_profile['energy_mean'] + 1e-10)
            if ratio > 2.5:
                lvl = min(1, (ratio - 1) / 4)
                return {'detected': lvl > 0.35, 'level': lvl, 'is_teacher': True, 'similarity': sim, 'status': 'active'}
            return {'detected': False, 'level': 0, 'is_teacher': True, 'similarity': sim, 'status': 'active'}
        else:
            lvl = min(1, 1 - sim)
            return {'detected': lvl > 0.40, 'level': lvl, 'is_teacher': False, 'similarity': sim, 'status': 'active'}


# ─── Audio Feature Extractor ───────────────────────────────────────────────
class AudioFeatureExtractor:
    def __init__(self, baseline_duration=5.0, sample_rate=16000):
        self.baseline_energy = []
        self.baseline_established = False
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        self.energy_window = deque(maxlen=30)
        self.speech_window = deque(maxlen=30)
        self.sample_rate = sample_rate

    def update_baseline(self, chunk):
        e = float(np.sqrt(np.mean(chunk ** 2)))
        self.baseline_energy.append(e)
        if len(self.baseline_energy) >= 10 and not self.baseline_established:
            self.baseline_mean = np.mean(self.baseline_energy)
            self.baseline_std = np.std(self.baseline_energy)
            self.baseline_established = True

    def extract(self, chunk, vad_prob=0, speaker_count=1, enrollment=None):
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        self.energy_window.append(energy)
        if vad_prob is not None:
            self.speech_window.append(vad_prob)

        noise_info = None
        if enrollment and enrollment.is_enrolled:
            noise_info = enrollment.get_student_noise(chunk, self.sample_rate, speaker_count)

        noise_lvl = 'low' if energy < self.baseline_mean - self.baseline_std else (
            'high' if energy > self.baseline_mean + self.baseline_std else 'medium'
        ) if self.baseline_established else 'unknown'

        excessive = energy > (self.baseline_mean + 2*self.baseline_std) if self.baseline_established else False

        features = {
            'audio_energy': energy,
            'speech_probability': float(vad_prob) if vad_prob else 0,
            'speaker_count': speaker_count,
            'student_noise_detected': noise_info['detected'] if noise_info else False,
            'student_noise_level': noise_info['level'] if noise_info else 0,
            'is_teacher_speaking': noise_info['is_teacher'] if noise_info else True,
            'teacher_similarity': noise_info.get('similarity', 1) if noise_info else 1,
            'similarity_threshold': enrollment.similarity_threshold if enrollment else 0.65,
            'background_noise_level': noise_lvl,
            'multiple_speakers_detected': speaker_count > 1,
            'excessive_noise': excessive,
        }

        # Audio engagement score
        base = 0.85
        if features['student_noise_detected']:
            base -= features['student_noise_level'] * 0.6
        elif features['is_teacher_speaking']:
            base = 0.90 if features['teacher_similarity'] > 0.8 else 0.75
        if features['excessive_noise']:
            base *= 0.85
        features['audio_engagement_score'] = max(0, min(1, base))
        return features
