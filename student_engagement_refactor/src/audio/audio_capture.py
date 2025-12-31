import pyaudio
import numpy as np
import threading
import queue
import time
from src.logging_utils import get_logger

logger = get_logger("AudioCapture")


class AudioCapture:
    """
    Real-time audio capture from microphone using PyAudio.
    Runs in separate thread to avoid blocking video processing.
    """
    
    def __init__(self, sample_rate=16000, chunk_duration=0.5, channels=1):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Audio sampling rate (16kHz is optimal for VAD)
            chunk_duration: Duration of each audio chunk in seconds
            channels: Number of audio channels (1=mono, 2=stereo)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration)
        
        self.audio = None
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False
        self.capture_thread = None
        
        logger.info(f"AudioCapture initialized: {sample_rate}Hz, {chunk_duration}s chunks")
    
    def start(self):
        """Start audio capture in background thread."""
        if self.running:
            logger.warning("Audio capture already running")
            return
        
        try:
            self.audio = pyaudio.PyAudio()
            
            # Find default input device
            default_device = self.audio.get_default_input_device_info()
            logger.info(f"Using audio device: {default_device['name']}")
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.running = True
            self.stream.start_stream()
            logger.info("Audio capture started")
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.cleanup()
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback to capture audio data."""
        if status:
            logger.debug(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Normalize to [-1, 1]
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Add to queue (non-blocking)
        try:
            self.audio_queue.put_nowait({
                'data': audio_data,
                'timestamp': time.time(),
                'sample_rate': self.sample_rate
            })
        except queue.Full:
            logger.debug("Audio queue full, dropping frame")
        
        return (in_data, pyaudio.paContinue)
    
    def get_audio_chunk(self, timeout=0.1):
        """
        Get latest audio chunk from queue.
        
        Args:
            timeout: Maximum time to wait for audio data
            
        Returns:
            Dictionary with audio data and metadata, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_queue(self):
        """Clear all pending audio chunks from queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def stop(self):
        """Stop audio capture."""
        if not self.running:
            return
        
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        logger.info("Audio capture stopped")
    
    def cleanup(self):
        """Clean up audio resources."""
        self.stop()
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        self.clear_queue()
        logger.info("Audio capture cleaned up")
    
    def is_running(self):
        """Check if audio capture is running."""
        return self.running and self.stream and self.stream.is_active()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
