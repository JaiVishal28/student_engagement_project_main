# Audio Integration Setup Guide

## Overview
This guide explains how to set up and use the multimodal audio + visual engagement detection system.

## Architecture

### Audio Processing Pipeline
```
Phone Mic → AudioCapture → VADDetector → AudioFeatureExtractor → Multimodal Fusion
                ↓              ↓                ↓                        ↓
         Threading Queue   Silero VAD   Noise/Speaker Analysis    Combined Score
```

## Installation

### 1. Install Audio Dependencies

```bash
pip install pyaudio torch torchaudio
```

**Note for Windows:**
- PyAudio may require manual installation
- Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
- Install: `pip install PyAudio-0.2.13-cp311-cp311-win_amd64.whl`

### 2. Verify Installation

```python
python -c "import pyaudio; import torch; print('Audio ready!')"
```

## Configuration

Edit [config.yaml](config.yaml) to configure audio settings:

```yaml
logging:
  enable_audio: true  # Set false to disable audio

audio:
  sample_rate: 16000        # 16kHz optimal for VAD
  chunk_duration: 0.5       # Process 0.5 sec chunks
  vad_threshold: 0.5        # Speech detection threshold
  baseline_duration: 5.0    # Seconds to learn teacher voice
  noise_threshold: 0.03     # Background noise threshold
```

## How It Works

### 1. **Teacher Voice Baseline (First 5 seconds)**
- System records audio profile during initialization
- Assumes only teacher speaks in this period
- Creates baseline for noise comparison

### 2. **Continuous Audio Analysis**
- **Voice Activity Detection (VAD)**: Detects when speech occurs
- **Speaker Counting**: Estimates number of simultaneous speakers
- **Noise Level Analysis**: Compares current audio to baseline

### 3. **Engagement Scoring**

**Audio Engagement Indicators:**
- ✅ **Low noise** → Students listening attentively
- ✅ **Single speaker** → Teacher lecturing (neutral/positive)
- ❌ **Multiple speakers** → Side conversations (disengagement)
- ❌ **Excessive noise** → Disruptions (disengagement)

### 4. **Multimodal Fusion**
- **Visual Score (65%)** + **Audio Score (35%)** = **Final Engagement**
- Audio modulates visual score based on classroom atmosphere

## Usage

### Run with Audio Enabled (Default)
```bash
python -m src.main
```

### Run Visual-Only Mode
Edit `config.yaml`:
```yaml
logging:
  enable_audio: false
```

### Monitor Audio Status
During execution, you'll see:
```
INFO: Audio processing enabled - establishing baseline...
INFO: Audio baseline established
```

On-screen overlay shows:
- `Audio: low/medium/high` - Current noise level
- `Speakers: 0/1/3` - Estimated speaker count

## Output Data

CSV includes both visual and audio features:

| Column | Description |
|--------|-------------|
| `audio_energy` | RMS audio energy level |
| `speech_probability` | VAD confidence (0-1) |
| `speaker_count` | Estimated speakers (0, 1, 3+) |
| `background_noise_level` | low/medium/high |
| `audio_engagement_score` | Audio-only engagement (0-1) |
| `engagement_score` | **Final multimodal score** |

## Troubleshooting

### Issue: "Failed to initialize audio"
**Solutions:**
1. Check microphone permissions
2. Verify PyAudio installation
3. System will automatically fall back to visual-only mode

### Issue: "Multiple speakers always detected"
**Cause:** Noisy baseline period
**Solution:** 
- Ensure silence/only teacher talks in first 5 seconds
- Increase `baseline_duration` in config

### Issue: Audio lag/stutter
**Solutions:**
1. Increase `chunk_duration` to 1.0 sec
2. Reduce `process_every_n_frames` in config
3. Check CPU usage

### Issue: PyAudio installation fails on Windows
**Solution:**
```bash
# Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio-0.2.13-cp<YOUR_PYTHON_VERSION>-win_amd64.whl
```

## Phone Mic Setup

### Using Phone as Microphone

**Option 1: WO Mic (Recommended)**
1. Download WO Mic: http://wolicheng.com/womic/
2. Install on both phone and computer
3. Connect via USB/WiFi/Bluetooth
4. System will detect as default mic

**Option 2: IP Webcam App**
- Already includes audio streaming
- Audio automatically available when using phone camera

**Option 3: DroidCam**
- Download DroidCam client
- Includes both video and audio streaming

## Audio Feature Extraction Details

### Speaker Count Estimation
- **0**: Silence or very low audio
- **1**: Single speaker (likely teacher)
- **3**: Multiple speakers (side conversations)

### Noise Level Classification
- **Low**: Quieter than baseline (attentive)
- **Medium**: Similar to baseline (normal)
- **High**: Louder than baseline (disruptive)

### Audio Engagement Formula
```python
base_score = 0.7
- Penalty for multiple speakers: -0.3
- Penalty for excessive noise: -0.2
- Bonus for low noise + single speaker: +0.2
```

## Performance Notes

- **CPU Usage**: ~5-10% for audio processing
- **Memory**: ~100MB for Silero VAD model
- **Latency**: <100ms per audio chunk
- **Threading**: Audio runs in separate thread (non-blocking)

## Advanced Features (Future)

Planned enhancements:
- ✨ Speaker diarization for precise teacher filtering
- ✨ Emotion detection from voice tone
- ✨ Question detection (rising intonation)
- ✨ Attention patterns from speaking cadence

## API Reference

### AudioCapture
```python
from src.audio import AudioCapture

capture = AudioCapture(sample_rate=16000, chunk_duration=0.5)
capture.start()
audio_chunk = capture.get_audio_chunk(timeout=0.1)
capture.cleanup()
```

### VADDetector
```python
from src.audio import VADDetector

vad = VADDetector(threshold=0.5, sample_rate=16000)
is_speech = vad.detect_speech(audio_array)
speech_prob = vad.detect_speech(audio_array, return_confidence=True)
```

### AudioFeatureExtractor
```python
from src.audio import AudioFeatureExtractor

extractor = AudioFeatureExtractor()
extractor.update_baseline(audio_chunk)  # First 5 seconds
features = extractor.extract_features(audio_chunk, vad_result, speaker_count)
```

## Citation

If using audio features in research:
```bibtex
@software{multimodal_engagement,
  title={Multimodal Classroom Engagement Detection},
  author={Your Name},
  year={2026},
  note={Audio + Visual engagement analysis system}
}
```

## Support

For issues or questions:
1. Check logs in terminal output
2. Verify audio device in system settings
3. Test with `python -m src.audio.audio_capture` (coming soon)

---

**Ready to test!** Run `python -m src.main` and speak near the microphone to see audio features in action.
