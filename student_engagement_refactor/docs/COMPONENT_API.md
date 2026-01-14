# Component API Documentation

## Overview

This document provides detailed API documentation for all components in the Student Engagement Detection system. Each module is documented with its classes, methods, parameters, return values, and usage examples.

---

## Table of Contents

1. [Detection Module](#1-detection-module)
2. [Tracking Module](#2-tracking-module)
3. [Features Module](#3-features-module)
4. [Fusion Module](#4-fusion-module)
5. [Audio Modules](#5-audio-modules)
6. [Capture Module](#6-capture-module)
7. [Data Logger Module](#7-data-logger-module)
8. [Evaluation Module](#8-evaluation-module)

---

## 1. Detection Module

### File: `src/detection/yolov_wrapper.py`

#### Class: `YoloDetector`

Wrapper for YOLOv8 object detection model.

##### Constructor

```python
def __init__(self, weights_path: str, device: str = "cpu", 
             conf: float = 0.35, iou: float = 0.45)
```

**Parameters**:
- `weights_path` (str): Path to YOLOv8 model weights file (e.g., "models/weights/yolov8s.pt")
- `device` (str, optional): Device to run inference on. Options: "cpu", "cuda", "mps". Default: "cpu"
- `conf` (float, optional): Confidence threshold for detections. Range: [0, 1]. Default: 0.35
- `iou` (float, optional): IoU threshold for Non-Maximum Suppression. Range: [0, 1]. Default: 0.45

**Raises**:
- `FileNotFoundError`: If weights_path does not exist

**Example**:
```python
detector = YoloDetector(
    weights_path="models/weights/yolov8s.pt",
    device="cpu",
    conf=0.20,
    iou=0.45
)
```

##### Method: `detect`

Detect objects in an image frame.

```python
def detect(self, frame: np.ndarray) -> List[Dict[str, Union[int, float]]]
```

**Parameters**:
- `frame` (np.ndarray): Input image in BGR format (H×W×3)

**Returns**:
- `List[Dict]`: List of detection dictionaries, each containing:
  - `xmin` (int): Left coordinate of bounding box
  - `ymin` (int): Top coordinate of bounding box
  - `xmax` (int): Right coordinate of bounding box
  - `ymax` (int): Bottom coordinate of bounding box
  - `conf` (float): Detection confidence score [0, 1]
  - `cls` (int): Class ID (0 = person in COCO dataset)

**Example**:
```python
frame = cv2.imread("classroom.jpg")
detections = detector.detect(frame)

for det in detections:
    if det['cls'] == 0:  # Person class
        x1, y1 = det['xmin'], det['ymin']
        x2, y2 = det['xmax'], det['ymax']
        confidence = det['conf']
        print(f"Person detected at ({x1},{y1}) with confidence {confidence:.2f}")
```

**Performance**:
- CPU: ~8-15ms per frame (1280×720)
- GPU: ~2-4ms per frame

---

## 2. Tracking Module

### File: `src/tracking/sort_tracker.py`

#### Class: `Sort`

Simple Online and Realtime Tracking (SORT) algorithm for multi-object tracking.

##### Constructor

```python
def __init__(self, max_age: int = 30, min_hits: int = 3, 
             iou_threshold: float = 0.3)
```

**Parameters**:
- `max_age` (int, optional): Maximum frames to keep a track without updates. Default: 30
- `min_hits` (int, optional): Minimum detections before assigning ID. Default: 3
- `iou_threshold` (float, optional): Minimum IoU for matching detection to track. Default: 0.3

**Example**:
```python
tracker = Sort(
    max_age=60,        # Keep tracks for 2 seconds at 30 FPS
    min_hits=2,        # Assign ID after 2 detections
    iou_threshold=0.25 # Lenient matching
)
```

##### Method: `update`

Update tracker with new detections.

```python
def update(self, dets: List[List[int]]) -> List[Tuple[int, int, int, int, int]]
```

**Parameters**:
- `dets` (List[List[int]]): List of detections in format `[[x1,y1,x2,y2], ...]`

**Returns**:
- `List[Tuple]`: List of tracked objects as `(x1, y1, x2, y2, track_id)`
  - `x1, y1` (int): Top-left corner
  - `x2, y2` (int): Bottom-right corner
  - `track_id` (int): Persistent unique ID

**Example**:
```python
# Convert YOLOv8 detections to SORT format
dets = [[d['xmin'], d['ymin'], d['xmax'], d['ymax']] for d in detections]

# Update tracker
tracked_objects = tracker.update(dets)

for x1, y1, x2, y2, track_id in tracked_objects:
    print(f"Student ID: {track_id}, Position: ({x1},{y1})-({x2},{y2})")
```

**Tracking Metrics**:
- MOTA (Multi-Object Tracking Accuracy): ~75-85%
- ID Switches: ~2-5 per 1000 frames
- Processing Time: <1ms per frame

#### Class: `Track`

Internal class representing a single tracked object.

**Attributes**:
- `bbox` (List[int]): Current bounding box [x1, y1, x2, y2]
- `id` (int): Unique track ID
- `hits` (int): Number of successful updates
- `no_losses` (int): Frames since last update
- `kf` (KalmanFilter): Kalman filter for state prediction

**Methods**:
- `predict()`: Predict next position using Kalman filter
- `update(bbox)`: Update track with new measurement

---

## 3. Features Module

### File: `src/features/visual_features.py`

#### Function: `extract_all`

Extract all visual features from a person bounding box.

```python
def extract_all(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                image_w: int, image_h: int) -> Dict[str, Any]
```

**Parameters**:
- `frame` (np.ndarray): Input image (H×W×3, BGR)
- `bbox` (Tuple[int]): Bounding box as (xmin, ymin, xmax, ymax)
- `image_w` (int): Image width
- `image_h` (int): Image height

**Returns**:
- `Dict[str, Any]`: Dictionary containing:
  - `gaze` (str): Gaze direction ("Forward", "Left", "Right", or None)
  - `mouth_open` (float): Mouth aspect ratio [0, 0.1] or None
  - `eye_openness` (float): Eye aspect ratio [0, 0.06] or None
  - `head_pitch` (float): Head pitch angle [-0.3, 0.3] or None
  - `face_landmarks` (List): MediaPipe face landmarks or None
  - `pose_landmarks` (PoseLandmarks): MediaPipe pose landmarks or None

**Example**:
```python
frame = cv2.imread("student.jpg")
h, w = frame.shape[:2]
bbox = (100, 50, 300, 400)  # Person bounding box

features = extract_all(frame, bbox, w, h)

print(f"Gaze: {features['gaze']}")
print(f"Eye Openness: {features['eye_openness']:.3f}")
print(f"Mouth Open: {features['mouth_open']:.3f}")
print(f"Head Pitch: {features['head_pitch']:.3f}")
```

**Feature Ranges**:
- `gaze`: Categorical ("Forward", "Left", "Right")
- `eye_openness`: 0.0 (closed) to ~0.06 (fully open)
- `mouth_open`: 0.0 (closed) to ~0.1 (yawning)
- `head_pitch`: -0.3 (looking down) to 0.3 (looking up)

**Fallback Behavior**:
If MediaPipe is unavailable, uses OpenCV Haar Cascade with heuristic feature estimation.

#### Function: `extract_features_opencv`

OpenCV-based fallback for feature extraction (internal).

```python
def extract_features_opencv(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                           image_w: int, image_h: int) -> Dict[str, Any]
```

Uses Haar Cascade face detection and position/intensity-based heuristics.

#### Helper Functions

```python
def compute_gaze(face_landmarks) -> str
def mouth_open_ratio(face_landmarks) -> float
def eye_openness(face_landmarks) -> float
def head_pose_from_pose(pose_landmarks, image_w, image_h) -> float
def relative_point(pt, w, h) -> Tuple[int, int]
```

---

## 4. Fusion Module

### File: `src/fusion/fusion.py`

#### Function: `simple_engagement_score`

Compute engagement score from visual features using weighted fusion.

```python
def simple_engagement_score(features: Dict[str, Any]) -> float
```

**Parameters**:
- `features` (Dict[str, Any]): Feature dictionary containing:
  - `gaze` (str): Gaze direction
  - `eye_openness` (float): Eye aspect ratio
  - `mouth_open` (float): Mouth aspect ratio
  - `head_pitch` (float): Head pitch angle
  - `movement` (float): Movement metric

**Returns**:
- `float`: Engagement score in [0, 1] range
  - 0.0 = Completely disengaged
  - 1.0 = Fully engaged

**Scoring Weights**:
- Gaze: 40%
- Eye openness: 25%
- Head pose: 20%
- Movement: 10%
- Mouth: 5%

**Example**:
```python
features = {
    'gaze': 'Forward',
    'eye_openness': 0.048,
    'mouth_open': 0.02,
    'head_pitch': 0.05,
    'movement': 5.0
}

score = simple_engagement_score(features)
print(f"Engagement: {score:.2f}")  # Output: ~0.87

# Access score breakdown
breakdown = features.get('_score_breakdown', {})
print(f"Gaze contribution: {breakdown['gaze']:.2f}")
print(f"Eye contribution: {breakdown['eye']:.2f}")
```

#### Function: `multimodal_engagement_score`

Compute multimodal engagement score from visual and audio features.

```python
def multimodal_engagement_score(visual_features: Dict[str, Any],
                                audio_features: Optional[Dict[str, Any]] = None,
                                visual_weight: float = 0.65,
                                audio_weight: float = 0.35,
                                verbose: bool = False) -> float
```

**Parameters**:
- `visual_features` (Dict): Visual feature dictionary (see `extract_all`)
- `audio_features` (Dict, optional): Audio feature dictionary (see AudioFeatureExtractor)
- `visual_weight` (float): Weight for visual modality. Default: 0.65
- `audio_weight` (float): Weight for audio modality. Default: 0.35
- `verbose` (bool): Print detailed score breakdown. Default: False

**Returns**:
- `float`: Combined engagement score [0, 1]

**Example**:
```python
visual_features = {
    'gaze': 'Forward',
    'eye_openness': 0.048,
    'mouth_open': 0.02,
    'head_pitch': 0.05,
    'movement': 5.0
}

audio_features = {
    'audio_engagement_score': 0.92,
    'student_noise_detected': False,
    'is_teacher_speaking': True
}

final_score = multimodal_engagement_score(
    visual_features, 
    audio_features,
    verbose=True
)

print(f"Final engagement: {final_score:.2f}")
# Output: Visual: 0.89, Audio: 0.92, Final: 0.90
```

#### Function: `compute_engagement_score`

Generic engagement computation with multiple methods.

```python
def compute_engagement_score(features: Dict[str, Any], 
                            method: str = 'weighted') -> float
```

**Parameters**:
- `features` (Dict): Feature dictionary
- `method` (str): Scoring method. Options:
  - `'weighted'`: Weighted sum (default)
  - `'threshold'`: Binary threshold-based
  - `'ml'`: Placeholder for ML models

**Returns**:
- `float`: Engagement score [0, 1]

---

## 5. Audio Modules

### 5.1 Audio Capture

#### File: `src/audio/audio_capture.py`

##### Class: `AudioCapture`

Real-time audio capture from microphone.

```python
def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.5)
```

**Parameters**:
- `sample_rate` (int): Audio sample rate in Hz. Default: 16000
- `chunk_duration` (float): Duration of each chunk in seconds. Default: 0.5

**Methods**:

###### `start()`
Start audio capture in background thread.

```python
def start(self)
```

###### `stop()`
Stop audio capture and release resources.

```python
def stop(self)
```

###### `get_audio_chunk()`
Get latest audio chunk.

```python
def get_audio_chunk(self, timeout: float = 0.01) -> Optional[Dict]
```

**Returns**:
- `Dict` or `None`: Audio chunk with keys:
  - `data` (np.ndarray): Audio samples, float32, shape (num_samples,)
  - `timestamp` (float): Capture timestamp

**Example**:
```python
audio_capture = AudioCapture(sample_rate=16000, chunk_duration=0.5)
audio_capture.start()

while True:
    chunk = audio_capture.get_audio_chunk(timeout=0.01)
    if chunk:
        audio_data = chunk['data']
        timestamp = chunk['timestamp']
        print(f"Captured {len(audio_data)} samples at {timestamp:.2f}s")
    time.sleep(0.1)

audio_capture.stop()
```

---

### 5.2 VAD Detector

#### File: `src/audio/vad_detector.py`

##### Class: `VADDetector`

Voice Activity Detection using Silero VAD model.

```python
def __init__(self, threshold: float = 0.5, sample_rate: int = 16000)
```

**Parameters**:
- `threshold` (float): Speech probability threshold. Default: 0.5
- `sample_rate` (int): Audio sample rate. Must be 8000 or 16000. Default: 16000

##### Method: `detect_speech`

Detect if audio chunk contains speech.

```python
def detect_speech(self, audio_chunk: np.ndarray, 
                 return_confidence: bool = False) -> Union[bool, float]
```

**Parameters**:
- `audio_chunk` (np.ndarray): Audio samples, float32, [-1, 1] range
- `return_confidence` (bool): Return probability instead of boolean. Default: False

**Returns**:
- `bool` or `float`: Speech detected (if return_confidence=False) or probability [0,1]

**Example**:
```python
vad = VADDetector(threshold=0.5, sample_rate=16000)

# Boolean detection
speech_detected = vad.detect_speech(audio_chunk)
print(f"Speech: {speech_detected}")

# Probability
speech_prob = vad.detect_speech(audio_chunk, return_confidence=True)
print(f"Speech probability: {speech_prob:.2f}")
```

##### Method: `count_speakers_estimate`

Estimate number of concurrent speakers.

```python
def count_speakers_estimate(self, audio_chunk: np.ndarray) -> int
```

**Returns**:
- `int`: Estimated speaker count (1, 2, or 3+)

---

### 5.3 Audio Feature Extractor

#### File: `src/audio/audio_features.py`

##### Class: `AudioFeatureExtractor`

Extract audio features for engagement analysis.

```python
def __init__(self, baseline_duration: float = 5.0, 
             noise_threshold: float = 0.03, 
             sample_rate: int = 16000)
```

**Parameters**:
- `baseline_duration` (float): Seconds to establish baseline. Default: 5.0
- `noise_threshold` (float): Energy threshold for noise. Default: 0.03
- `sample_rate` (int): Audio sample rate. Default: 16000

##### Method: `update_baseline`

Update baseline with teacher's voice profile.

```python
def update_baseline(self, audio_chunk: np.ndarray)
```

**Parameters**:
- `audio_chunk` (np.ndarray): Audio data from teacher speaking

Call during initialization period (first 10-20 seconds).

##### Method: `extract_features`

Extract engagement features from audio chunk.

```python
def extract_features(self, audio_chunk: np.ndarray,
                    vad_result: Optional[float] = None,
                    speaker_count: int = 1,
                    speaker_enrollment: Optional['SpeakerEnrollment'] = None) -> Dict
```

**Parameters**:
- `audio_chunk` (np.ndarray): Audio samples
- `vad_result` (float, optional): Speech probability from VAD
- `speaker_count` (int): Estimated number of speakers
- `speaker_enrollment` (SpeakerEnrollment, optional): For teacher filtering

**Returns**:
- `Dict`: Audio features containing:
  - `audio_energy` (float): RMS energy
  - `zero_crossing_rate` (float): ZCR metric
  - `speech_probability` (float): From VAD
  - `speaker_count` (int): Number of speakers
  - `student_noise_detected` (bool): Student noise present
  - `student_noise_level` (float): Noise intensity
  - `is_teacher_speaking` (bool): Teacher voice detected
  - `audio_engagement_score` (float): Engagement [0, 1]

**Example**:
```python
extractor = AudioFeatureExtractor(baseline_duration=5.0)

# Establish baseline (first 10 seconds)
for i in range(20):
    chunk = audio_capture.get_audio_chunk()
    extractor.update_baseline(chunk['data'])

# Extract features
features = extractor.extract_features(
    audio_chunk=chunk['data'],
    vad_result=speech_prob,
    speaker_count=1,
    speaker_enrollment=enrollment
)

print(f"Student noise: {features['student_noise_detected']}")
print(f"Engagement: {features['audio_engagement_score']:.2f}")
```

---

### 5.4 Speaker Enrollment

#### File: `src/audio/speaker_enrollment.py`

##### Class: `SpeakerEnrollment`

Speaker identification using voice embeddings.

```python
def __init__(self, enrollment_duration: float = 10.0,
             similarity_threshold: float = 0.25)
```

**Parameters**:
- `enrollment_duration` (float): Seconds to enroll teacher. Default: 10.0
- `similarity_threshold` (float): Threshold for teacher classification. Default: 0.25

##### Method: `add_enrollment_sample`

Add audio sample for teacher enrollment.

```python
def add_enrollment_sample(self, audio_data: np.ndarray,
                         sample_rate: int = 16000) -> bool
```

**Parameters**:
- `audio_data` (np.ndarray): Audio samples from teacher
- `sample_rate` (int): Sample rate

**Returns**:
- `bool`: True if enrollment complete

**Example**:
```python
enrollment = SpeakerEnrollment(enrollment_duration=10.0)

print("Please speak for 10 seconds...")
while not enrollment.is_enrolled:
    chunk = audio_capture.get_audio_chunk()
    complete = enrollment.add_enrollment_sample(chunk['data'])
    if complete:
        print("Enrollment complete!")
        break
```

##### Method: `is_teacher_speaking`

Check if current audio is from teacher.

```python
def is_teacher_speaking(self, audio_data: np.ndarray,
                       sample_rate: int = 16000) -> bool
```

**Returns**:
- `bool`: True if teacher voice detected

##### Method: `get_student_noise_level`

Analyze student noise with teacher filtering.

```python
def get_student_noise_level(self, audio_data: np.ndarray,
                            sample_rate: int = 16000,
                            speaker_count: int = 1) -> Dict
```

**Returns**:
- `Dict`: Noise analysis with keys:
  - `student_noise_detected` (bool)
  - `noise_level` (float)
  - `is_teacher` (bool)
  - `teacher_similarity` (float)

---

## 6. Capture Module

### File: `src/capture.py`

#### Class: `Camera`

Video capture wrapper for webcam, file, or stream.

```python
def __init__(self, src=0, width: int = 1280, height: int = 720)
```

**Parameters**:
- `src`: Video source. Options:
  - `int`: Webcam index (0 = default webcam)
  - `str`: Video file path or RTSP/HTTP stream URL
- `width` (int): Desired frame width. Default: 1280
- `height` (int): Desired frame height. Default: 720

**Example**:
```python
# Webcam
cam = Camera(src=0, width=1280, height=720)

# Video file
cam = Camera(src="lecture.mp4")

# IP camera stream
cam = Camera(src="http://192.168.1.100:8080/video")
```

##### Method: `read`

Read next frame from video source.

```python
def read(self) -> Optional[np.ndarray]
```

**Returns**:
- `np.ndarray` or `None`: Frame in BGR format (H×W×3) or None if stream ended

**Example**:
```python
cam = Camera(src=0)

while True:
    frame = cam.read()
    if frame is None:
        print("Stream ended")
        break
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
```

##### Method: `release`

Release video capture resources.

```python
def release(self)
```

---

## 7. Data Logger Module

### File: `src/data_logger.py`

#### Class: `DataLogger`

CSV logger for timestamped engagement data.

```python
def __init__(self, csv_path: str = "data/labels/engagement_data.csv")
```

**Parameters**:
- `csv_path` (str): Path to output CSV file

##### Method: `log`

Log engagement data for a student.

```python
def log(self, timestamp: str, frame: int, student_id: int,
        features: Dict, engagement_score: float,
        bbox: Tuple[int, int, int, int])
```

**Parameters**:
- `timestamp` (str): ISO format timestamp
- `frame` (int): Frame number
- `student_id` (int): Track ID
- `features` (Dict): Feature dictionary
- `engagement_score` (float): Final engagement score
- `bbox` (Tuple): Bounding box (x, y, w, h)

**Example**:
```python
logger = DataLogger(csv_path="data/engagement.csv")

logger.log(
    timestamp="2026-01-14 10:30:15.123",
    frame=100,
    student_id=0,
    features=features,
    engagement_score=0.87,
    bbox=(120, 80, 180, 320)
)

logger.close()
```

##### Method: `get_summary`

Get summary statistics for logged session.

```python
def get_summary(self) -> Dict
```

**Returns**:
- `Dict`: Summary containing:
  - `total_entries` (int)
  - `unique_students` (int)
  - `avg_engagement` (float)
  - `duration` (float): Session duration in seconds

##### Method: `close`

Close CSV file and flush buffer.

```python
def close(self)
```

---

## 8. Evaluation Module

### File: `src/evaluation/metrics.py`

#### Function: `compute_regression_metrics`

Compute regression metrics for engagement prediction.

```python
def compute_regression_metrics(y_true: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, float]
```

**Parameters**:
- `y_true` (np.ndarray): Ground truth engagement scores
- `y_pred` (np.ndarray): Predicted engagement scores

**Returns**:
- `Dict`: Metrics including:
  - `mae`: Mean Absolute Error
  - `rmse`: Root Mean Square Error
  - `pearson_r`: Pearson correlation coefficient
  - `pearson_p`: P-value

**Example**:
```python
import pandas as pd
from src.evaluation.metrics import compute_regression_metrics

# Load predictions
df = pd.read_csv("engagement.csv")
y_true = df['ground_truth'].values
y_pred = df['engagement_score'].values

metrics = compute_regression_metrics(y_true, y_pred)
print(f"MAE: {metrics['mae']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"Correlation: {metrics['pearson_r']:.3f}")
```

#### Function: `compute_classification_metrics`

Compute classification metrics for engagement levels.

```python
def compute_classification_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   labels: List[str] = None) -> Dict
```

**Parameters**:
- `y_true` (np.ndarray): Ground truth labels
- `y_pred` (np.ndarray): Predicted labels
- `labels` (List[str], optional): Label names

**Returns**:
- `Dict`: Metrics including:
  - `accuracy`: Overall accuracy
  - `precision`: Per-class precision
  - `recall`: Per-class recall
  - `f1_score`: Per-class F1 score
  - `confusion_matrix`: Confusion matrix

---

## Usage Examples

### Complete Pipeline Example

```python
import cv2
from src.detection.yolov_wrapper import YoloDetector
from src.tracking.sort_tracker import Sort
from src.features.visual_features import extract_all
from src.fusion.fusion import multimodal_engagement_score
from src.data_logger import DataLogger

# Initialize components
detector = YoloDetector("models/weights/yolov8s.pt")
tracker = Sort(max_age=60, min_hits=2)
logger = DataLogger("engagement.csv")

# Open video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    
    # Detect persons
    detections = detector.detect(frame)
    person_dets = [d for d in detections if d['cls'] == 0]
    
    # Convert to SORT format
    dets_sort = [[d['xmin'], d['ymin'], d['xmax'], d['ymax']] 
                 for d in person_dets]
    
    # Track
    tracked = tracker.update(dets_sort)
    
    # Process each student
    for x1, y1, x2, y2, student_id in tracked:
        # Extract features
        features = extract_all(frame, (x1, y1, x2, y2), w, h)
        
        # Compute engagement
        score = simple_engagement_score(features)
        
        # Log data
        logger.log(
            timestamp=datetime.now().isoformat(),
            frame=cap.get(cv2.CAP_PROP_POS_FRAMES),
            student_id=student_id,
            features=features,
            engagement_score=score,
            bbox=(x1, y1, x2-x1, y2-y1)
        )
        
        # Visualize
        color = (0, 255, 0) if score > 0.6 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{student_id} {score:.2f}",
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 2)
    
    cv2.imshow("Engagement", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
logger.close()
cv2.destroyAllWindows()
```

---

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Process multiple frames in batches for GPU efficiency
2. **Frame Skipping**: Skip frames during low activity periods
3. **ROI Caching**: Cache feature extraction results for stable tracks
4. **Async Audio**: Keep audio processing in separate thread
5. **Model Quantization**: Use quantized models for edge deployment

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| YOLOv8s | ~100MB | Model + activations |
| SORT Tracker | ~5MB | For 50 tracks |
| MediaPipe | ~50MB | Face mesh model |
| Audio Buffer | ~10MB | 15 seconds at 16kHz |
| **Total** | **~165MB** | Typical usage |

---

**Last Updated**: January 2026
**Version**: 1.0
