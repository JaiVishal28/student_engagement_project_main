# Complete Project Flow Documentation

## Overview

This document provides a comprehensive end-to-end flow of the Student Engagement Detection system, from startup to final output. It covers initialization, real-time processing, data logging, and system shutdown.

---

## Table of Contents

1. [System Initialization](#1-system-initialization)
2. [Runtime Processing Loop](#2-runtime-processing-loop)
3. [Visual Processing Pipeline](#3-visual-processing-pipeline)
4. [Audio Processing Pipeline](#4-audio-processing-pipeline)
5. [Multimodal Fusion](#5-multimodal-fusion)
6. [Visualization and Logging](#6-visualization-and-logging)
7. [System Shutdown](#7-system-shutdown)
8. [Data Flow Diagrams](#8-data-flow-diagrams)
9. [Timing and Performance](#9-timing-and-performance)

---

## 1. System Initialization

### 1.1 Startup Sequence

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM STARTUP                           │
└─────────────────────────────────────────────────────────────┘

Step 1: Parse Command-Line Arguments
  ├─ --source: Video source (0=webcam, file path, URL)
  ├─ --image: Single image mode
  ├─ --no-display: Headless mode (no GUI)
  ├─ --max-frames: Frame limit
  └─ --help: Show usage

Step 2: Load Configuration
  ├─ Read config.yaml
  ├─ Get capture settings (width, height, FPS)
  ├─ Get detection settings (weights, conf, IoU)
  ├─ Get tracking settings (max_age, min_hits)
  ├─ Get audio settings (sample_rate, VAD threshold)
  └─ Get logging settings (CSV path, enable_audio)

Step 3: Initialize Video Capture
  ├─ Create Camera object
  ├─ Open video source (webcam/file/stream)
  ├─ Set resolution (1280x720 default)
  └─ Verify frame capture

Step 4: Load Detection Model
  ├─ Check if yolov8s.pt exists
  ├─ Initialize YoloDetector
  ├─ Load model weights
  ├─ Set device (CPU/GPU)
  └─ Warm-up inference (1 dummy frame)

Step 5: Initialize Tracker
  ├─ Create SORT tracker
  ├─ Set max_age (60 frames)
  ├─ Set min_hits (2 detections)
  └─ Set IoU threshold (0.25)

Step 6: Initialize Logger
  ├─ Create DataLogger
  ├─ Generate CSV filename (engagement_YYYYMMDD_HHMMSS.csv)
  ├─ Create CSV header
  └─ Prepare output directory

Step 7: Initialize Audio System (if enabled)
  ├─ Create AudioCapture (16kHz, mono)
  ├─ Initialize VAD detector (Silero)
  ├─ Create AudioFeatureExtractor
  ├─ Create SpeakerEnrollment
  ├─ Start audio capture thread
  └─ Enter enrollment phase

Step 8: Teacher Voice Enrollment (10 seconds)
  ├─ Display enrollment screen
  ├─ Prompt teacher to speak
  ├─ Collect audio samples (20 chunks × 500ms)
  ├─ Extract MFCC features
  ├─ Compute teacher voice profile
  └─ Complete enrollment

Step 9: Audio Baseline Establishment (10 seconds)
  ├─ Continue collecting audio
  ├─ Compute baseline energy (mean, std)
  ├─ Store baseline for normalization
  └─ Mark baseline as established

Step 10: Start Processing
  ├─ Initialize frame counter
  ├─ Initialize FPS counter
  ├─ Create display window (if not headless)
  └─ Enter main processing loop
```

### 1.2 Initialization Code Flow

**File**: `src/main.py`

```python
def run_video_mode(source=None, display=True, max_frames=None):
    # 1. Load configuration
    cfg = yaml.safe_load(open('config.yaml'))
    
    # 2. Initialize camera
    cam = Camera(src=source, width=1280, height=720)
    
    # 3. Load detector
    detector = YoloDetector(weights_path="models/weights/yolov8s.pt")
    
    # 4. Initialize tracker
    tracker = Sort(max_age=60, min_hits=2, iou_threshold=0.25)
    
    # 5. Create logger
    datalog = DataLogger(csv_path="data/labels/engagement_data.csv")
    
    # 6. Initialize audio (if enabled)
    if use_audio:
        audio_capture = AudioCapture(sample_rate=16000)
        vad_detector = VADDetector(threshold=0.5)
        audio_extractor = AudioFeatureExtractor()
        speaker_enrollment = SpeakerEnrollment()
        audio_capture.start()
    
    # 7. Main loop
    while True:
        # ... processing (see next section)
```

---

## 2. Runtime Processing Loop

### 2.1 Main Loop Overview

```
┌──────────────────────────────────────────────────────────────┐
│                MAIN PROCESSING LOOP (per frame)              │
└──────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Read Frame     │ ← Camera.read()
└────────┬────────┘
         │
         ├─► Check for None (stream end)
         │
         ▼
┌─────────────────┐
│  Process Audio  │ ← Parallel audio thread
└────────┬────────┘
         │
         ├─► Get audio chunk (500ms)
         ├─► VAD speech detection
         ├─► Speaker identification
         ├─► Extract audio features
         │
         ▼
┌─────────────────┐
│ Enrollment      │ ← First 20 seconds only
│ Phase Check     │
└────────┬────────┘
         │
         ├─► If enrolling: Show overlay, skip visual
         ├─► If enrolled: Continue to visual processing
         │
         ▼
┌─────────────────┐
│ Person          │ ← YOLOv8 detection
│ Detection       │
└────────┬────────┘
         │
         ├─► Filter class=0 (person)
         ├─► Filter conf > 0.20
         │
         ▼
┌─────────────────┐
│ Multi-Object    │ ← SORT tracking
│ Tracking        │
└────────┬────────┘
         │
         ├─► Associate detections to tracks
         ├─► Assign student IDs
         ├─► Update Kalman filters
         │
         ▼
┌─────────────────┐
│ For Each        │ ← Loop over tracked students
│ Student         │
└────────┬────────┘
         │
         ├───────────────────────────────────┐
         │                                   │
         ▼                                   ▼
┌─────────────────┐              ┌──────────────────┐
│ Extract Visual  │              │ Get Audio        │
│ Features        │              │ Features         │
└────────┬────────┘              └────────┬─────────┘
         │                                │
         ├─► Gaze direction               ├─► Audio energy
         ├─► Eye openness (EAR)           ├─► Speech probability
         ├─► Mouth open (MAR)             ├─► Student noise detected
         ├─► Head pose                    ├─► Speaker count
         ├─► Movement                     └─► Audio engagement score
         │
         ▼
┌─────────────────┐
│ Compute Visual  │ ← Weighted fusion of features
│ Engagement      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Multimodal      │ ← Combine visual + audio
│ Fusion          │    (65% visual + 35% audio)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Temporal        │ ← Exponential moving average
│ Smoothing       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │ ← Draw bounding boxes, scores, labels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Log Data        │ ← Write to CSV
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Display Frame   │ ← cv2.imshow() (if not headless)
└────────┬────────┘
         │
         ├─► Check for key press (q=quit, s=screenshot)
         ├─► Update FPS counter
         ├─► Check max_frames limit
         │
         └─► Loop back to Read Frame
```

### 2.2 Loop Timing

```
Target FPS: 15-30 FPS
Frame Budget: 33-66ms per frame

Breakdown:
  Camera read:           ~2-5ms
  YOLOv8 detection:      ~8-15ms
  SORT tracking:         ~0.5ms
  Feature extraction:    ~15-30ms (per student)
  Fusion + smoothing:    ~0.1ms
  Visualization:         ~2-5ms
  Display + logging:     ~2-5ms
  ──────────────────────────────
  Total:                 ~30-60ms (17-33 FPS)

Audio processing: Parallel thread (no frame budget impact)
```

---

## 3. Visual Processing Pipeline

### 3.1 Detailed Visual Flow

```
┌──────────────────────────────────────────────────────────────┐
│              VISUAL PROCESSING PIPELINE                      │
└──────────────────────────────────────────────────────────────┘

INPUT: RGB Frame (1280×720×3)
  │
  ▼
┌─────────────────────────────────────┐
│  YOLOv8 Person Detection            │
│                                     │
│  1. Preprocess:                     │
│     - Resize to 640×640            │
│     - Normalize to [0,1]            │
│                                     │
│  2. Forward pass:                   │
│     - Backbone: CSPDarknet53        │
│     - Neck: PANet                   │
│     - Head: Detection head          │
│                                     │
│  3. Post-process:                   │
│     - NMS (IoU=0.45)               │
│     - Confidence filter (>0.20)     │
│     - Class filter (class=0)        │
└─────────────┬───────────────────────┘
              │
              │ OUTPUT: List of bboxes
              │ [{x1, y1, x2, y2, conf}, ...]
              │
              ▼
┌─────────────────────────────────────┐
│  SORT Multi-Object Tracking         │
│                                     │
│  For each existing track:           │
│    1. Predict next position         │
│       (Kalman filter)               │
│                                     │
│  2. Compute IoU matrix              │
│     (detections × tracks)           │
│                                     │
│  3. Hungarian assignment            │
│     (optimal matching)              │
│                                     │
│  4. Update matched tracks           │
│     (measurement update)            │
│                                     │
│  5. Create new tracks               │
│     (unmatched detections)          │
│                                     │
│  6. Delete old tracks               │
│     (age > max_age)                 │
└─────────────┬───────────────────────┘
              │
              │ OUTPUT: Tracked students
              │ [{x1,y1,x2,y2,id}, ...]
              │
              ▼
┌─────────────────────────────────────┐
│  FOR EACH STUDENT (Loop)            │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Crop Student ROI                   │
│                                     │
│  1. Extract bbox + padding          │
│  2. Handle boundary clipping        │
│  3. Resize if needed                │
└─────────────┬───────────────────────┘
              │
              │ OUTPUT: Student crop image
              │
              ▼
┌─────────────────────────────────────┐
│  MediaPipe Face Mesh                │
│  (or OpenCV Haar Cascade fallback)  │
│                                     │
│  1. Detect face in crop             │
│  2. Extract 468 landmarks           │
│  3. Normalize coordinates           │
└─────────────┬───────────────────────┘
              │
              │ OUTPUT: Face landmarks
              │
              ▼
┌─────────────────────────────────────┐
│  Feature Extraction                 │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 1. GAZE DIRECTION           │   │
│  │    - Use nose + eye corners │   │
│  │    - Classify: Forward/     │   │
│  │      Left/Right             │   │
│  │    - Output: Category       │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 2. EYE ASPECT RATIO (EAR)   │   │
│  │    - Compute eye opening    │   │
│  │    - Formula: vertical/     │   │
│  │      horizontal distance    │   │
│  │    - Output: Float [0,0.06] │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 3. MOUTH ASPECT RATIO (MAR) │   │
│  │    - Compute mouth opening  │   │
│  │    - Detect talking/yawning │   │
│  │    - Output: Float [0,0.1]  │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 4. HEAD POSE                │   │
│  │    - Compute pitch/yaw/roll │   │
│  │    - 3D landmark geometry   │   │
│  │    - Output: Angles (deg)   │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 5. MOVEMENT                 │   │
│  │    - Track bbox center      │   │
│  │    - Compute displacement   │   │
│  │    - Buffer last 5 frames   │   │
│  │    - Output: Pixels/frame   │   │
│  └─────────────────────────────┘   │
└─────────────┬───────────────────────┘
              │
              │ OUTPUT: Feature dict
              │ {gaze, eye, mouth, pose, movement}
              │
              ▼
┌─────────────────────────────────────┐
│  Normalize Features                 │
│                                     │
│  - Gaze: → {0.0, 0.2, 1.0}         │
│  - EAR: [0,0.06] → [0,1]           │
│  - MAR: [0,0.1] → [0,1]            │
│  - Pose: [-0.3,0.3] → [0,1]        │
│  - Movement: [0,50] → [0,1]        │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Compute Visual Engagement Score    │
│                                     │
│  Formula:                           │
│    score = 0.40 × gaze_score        │
│          + 0.25 × eye_score         │
│          + 0.20 × head_score        │
│          + 0.10 × movement_score    │
│          + 0.05 × mouth_score       │
│                                     │
│  Output: score ∈ [0, 1]            │
└─────────────┬───────────────────────┘
              │
              ▼
          Continue to Fusion
```

---

## 4. Audio Processing Pipeline

### 4.1 Detailed Audio Flow

```
┌──────────────────────────────────────────────────────────────┐
│              AUDIO PROCESSING PIPELINE                       │
│              (Parallel Thread)                               │
└──────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ Microphone      │ ← PyAudio stream
└────────┬────────┘
         │
         │ Continuous capture (16kHz, mono)
         │
         ▼
┌─────────────────────────────────────┐
│  Audio Capture Buffer               │
│                                     │
│  - Ring buffer (15 seconds)         │
│  - Chunk size: 8000 samples         │
│  - Duration: 500ms per chunk        │
│  - Format: float32 [-1, 1]          │
└─────────────┬───────────────────────┘
              │
              │ Every 500ms: New chunk
              │
              ▼
┌─────────────────────────────────────┐
│  PHASE 1: Teacher Enrollment        │
│  (First 10 seconds, 20 chunks)      │
│                                     │
│  1. Collect audio samples           │
│  2. Extract MFCC features           │
│     - 13 coefficients               │
│     - 25ms window, 10ms hop         │
│  3. Average MFCCs → teacher profile │
│  4. Store teacher embedding         │
│                                     │
│  Status: Enrollment progress 0-100% │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  PHASE 2: Baseline Establishment    │
│  (Next 10 seconds, 20 chunks)       │
│                                     │
│  1. Continue audio capture          │
│  2. Compute energy statistics       │
│     - baseline_mean                 │
│     - baseline_std                  │
│  3. Mark baseline complete          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  PHASE 3: Active Monitoring         │
│  (Remainder of session)             │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Voice Activity Detection (VAD)     │
│                                     │
│  1. Convert to torch tensor         │
│  2. Silero VAD model inference      │
│  3. Get speech probability [0,1]    │
│  4. Threshold: speech = prob > 0.5  │
│                                     │
│  Output: speech_detected (bool)     │
│          speech_prob (float)        │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Speaker Identification             │
│                                     │
│  1. Extract current MFCC features   │
│  2. Compare to teacher embedding    │
│  3. Compute cosine similarity       │
│  4. Classify:                       │
│     - similarity > 0.25: Teacher    │
│     - similarity ≤ 0.25: Student    │
│                                     │
│  Output: is_teacher (bool)          │
│          similarity (float)         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Speaker Count Estimation           │
│                                     │
│  1. Segment audio (50ms frames)     │
│  2. Compute energy per frame        │
│  3. Analyze variance + peaks        │
│  4. Estimate concurrent speakers    │
│                                     │
│  Output: speaker_count (int)        │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Audio Feature Extraction           │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 1. RMS ENERGY               │   │
│  │    energy = sqrt(mean(x²))  │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 2. ZERO-CROSSING RATE       │   │
│  │    ZCR = sign changes/len   │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 3. STUDENT NOISE DETECTION  │   │
│  │    if not teacher:          │   │
│  │      noise = True           │   │
│  │    noise_level = (E-base)/E │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ 4. TEMPORAL FEATURES        │   │
│  │    - avg_energy_recent      │   │
│  │    - energy_variance        │   │
│  │    - speech_activity_ratio  │   │
│  └─────────────────────────────┘   │
└─────────────┬───────────────────────┘
              │
              │ OUTPUT: Audio feature dict
              │
              ▼
┌─────────────────────────────────────┐
│  Compute Audio Engagement Score     │
│                                     │
│  base_score = 0.85                  │
│                                     │
│  if is_teacher and not noise:       │
│    score = 0.95  # Listening        │
│                                     │
│  elif student_noise_detected:       │
│    penalty = noise_level × 0.40     │
│    score = base_score - penalty     │
│    if speaker_count > 1:            │
│      score -= 0.15  # Side talk     │
│                                     │
│  else:                              │
│    score = 0.70  # Neutral          │
│                                     │
│  Output: audio_engagement ∈ [0,1]   │
└─────────────┬───────────────────────┘
              │
              ▼
          Continue to Fusion
```

---

## 5. Multimodal Fusion

### 5.1 Fusion Process

```
┌──────────────────────────────────────────────────────────────┐
│                  MULTIMODAL FUSION                           │
└──────────────────────────────────────────────────────────────┘

INPUT:
  - visual_score ∈ [0, 1]
  - audio_score ∈ [0, 1]

┌─────────────────────────────────────┐
│  Weighted Linear Combination        │
│                                     │
│  final_score = α × visual_score     │
│              + β × audio_score      │
│                                     │
│  Where:                             │
│    α = 0.65  (visual weight)        │
│    β = 0.35  (audio weight)         │
│    α + β = 1.0                      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Temporal Smoothing                 │
│                                     │
│  Method: Exponential Moving Average │
│                                     │
│  smoothed[t] = λ × final[t]         │
│              + (1-λ) × smoothed[t-1]│
│                                     │
│  Where: λ = 0.3                     │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Clamp to Valid Range               │
│                                     │
│  score = max(0.0, min(1.0, score))  │
└─────────────┬───────────────────────┘
              │
              ▼
OUTPUT: Final engagement score ∈ [0, 1]
```

### 5.2 Fusion Example

```
Example: Student looking forward, eyes open, quiet classroom

Visual Features:
  - gaze: "Forward" → 1.0
  - eye_openness: 0.048 → normalize(0.048, 0, 0.06) = 0.80
  - mouth_open: 0.02 → score = 1.0 (closed)
  - head_pitch: 0.05 → score = 0.75 (slight deviation)
  - movement: 5 pixels → normalize(5, 0, 50) = 0.90

Visual Score:
  = 0.40×1.0 + 0.25×0.80 + 0.20×0.75 + 0.10×0.90 + 0.05×1.0
  = 0.40 + 0.20 + 0.15 + 0.09 + 0.05
  = 0.89

Audio Features:
  - is_teacher: True
  - student_noise: False
  - speaker_count: 1
  
Audio Score:
  = 0.95  (teacher speaking, students listening)

Fusion:
  final = 0.65 × 0.89 + 0.35 × 0.95
        = 0.579 + 0.333
        = 0.912

Smoothing (assuming previous = 0.85):
  smoothed = 0.3 × 0.912 + 0.7 × 0.85
           = 0.274 + 0.595
           = 0.869

Final Engagement: 0.87 → "Highly Engaged"
```

---

## 6. Visualization and Logging

### 6.1 Visualization Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    VISUALIZATION                             │
└──────────────────────────────────────────────────────────────┘

INPUT: Frame, tracked_students, engagement_scores

┌─────────────────────────────────────┐
│  For Each Student                   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  1. Determine Color                 │
│                                     │
│     if score >= 0.80:               │
│       color = GREEN                 │
│     elif score >= 0.60:             │
│       color = LIGHT_GREEN           │
│     elif score >= 0.40:             │
│       color = YELLOW                │
│     elif score >= 0.20:             │
│       color = ORANGE                │
│     else:                           │
│       color = RED                   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  2. Draw Bounding Box               │
│                                     │
│     cv2.rectangle(frame,            │
│                   (x1, y1),         │
│                   (x2, y2),         │
│                   color,            │
│                   thickness=2)      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  3. Draw Label Background           │
│                                     │
│     cv2.rectangle(frame,            │
│                   (x1, y1-25),      │
│                   (x1+150, y1),     │
│                   color,            │
│                   thickness=-1)     │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  4. Draw Text Label                 │
│                                     │
│     text = f"ID:{id} {score:.2f}"   │
│     cv2.putText(frame, text,        │
│                 (x1+5, y1-8),       │
│                 font, 0.5,          │
│                 (255,255,255), 1)   │
└─────────────┬───────────────────────┘
              │
              ▼
          Next Student

┌─────────────────────────────────────┐
│  5. Draw Global Overlays            │
│                                     │
│  - FPS counter (top-left)           │
│  - Frame number                     │
│  - Student count                    │
│  - Average class engagement         │
│  - Audio status indicators          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  6. Display Frame                   │
│                                     │
│     cv2.imshow("Student Engagement",│
│                display_frame)       │
└─────────────────────────────────────┘
```

### 6.2 Logging Flow

```
┌──────────────────────────────────────────────────────────────┐
│                       DATA LOGGING                           │
└──────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────┐
│  For Each Student                   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Prepare Log Entry                  │
│                                     │
│  entry = {                          │
│    'timestamp': current_time,       │
│    'frame': frame_number,           │
│    'student_id': id,                │
│    'bbox_x': x1,                    │
│    'bbox_y': y1,                    │
│    'bbox_w': x2-x1,                 │
│    'bbox_h': y2-y1,                 │
│    'gaze': gaze_direction,          │
│    'eye_openness': ear_value,       │
│    'mouth_open': mar_value,         │
│    'head_pitch': pitch_angle,       │
│    'movement': movement_pixels,     │
│    'visual_score': visual_eng,      │
│    'audio_energy': audio_energy,    │
│    'speech_prob': speech_prob,      │
│    'student_noise': noise_detected, │
│    'audio_score': audio_eng,        │
│    'engagement_score': final_score  │
│  }                                  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Write to CSV                       │
│                                     │
│  datalog.log(entry)                 │
│                                     │
│  → Append to engagement_*.csv       │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Flush Buffer (every 30 rows)       │
│                                     │
│  Ensures data persisted to disk     │
└─────────────────────────────────────┘
```

**CSV Output Format**:
```csv
timestamp,frame,student_id,bbox_x,bbox_y,bbox_w,bbox_h,gaze,eye_openness,mouth_open,head_pitch,movement,visual_score,audio_energy,speech_prob,student_noise,audio_score,engagement_score
2026-01-14 10:30:15.123,1,0,120,80,180,320,Forward,0.048,0.02,0.05,5,0.89,0.012,0.85,False,0.95,0.91
2026-01-14 10:30:15.456,1,1,450,100,165,305,Left,0.035,0.04,0.12,8,0.62,0.012,0.85,False,0.95,0.72
...
```

---

## 7. System Shutdown

### 7.1 Shutdown Sequence

```
┌──────────────────────────────────────────────────────────────┐
│                    GRACEFUL SHUTDOWN                         │
└──────────────────────────────────────────────────────────────┘

Trigger: User presses 'q' or max_frames reached

Step 1: Exit Main Loop
  ├─ Stop reading new frames
  └─ Break from while loop

Step 2: Stop Audio Capture (if enabled)
  ├─ audio_capture.stop()
  ├─ Terminate audio thread
  └─ Release PyAudio resources

Step 3: Finalize Data Logger
  ├─ Flush remaining CSV buffer
  ├─ Compute session statistics
  ├─ Print summary:
  │  ├─ Total frames processed
  │  ├─ Total students detected
  │  ├─ Average engagement score
  │  ├─ Session duration
  │  └─ CSV file location
  └─ Close CSV file

Step 4: Release Video Capture
  ├─ cam.release()
  └─ Close camera/video stream

Step 5: Destroy Display Windows
  ├─ cv2.destroyAllWindows()
  └─ Free GUI resources

Step 6: Print Final Summary
  ├─ Display statistics
  ├─ Show output file paths
  └─ Exit message

Step 7: Exit Program
  └─ return or sys.exit(0)
```

### 7.2 Output Files

After shutdown, the following files are created:

```
student_engagement_refactor/
├── data/
│   └── labels/
│       └── engagement_20260114_103015.csv  ← Session data
├── results/ (if screenshots saved)
│   ├── screenshot_001.jpg
│   ├── screenshot_002.jpg
│   └── ...
└── logs/ (if logging enabled)
    └── engagement_20260114.log
```

---

## 8. Data Flow Diagrams

### 8.1 High-Level Data Flow

```
┌──────────┐        ┌──────────┐
│  Camera  │        │   Mic    │
└────┬─────┘        └────┬─────┘
     │ Frames            │ Audio chunks
     │ (30 FPS)          │ (2 Hz)
     │                   │
     ▼                   ▼
┌─────────┐        ┌──────────┐
│ YOLOv8  │        │ Silero   │
│ Detector│        │   VAD    │
└────┬────┘        └────┬─────┘
     │ Bboxes           │ Speech prob
     │                  │
     ▼                  ▼
┌─────────┐        ┌──────────┐
│  SORT   │        │ Speaker  │
│ Tracker │        │   ID     │
└────┬────┘        └────┬─────┘
     │ Tracked IDs      │ Teacher/Student
     │                  │
     ▼                  ▼
┌──────────────┐   ┌──────────┐
│  MediaPipe   │   │  Audio   │
│  Face Mesh   │   │ Features │
└──────┬───────┘   └────┬─────┘
       │                │
       │ Landmarks      │ Energy, noise
       │                │
       ▼                ▼
┌──────────────┐   ┌──────────┐
│   Visual     │   │  Audio   │
│  Features    │   │  Score   │
└──────┬───────┘   └────┬─────┘
       │                │
       ▼                ▼
┌──────────────┐   ┌──────────┐
│   Visual     │   │          │
│   Score      ├──►│  FUSION  │
└──────────────┘   └────┬─────┘
                        │
                        ▼
                ┌───────────────┐
                │  Engagement   │
                │    Score      │
                └───────┬───────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
      ┌───────────┐       ┌──────────┐
      │  Display  │       │   CSV    │
      │  (GUI)    │       │  Logger  │
      └───────────┘       └──────────┘
```

### 8.2 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                          main.py                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Main Processing Loop                        │   │
│  └──┬────────────────────────────────────────────┬─────┘   │
│     │                                            │          │
│     ▼                                            ▼          │
│  ┌──────────┐  ┌────────┐  ┌─────────┐  ┌───────────┐    │
│  │ capture  │  │  yolov │  │  sort   │  │  features │    │
│  │          │  │wrapper │  │ tracker │  │  visual   │    │
│  └──────────┘  └────────┘  └─────────┘  └───────────┘    │
│                                                             │
│  ┌──────────┐  ┌────────┐  ┌─────────┐  ┌───────────┐    │
│  │  audio   │  │  vad   │  │ speaker │  │   audio   │    │
│  │ capture  │  │detector│  │enrollment│  │  features │    │
│  └──────────┘  └────────┘  └─────────┘  └───────────┘    │
│                                                             │
│  ┌──────────────────────────┐  ┌──────────────────────┐   │
│  │     fusion.py            │  │   data_logger.py     │   │
│  │  multimodal_engagement   │  │      log()           │   │
│  └──────────────────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Timing and Performance

### 9.1 Frame Processing Breakdown

```
Typical Frame Processing Timeline (40ms total):

0ms  ┌─────────────────────────────────────────────────┐
     │ Read frame from camera                (2ms)    │
2ms  ├─────────────────────────────────────────────────┤
     │ YOLOv8 person detection              (12ms)    │
14ms ├─────────────────────────────────────────────────┤
     │ SORT tracking                        (0.5ms)   │
15ms ├─────────────────────────────────────────────────┤
     │ For each student (3 students):                 │
     │   - MediaPipe face mesh              (8ms)     │
     │   - Feature extraction               (2ms)     │
     │   - Visual score                     (0.1ms)   │
     │ Total:                               (30ms)    │
38ms ├─────────────────────────────────────────────────┤
     │ Multimodal fusion                    (0.2ms)   │
     │ Temporal smoothing                   (0.1ms)   │
     │ Visualization                        (3ms)     │
     │ Logging                              (1ms)     │
44ms └─────────────────────────────────────────────────┘

Result: ~22 FPS (44ms per frame)

Audio processing: Separate thread, no impact on frame rate
```

### 9.2 Performance Metrics

**System Requirements**:
- **CPU**: Intel i5 8th gen or equivalent
- **RAM**: 4GB minimum (8GB recommended)
- **Camera**: 720p webcam or higher
- **Microphone**: Any USB/built-in microphone

**Performance on Different Hardware**:

| Hardware | FPS | Students | Latency |
|----------|-----|----------|---------|
| i7 + GPU | 30+ | 10+ | 33ms |
| i5 CPU only | 20-25 | 5-8 | 50ms |
| i3 CPU only | 12-15 | 3-5 | 80ms |
| Raspberry Pi 4 | 5-8 | 2-3 | 150ms |

---

## 10. Error Handling and Edge Cases

### 10.1 Error Scenarios

**No Frame Captured**:
```python
if frame is None:
    logger.warning("No frame read, ending stream")
    break  # Exit gracefully
```

**Model Weights Missing**:
```python
if not os.path.exists(weights_path):
    logger.error(f"Model weights not found: {weights_path}")
    print("Download: https://github.com/ultralytics/assets/...")
    return  # Exit with error message
```

**No Students Detected**:
```python
if len(detections) == 0:
    # Continue to next frame
    # Display "No students detected"
    continue
```

**Audio Device Unavailable**:
```python
try:
    audio_capture = AudioCapture()
except Exception as e:
    logger.warning(f"Audio unavailable: {e}")
    use_audio = False  # Fall back to visual-only mode
```

**Face Not Detected**:
```python
if face_landmarks is None:
    # Use last known features or default neutral score
    features = last_features.get(student_id, default_features)
```

### 10.2 Edge Case Handling

**Student Leaves Frame**:
- SORT tracker maintains ID for `max_age=60` frames
- If returns within 60 frames, same ID retained
- Otherwise, new ID assigned on re-entry

**Occlusion**:
- Partial occlusion: Use visible features
- Full occlusion: Kalman filter prediction
- Re-appearance: Match to existing track by IoU

**Multiple Faces in One Bbox**:
- MediaPipe: Process largest face only
- Others ignored until separate bbox detected

---

## 11. Future Enhancements

### 11.1 Planned Improvements

1. **GPU Acceleration**
   - CUDA support for YOLOv8
   - Expected: 3-5x speedup (60+ FPS)

2. **DeepSORT Tracking**
   - Add appearance model for better re-ID
   - Reduce ID switches by 80%

3. **Attention Heatmap**
   - Spatial attention visualization
   - Show which areas students focus on

4. **Multi-Camera Support**
   - Process multiple camera streams
   - Merge detections for full classroom coverage

5. **Real-time Alerts**
   - Alert teacher if class engagement drops
   - Identify struggling students automatically

---

## Conclusion

This document provides a complete end-to-end flow of the Student Engagement Detection system. The pipeline combines real-time video processing, audio analysis, and multimodal fusion to assess student engagement continuously. The system is designed for efficiency, robustness, and scalability for classroom deployment.

---

**Last Updated**: January 2026
**Version**: 1.0
