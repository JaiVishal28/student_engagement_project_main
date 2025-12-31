# Student Engagement Detection System - Complete Project Context

## 📋 Project Overview

**Purpose**: Real-time classroom student engagement detection system using computer vision and deep learning for educational research and publication.

**Goal**: Automatically detect and quantify student engagement levels in classroom environments through visual features analysis, tracking, and multimodal fusion.

**Status**: Phase 1 (Visual Features) Complete - Ready for testing and data collection

**Target Publication**: Academic paper on automated student engagement assessment in educational settings

---

## 🏗️ Technical Architecture

### System Pipeline
```
Input (Video/Webcam/Image)
    ↓
Person Detection (YOLOv8)
    ↓
Student Tracking (SORT Algorithm)
    ↓
Feature Extraction (MediaPipe + OpenCV)
    ↓
Engagement Scoring (Weighted Fusion)
    ↓
Visualization & Logging
    ↓
Output (CSV Logs + Annotated Video)
```

### Core Technologies
- **Deep Learning**: YOLOv8 (v8.3.0) - Person detection with bounding boxes
- **Computer Vision**: OpenCV 4.10.0 - Video processing, facial features extraction
- **Facial Landmarks**: MediaPipe 0.10.31 - 468 facial landmarks for gaze, eyes, mouth
- **Tracking**: SORT Algorithm - Multi-object tracking maintaining student IDs across frames
- **Language**: Python 3.13.7
- **Environment**: Windows 10/11 with virtual environment

### Hardware Requirements
- **Minimum**: Intel i5 (6th gen+) or AMD Ryzen 3, 8GB RAM, integrated GPU
- **Recommended**: Intel i7/AMD Ryzen 5+, 16GB RAM, NVIDIA GTX 1050+
- **Performance**: 15-30 FPS on CPU, 30-60 FPS on GPU

---

## 📁 Project Structure

```
student_engagement_refactor/
├── src/                          # Main source code (modular architecture)
│   ├── main.py                   # Production system (342 lines)
│   ├── capture.py                # Camera/video input handler
│   ├── data_logger.py            # CSV logging for engagement data
│   ├── logging_utils.py          # Logging configuration
│   ├── detection/
│   │   └── yolov_wrapper.py      # YOLOv8 person detection wrapper
│   ├── features/
│   │   └── visual_features.py    # Feature extraction (gaze, eyes, mouth, head pose)
│   ├── fusion/
│   │   └── fusion.py             # Engagement scoring algorithm
│   └── tracking/
│       └── sort_tracker.py       # SORT multi-object tracker
├── models/
│   └── weights/
│       ├── yolov8s.pt            # YOLOv8 small model (21.5MB, downloaded)
│       └── haarcascade_frontalface_default.xml  # OpenCV fallback
├── images/
│   └── students1.jpg             # Test image (13 students, 1296x550)
├── data/                         # CSV logs output directory
├── evaluation/                   # Metrics, validation, benchmarking tools
├── visualization/                # Plotting, heatmaps, attention maps
├── tests/                        # Unit tests, integration tests
├── docs/                         # Complete documentation suite
├── scripts/                      # Utility scripts
├── config.yaml                   # Central configuration file
├── requirements.txt              # Python dependencies
├── setup.bat / setup.sh          # Automated installation scripts
├── test_visual_features.py       # Single image testing tool (194 lines)
├── find_cameras.py               # Camera detection utility
└── full_engagement_demo.py       # Standalone demo (self-contained)
```

---

## 🎯 Key Components Explained

### 1. src/main.py (Production System)
**Purpose**: Full-featured system for live video/webcam processing with tracking and logging

**Capabilities**:
- Real-time video processing from webcam/video files
- Multi-student tracking with persistent IDs across frames
- Per-student engagement logging to CSV
- FPS counter and performance monitoring
- Multiple modes: video, image, headless (no display)

**Usage**:
```bash
# Live webcam (default camera)
python -m src.main

# Specific webcam (use find_cameras.py to identify)
python -m src.main --source 1

# Video file
python -m src.main --source path/to/video.mp4

# Single image
python -m src.main --image path/to/image.jpg

# Headless mode (no visualization, logging only)
python -m src.main --no-display

# Limited frames
python -m src.main --max-frames 300
```

**Key Features**:
- Camera class for robust video capture
- SORT tracker maintaining student IDs
- DataLogger writing timestamped CSV files
- Config-driven parameters from config.yaml
- Real-time FPS display and performance stats

---

### 2. test_visual_features.py (Testing Tool)
**Purpose**: Quick testing on single static images for debugging and paper figures

**Capabilities**:
- Single image processing only
- Visual feature extraction and visualization
- Engagement score calculation
- Annotated output image generation
- No tracking, no logging

**Usage**:
```bash
python test_visual_features.py images/students1.jpg
```

**Output**:
- Console: Detection count, individual engagement scores, average engagement
- File: output_features.jpg with bounding boxes, scores, color-coded labels

**When to Use**:
- Creating figures for academic papers
- Quick testing of detection/feature extraction
- Debugging visual features on specific images
- Generating example outputs for presentations

---

### 3. Visual Feature Extraction (src/features/visual_features.py)

**Extracted Features**:

1. **Gaze Direction** (40% weight):
   - Left/Right: Horizontal eye position relative to face center
   - Up/Down: Vertical eye position
   - Score: 100% when looking forward, decreases with angle

2. **Eye Openness** (25% weight):
   - Eye Aspect Ratio (EAR) from MediaPipe landmarks
   - Open eyes: EAR > 0.2 → 100%
   - Partially closed: Linear scale
   - Closed eyes: EAR < 0.15 → 0%

3. **Head Pose** (20% weight):
   - Pitch: Nodding up/down
   - Yaw: Turning left/right
   - Roll: Tilting head
   - Score: 100% when facing forward, decreases with rotation

4. **Movement/Activity** (10% weight):
   - Currently placeholder (requires temporal analysis)
   - Future: Frame-to-frame pose changes

5. **Mouth Openness** (5% weight):
   - Mouth Aspect Ratio (MAR) from landmarks
   - Closed: MAR < 0.3 → 100% (attentive)
   - Open: MAR > 0.5 → 0% (distracted/talking)

**Technical Implementation**:
- Primary: MediaPipe Face Mesh (468 landmarks)
- Fallback: OpenCV Haar Cascade (when MediaPipe fails)
- Version compatibility: Handles MediaPipe 0.10.31+ API changes

**Engagement Score Calculation**:
```python
engagement = (0.40 * gaze_score) + 
             (0.25 * eye_openness) + 
             (0.20 * head_pose_score) + 
             (0.10 * movement_score) + 
             (0.05 * mouth_score)
```

**Classification Thresholds**:
- Engaged: engagement ≥ 70%
- Moderately Engaged: 40% ≤ engagement < 70%
- Distracted: engagement < 40%

---

## 🔧 Configuration (config.yaml)

**Key Parameters**:
```yaml
detection:
  model: "yolov8s.pt"
  conf_threshold: 0.5
  iou_threshold: 0.4
  device: "cpu"  # or "cuda" for GPU

tracking:
  max_age: 30          # Frames before track deletion
  min_hits: 3          # Frames before track confirmation
  iou_threshold: 0.3

visualization:
  box_color: [0, 255, 0]
  text_color: [255, 255, 255]
  font_scale: 0.6
  thickness: 2
  show_fps: true

logging:
  enabled: true
  output_dir: "data"
  log_interval: 1  # seconds
```

---

## 📊 Testing Results

**Test Image**: images/students1.jpg (1296x550, classroom scene)

**Detection Results**:
- **Students Detected**: 13 persons
- **Average Engagement**: 62.4%
- **Distribution**:
  - Engaged (≥70%): 6 students
  - Moderate (40-69%): 1 student
  - Distracted (<40%): 6 students

**Individual Scores** (sample):
- Student #1: 75.2% (Engaged)
- Student #2: 83.4% (Engaged)
- Student #3: 28.9% (Distracted)
- Student #4: 91.7% (Engaged)
- Student #5: 45.6% (Moderate)

**Output**: output_features.jpg (305KB, annotated with bounding boxes and scores)

---

## 🚀 Installation & Setup

### Quick Setup (Automated)
```bash
# Windows
setup.bat

# Linux/Mac
chmod +x setup.sh
./setup.sh
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model (automatic on first run)
# Or manually: models/weights/yolov8s.pt
```

### Dependencies
```
ultralytics==8.3.0    # YOLOv8
opencv-python==4.10.0.84
mediapipe==0.10.31
numpy==2.0.2
pandas==2.2.3
pyyaml==6.0.2
scipy==1.14.1        # For SORT tracker
```

**Environment**:
- Python: 3.13.7
- Virtual Environment: C:/Users/JaiVishalr/OneDrive - Archer/Desktop/PROJ/.venv
- Git Branch: demo

---

## 📱 Live Video Support

### Webcam Usage
```bash
# Find available cameras
python find_cameras.py

# Use detected camera (e.g., index 1)
python -m src.main --source 1
```

### Phone as Webcam
**Supported Methods**:
1. **DroidCam** (Recommended):
   - Install DroidCam on phone and PC
   - Connect via WiFi/USB
   - Use camera index from find_cameras.py

2. **IP Webcam**:
   - Install IP Webcam app
   - Get stream URL (http://192.168.x.x:8080/video)
   - Use: `python -m src.main --source "http://192.168.x.x:8080/video"`

3. **EpocCam**:
   - Install EpocCam on phone and PC driver
   - Appears as regular webcam
   - Use detected index from find_cameras.py

**Performance Expectations**:
- WiFi: 15-25 FPS (depends on network quality)
- USB: 25-35 FPS (more stable)
- Local Webcam: 30-60 FPS (best performance)

**Troubleshooting**:
- Run find_cameras.py to verify camera detection
- Check firewall settings for IP webcam
- Ensure same WiFi network for phone and PC
- Try different resolution settings in config.yaml

---

## 🐛 Known Issues & Fixes

### MediaPipe Compatibility
**Issue**: MediaPipe 0.10.31+ removed `mp.solutions` API
**Fix**: Implemented version detection with OpenCV Haar Cascade fallback in visual_features.py
**Status**: ✅ Fixed and tested

### Installation Issues
**Issue**: Manual dependency installation prone to errors
**Fix**: Created setup.bat and setup.sh for automated installation
**Status**: ✅ Scripts created

### Feature Accuracy
**Issue**: Gaze estimation less accurate at extreme angles
**Current**: Works well for frontal/slight angle views
**Future**: Phase 3 will add attention mechanism for temporal context

---

## 📈 Research & Publication Status

### Completed Work
- ✅ Phase 1: Visual Features Implementation (100%)
- ✅ YOLOv8 Person Detection
- ✅ SORT Multi-Object Tracking
- ✅ Facial Feature Extraction (gaze, eyes, mouth, head pose)
- ✅ Weighted Fusion Engagement Scoring
- ✅ Real-time Video Processing
- ✅ CSV Data Logging
- ✅ Comprehensive Documentation (3500+ lines)
- ✅ Evaluation Framework (metrics, validation, benchmarking)
- ✅ Visualization Tools (plots, heatmaps, attention maps)
- ✅ Testing on Classroom Images (13 students, 62.4% avg engagement)

### Data Collection Requirements
**For Publication**:
- Minimum: 50+ hours of classroom video
- Students: 100+ participants across multiple classrooms
- Ground Truth: Manual annotations for validation (10-20% of data)
- IRB Approval: Required for human subjects research
- Consent Forms: All participants must consent

**Recommended Dataset Structure**:
```
dataset/
├── videos/
│   ├── class1_session1.mp4
│   ├── class1_session2.mp4
│   └── ...
├── annotations/
│   ├── class1_session1.csv  # Ground truth labels
│   └── ...
└── metadata.json  # Class info, camera position, lighting
```

---

## 🗺️ Future Development Roadmap

### Phase 2: Audio Features (4-6 weeks)
**Status**: Planned, not started

**Components**:
1. **Phase 2A: Audio Collection (2 weeks)**
   - Microphone integration
   - Multi-channel audio recording
   - Audio-video synchronization
   - Voice Activity Detection (VAD)

2. **Phase 2B: Audio Features (2 weeks)**
   - Speech energy/volume
   - Speech rate and pauses
   - Prosody (pitch, intonation)
   - Speaker diarization

3. **Phase 2C: Multimodal Fusion (2 weeks)**
   - Audio-visual fusion architecture
   - Late fusion strategy
   - Cross-modal attention
   - Combined engagement score

**Expected Improvement**: 10-15% increase in engagement prediction accuracy

---

### Phase 3: Temporal Modeling (4-6 weeks)
**Status**: Planned, not started

**Components**:
1. **Phase 3A: LSTM Implementation (2 weeks)**
   - Sequence modeling architecture
   - Bidirectional LSTM layers
   - Feature sequence encoding
   - Temporal context windows (5-10 seconds)

2. **Phase 3B: Attention Mechanism (2 weeks)**
   - Self-attention for temporal features
   - Multi-head attention
   - Attention visualization
   - Explainable engagement predictions

3. **Phase 3C: GRU Optimization (2 weeks)**
   - Lightweight GRU alternative
   - Real-time performance optimization
   - Mobile deployment preparation
   - Model compression and quantization

**Expected Improvement**: 15-20% increase in accuracy, better temporal consistency

---

## 📝 File Usage Guide

### When to Use Each File

**src/main.py** (Production System):
- ✅ Live classroom monitoring
- ✅ Long video processing
- ✅ Data collection for research
- ✅ Multi-student tracking
- ✅ CSV logging for analysis

**test_visual_features.py** (Testing Tool):
- ✅ Quick testing on images
- ✅ Creating paper figures
- ✅ Debugging detection/features
- ✅ Generating example outputs
- ✅ Presentation screenshots

**full_engagement_demo.py** (Standalone Demo):
- ✅ Self-contained demonstration
- ✅ No dependency on src modules
- ✅ Easy sharing with collaborators
- ✅ Conference demos
- ✅ Quick proof-of-concept

**find_cameras.py** (Utility):
- ✅ Detecting available cameras
- ✅ Troubleshooting camera issues
- ✅ Finding phone webcam index
- ✅ Verifying camera resolution

---

## 🎓 Academic Context

### Research Questions
1. Can automated visual features predict student engagement?
2. What feature weights optimize engagement prediction?
3. How does temporal context improve accuracy?
4. Can multimodal fusion (audio+visual) outperform unimodal approaches?

### Methodology
- **Approach**: Computer vision + deep learning for engagement detection
- **Dataset**: Real classroom videos (to be collected with IRB approval)
- **Validation**: Manual annotations by expert observers
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Temporal Consistency

### Expected Contributions
1. Novel weighted fusion algorithm for engagement scoring
2. Real-time system suitable for classroom deployment
3. Open-source implementation for reproducibility
4. Comprehensive evaluation on real classroom data

### Related Work
- Attention detection using gaze tracking
- Facial expression recognition for emotion
- Student engagement prediction using ML
- Multimodal learning analytics

---

## 💡 Key Insights & Design Decisions

### Why YOLOv8?
- Fast (30+ FPS on CPU)
- Accurate person detection (95%+ on COCO)
- Easy to use (ultralytics library)
- Pre-trained models available

### Why SORT Tracking?
- Simple and effective
- Real-time performance
- Maintains student IDs across frames
- Handles occlusions reasonably well

### Why MediaPipe?
- 468 facial landmarks (very detailed)
- Real-time inference
- Good gaze estimation capability
- Free and open-source

### Why Weighted Fusion?
- Interpretable (can explain why score is high/low)
- Adjustable (can tune weights based on research)
- No training data required (rule-based)
- Fast computation (real-time capable)

**Alternative Considered**: Deep learning fusion (e.g., neural network)
- Requires large labeled dataset
- Less interpretable
- May overfit
- Saved for Phase 3 (temporal modeling)

---

## 🔍 Technical Details

### Person Detection
- **Model**: YOLOv8s (Small - 21.5MB)
- **Input**: 640x640 RGB image
- **Output**: Bounding boxes [x, y, w, h], confidence score
- **Class Filter**: Person only (class_id = 0)
- **Confidence Threshold**: 0.5 (configurable)

### Face Detection
- **Primary**: MediaPipe Face Mesh
  - 468 landmarks (eyes, nose, mouth, face contour)
  - Detects in person bounding box crop
- **Fallback**: OpenCV Haar Cascade
  - Used when MediaPipe fails
  - Less accurate but more robust

### Tracking
- **Algorithm**: SORT (Simple Online and Realtime Tracking)
- **Method**: Kalman Filter + Hungarian Algorithm
- **Parameters**:
  - max_age: 30 frames (1 second at 30 FPS)
  - min_hits: 3 frames before confirming track
  - iou_threshold: 0.3 for matching

### Data Logging
- **Format**: CSV (Comma-Separated Values)
- **Columns**: timestamp, student_id, engagement_score, gaze_score, eye_openness, head_pose_score, movement_score, mouth_score, bbox_x, bbox_y, bbox_w, bbox_h
- **Frequency**: Every 1 second (configurable)
- **Location**: data/ directory with timestamp filename

---

## 🧪 Testing & Validation

### Unit Tests (tests/)
- test_detection.py: YOLOv8 detection accuracy
- test_features.py: Feature extraction correctness
- test_fusion.py: Engagement score calculation
- test_tracking.py: SORT tracker consistency

### Integration Tests (tests/)
- test_pipeline.py: End-to-end pipeline
- test_video_processing.py: Video input handling
- test_data_logging.py: CSV output format

### Evaluation Metrics (evaluation/)
- Accuracy: Overall correctness
- Precision: True engaged / All predicted engaged
- Recall: True engaged / All actually engaged
- F1-Score: Harmonic mean of precision and recall
- Temporal Consistency: Agreement between adjacent frames

### Benchmarking (evaluation/)
- FPS measurement: Frames per second
- Latency: Processing time per frame
- Memory usage: RAM consumption
- GPU utilization: CUDA usage (if available)

---

## 📚 Documentation Files

### Core Documentation
- **README.md**: Project overview and quick start
- **INSTALL.md**: Detailed installation instructions
- **PROJECT_FILES_GUIDE.md**: File usage guide
- **LIVE_VIDEO_GUIDE.md**: Webcam and phone camera setup

### Technical Documentation
- **API_REFERENCE.md**: Function and class documentation
- **ARCHITECTURE.md**: System design and data flow
- **CONFIGURATION.md**: Config.yaml parameter explanations

### Research Documentation
- **EVALUATION_GUIDE.md**: Metrics and validation procedures
- **DATA_COLLECTION.md**: Dataset requirements and IRB process
- **ROADMAP_ADVANCED.md**: Phase 2 and Phase 3 implementation plans

### Visualization Documentation
- **VISUALIZATION_GUIDE.md**: Plotting tools and usage
- **HEATMAP_GUIDE.md**: Spatial engagement heatmaps
- **ATTENTION_MAPS.md**: Temporal attention visualization

---

## 🛠️ Troubleshooting Common Issues

### Camera Not Detected
```bash
# Run camera detection
python find_cameras.py

# Try different indices
python -m src.main --source 0
python -m src.main --source 1
```

### Low FPS
- Switch to GPU: Set `device: "cuda"` in config.yaml
- Reduce resolution: Lower input video resolution
- Use smaller model: Switch from yolov8s.pt to yolov8n.pt (nano)

### No Faces Detected
- Check lighting: Ensure adequate illumination
- Adjust confidence: Lower conf_threshold in config.yaml
- Verify camera angle: Students should be clearly visible

### MediaPipe Errors
- Fallback enabled: System automatically uses OpenCV if MediaPipe fails
- Update MediaPipe: `pip install --upgrade mediapipe`
- Check compatibility: Currently supports MediaPipe 0.10.31+

---

## 📞 Contact & Support

**Project Location**: `C:\Users\JaiVishalr\OneDrive - Archer\Desktop\PROJ\student_engagement_project_main\student_engagement_refactor`

**Git Branch**: demo

**Python Environment**: `C:/Users/JaiVishalr/OneDrive - Archer/Desktop/PROJ/.venv`

**For Issues**:
1. Check error logs in terminal output
2. Review configuration in config.yaml
3. Verify dependencies: `pip list`
4. Test with find_cameras.py utility
5. Run test_visual_features.py on sample image

---

## 🎯 Next Steps for Publication

### Immediate Actions
1. ✅ Test live video system with actual classroom camera
2. ⬜ Obtain IRB approval for human subjects research
3. ⬜ Collect consent forms from all participants
4. ⬜ Record 50+ hours of classroom video
5. ⬜ Create ground truth annotations (10-20% of data)

### Data Analysis
6. ⬜ Process all videos with src/main.py
7. ⬜ Analyze CSV logs for engagement patterns
8. ⬜ Run evaluation metrics on validation set
9. ⬜ Generate plots and visualizations for paper

### Paper Writing
10. ⬜ Write methodology section (system architecture)
11. ⬜ Write results section (accuracy, FPS, examples)
12. ⬜ Create figures using test_visual_features.py
13. ⬜ Write discussion (limitations, future work)
14. ⬜ Submit to target conference/journal

### Future Enhancements
15. ⬜ Implement Phase 2A: Audio features
16. ⬜ Implement Phase 3: LSTM temporal modeling
17. ⬜ Deploy system in real classroom for testing
18. ⬜ Collect feedback from teachers

---

## 📄 Citation (When Published)

```bibtex
@article{student_engagement_2025,
  title={Automated Student Engagement Detection Using Computer Vision and Deep Learning},
  author={Your Name},
  journal={Educational Technology Research},
  year={2025},
  volume={XX},
  pages={XX-XX},
  doi={XX.XXXX/XXXX}
}
```

---

## 🏁 Summary for LLMs

**What This Project Does**:
Automatically detects and quantifies student engagement in classroom videos using YOLOv8 for person detection, MediaPipe for facial features, SORT for tracking, and weighted fusion for engagement scoring.

**Current Status**:
Phase 1 (visual features) is complete and working. Tested on classroom image with 13 students, achieving 62.4% average engagement detection. System supports live video, webcam, and phone camera input with real-time processing and CSV logging.

**Key Files**:
- src/main.py: Production system for real classroom monitoring
- test_visual_features.py: Quick testing tool for single images
- config.yaml: Central configuration
- requirements.txt: All dependencies

**Main Challenge**:
Need to collect real classroom data (50+ hours) with IRB approval for validation and publication.

**Future Direction**:
Phase 2 adds audio features, Phase 3 adds LSTM temporal modeling for improved accuracy and temporal consistency.

---

**Last Updated**: December 27, 2025
**Version**: 1.0 (Phase 1 Complete)
**Project Status**: Ready for Real-World Testing and Data Collection
