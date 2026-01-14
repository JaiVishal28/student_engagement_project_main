# Documentation Index

Welcome to the Student Engagement Detection System documentation. This index provides a comprehensive guide to all documentation files organized by category.

---

## 📚 Quick Start

New to the project? Start here:

1. **[README.md](../README.md)** - Project overview and quick start guide
2. **[INSTALL.md](INSTALL.md)** - Detailed installation instructions
3. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Common commands and usage patterns
4. **[SETUP.md](SETUP.md)** - Environment setup guide

---

## 🎯 Core Documentation

### Technical Specifications

- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** - Complete technical report covering architecture, methodology, performance, and deployment
- **[MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md)** - Detailed documentation of all models (YOLOv8, MediaPipe, Silero VAD, SORT)
- **[MULTIMODAL_METHODOLOGY.md](MULTIMODAL_METHODOLOGY.md)** - In-depth explanation of multimodal approach, fusion techniques, and algorithms
- **[COMPLETE_PROJECT_FLOW.md](COMPLETE_PROJECT_FLOW.md)** - End-to-end system flow from initialization to shutdown
- **[COMPONENT_API.md](COMPONENT_API.md)** - Complete API documentation for all modules with usage examples

### Project Understanding

- **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)** - Project background, motivation, and objectives
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - High-level summary of features and improvements
- **[PROJECT_FILES_GUIDE.md](PROJECT_FILES_GUIDE.md)** - Guide to project structure and file organization

---

## 🔧 Setup & Configuration

### Installation & Setup

- **[INSTALL.md](INSTALL.md)** - Step-by-step installation guide
  - Python environment setup
  - Dependency installation
  - Model weight download
  - Troubleshooting common issues

- **[SETUP.md](SETUP.md)** - System configuration
  - Hardware requirements
  - Software prerequisites
  - Configuration file (config.yaml) guide

- **[AUDIO_SETUP.md](AUDIO_SETUP.md)** - Audio system configuration
  - Microphone setup
  - Teacher enrollment process
  - Audio troubleshooting

### Usage Guides

- **[LIVE_VIDEO_GUIDE.md](LIVE_VIDEO_GUIDE.md)** - Guide for live video processing
  - Webcam usage
  - IP camera streams
  - Video file processing
  - Real-time visualization

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference
  - Common CLI commands
  - Configuration snippets
  - Troubleshooting tips

---

## 📊 Models & Algorithms

### Model Documentation

**[MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md)** - Comprehensive model documentation:

1. **Computer Vision Models**
   - YOLOv8 (Person Detection)
     - Architecture details
     - Performance metrics
     - Configuration options
   - MediaPipe Face Mesh
     - 468 facial landmarks
     - Feature extraction methods
   - OpenCV Haar Cascade (Fallback)
     - Face detection
     - Heuristic feature estimation

2. **Audio Processing Models**
   - Silero VAD (Voice Activity Detection)
     - Architecture and specifications
     - Real-time speech detection
   - Speaker Enrollment System
     - MFCC-based speaker identification
     - Teacher voice profile

3. **Tracking Algorithm**
   - SORT (Simple Online Realtime Tracking)
     - Kalman filter predictions
     - Hungarian assignment
     - Track management

4. **Feature Extraction Models**
   - Gaze estimation
   - Eye Aspect Ratio (EAR)
   - Mouth Aspect Ratio (MAR)
   - Head pose estimation

### Methodology Documentation

**[MULTIMODAL_METHODOLOGY.md](MULTIMODAL_METHODOLOGY.md)** - Detailed methodology:

1. **System Architecture**
   - Multimodal pipeline overview
   - Data flow timing

2. **Visual Modality**
   - Person detection pipeline
   - Multi-object tracking
   - Facial feature extraction
   - Visual engagement scoring

3. **Audio Modality**
   - Audio capture and processing
   - Teacher enrollment
   - Voice activity detection
   - Audio engagement scoring

4. **Multimodal Fusion**
   - Late fusion strategy
   - Weighted combination (65/35)
   - Temporal smoothing
   - Decision logic

5. **Engagement Scoring Algorithm**
   - Complete scoring pipeline
   - Engagement levels
   - Per-student vs class-level metrics

6. **Temporal Analysis**
   - Short/medium/long-term features
   - Moving window analysis
   - Event detection

7. **Performance Optimization**
   - Processing efficiency
   - Multimodal synchronization

8. **Evaluation Metrics**
   - Modality-specific metrics
   - Fusion metrics
   - Temporal consistency

---

## 🔄 System Flow

**[COMPLETE_PROJECT_FLOW.md](COMPLETE_PROJECT_FLOW.md)** - Complete system flow:

1. **System Initialization**
   - Startup sequence
   - Configuration loading
   - Model initialization
   - Teacher enrollment

2. **Runtime Processing Loop**
   - Main loop overview
   - Frame-by-frame processing
   - Loop timing

3. **Visual Processing Pipeline**
   - Person detection
   - Tracking
   - Feature extraction
   - Visual scoring

4. **Audio Processing Pipeline**
   - Audio capture
   - VAD detection
   - Speaker identification
   - Audio scoring

5. **Multimodal Fusion**
   - Fusion process
   - Example calculations
   - Decision logic

6. **Visualization and Logging**
   - Real-time display
   - CSV data logging

7. **System Shutdown**
   - Graceful shutdown
   - Output files

8. **Data Flow Diagrams**
   - High-level data flow
   - Component interactions

9. **Timing and Performance**
   - Frame processing breakdown
   - Performance metrics

---

## 💻 API Reference

**[COMPONENT_API.md](COMPONENT_API.md)** - Complete API documentation:

### Modules Covered:

1. **Detection Module** (`src/detection/yolov_wrapper.py`)
   - `YoloDetector` class
   - `detect()` method

2. **Tracking Module** (`src/tracking/sort_tracker.py`)
   - `Sort` class
   - `Track` class
   - `update()` method

3. **Features Module** (`src/features/visual_features.py`)
   - `extract_all()` function
   - `extract_features_opencv()` fallback
   - Helper functions

4. **Fusion Module** (`src/fusion/fusion.py`)
   - `simple_engagement_score()` function
   - `multimodal_engagement_score()` function
   - `compute_engagement_score()` function

5. **Audio Modules** (`src/audio/`)
   - `AudioCapture` class
   - `VADDetector` class
   - `AudioFeatureExtractor` class
   - `SpeakerEnrollment` class

6. **Capture Module** (`src/capture.py`)
   - `Camera` class

7. **Data Logger Module** (`src/data_logger.py`)
   - `DataLogger` class

8. **Evaluation Module** (`src/evaluation/metrics.py`)
   - Regression metrics
   - Classification metrics

Each section includes:
- Class/function signatures
- Parameters and return types
- Usage examples
- Performance notes

---

## 📈 Research & Publication

### Publication Materials

- **[README_PUBLICATION.md](README_PUBLICATION.md)** - Publication-ready project description
- **[PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md)** - Checklist for preparing academic publications
- **[METHODOLOGY_OLD.md](METHODOLOGY_OLD.md)** - Original methodology documentation (legacy)

### Future Development

- **[ROADMAP_ADVANCED.md](ROADMAP_ADVANCED.md)** - Advanced feature roadmap
  - Planned enhancements
  - Research directions
  - Timeline

---

## 🎓 For Your Final Report

### Recommended Reading Order for Report Preparation:

1. **Understanding the Project**
   - [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) - Background and motivation
   - [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) - Complete technical overview

2. **Models & Techniques**
   - [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md) - All models in detail
   - [MULTIMODAL_METHODOLOGY.md](MULTIMODAL_METHODOLOGY.md) - Multimodal approach

3. **System Implementation**
   - [COMPLETE_PROJECT_FLOW.md](COMPLETE_PROJECT_FLOW.md) - System flow
   - [COMPONENT_API.md](COMPONENT_API.md) - Component details

4. **Performance & Results**
   - [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) Section 5 - Performance Analysis
   - [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Improvements and optimizations

### Key Sections to Include in Your Report:

**Introduction**:
- Use [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for problem statement
- Reference [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) Section 1

**Literature Review**:
- Model references from [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md)
- Methodology papers from [MULTIMODAL_METHODOLOGY.md](MULTIMODAL_METHODOLOGY.md)

**Methodology**:
- System architecture from [MULTIMODAL_METHODOLOGY.md](MULTIMODAL_METHODOLOGY.md) Section 1
- Algorithms from [MULTIMODAL_METHODOLOGY.md](MULTIMODAL_METHODOLOGY.md) Sections 2-5
- Flow diagrams from [COMPLETE_PROJECT_FLOW.md](COMPLETE_PROJECT_FLOW.md)

**Implementation**:
- Component details from [COMPONENT_API.md](COMPONENT_API.md)
- System flow from [COMPLETE_PROJECT_FLOW.md](COMPLETE_PROJECT_FLOW.md)

**Models & Techniques**:
- [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md) - All models
  - YOLOv8 architecture and performance
  - MediaPipe Face Mesh specifications
  - SORT tracking algorithm
  - Silero VAD details
  - Speaker enrollment system
- [MULTIMODAL_METHODOLOGY.md](MULTIMODAL_METHODOLOGY.md) - Fusion techniques
  - Visual feature extraction
  - Audio feature extraction
  - Multimodal fusion strategy

**Results**:
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) Section 5 - Performance Analysis
  - Computational performance
  - Accuracy evaluation
  - Modality contributions
  - Scalability analysis

**Discussion**:
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) Section 8 - Limitations & Future Work

**Conclusion**:
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) Section 10

---

## 📁 Document Organization

```
docs/
├── README_DOCS.md                     ← This file (Documentation Index)
├── TECHNICAL_REPORT.md                ← Complete technical report
├── MODELS_DOCUMENTATION.md            ← All models in detail
├── MULTIMODAL_METHODOLOGY.md          ← Multimodal approach & algorithms
├── COMPLETE_PROJECT_FLOW.md           ← End-to-end system flow
├── COMPONENT_API.md                   ← API documentation
├── INSTALL.md                         ← Installation guide
├── SETUP.md                           ← Configuration guide
├── AUDIO_SETUP.md                     ← Audio setup guide
├── LIVE_VIDEO_GUIDE.md                ← Video processing guide
├── QUICK_REFERENCE.md                 ← Quick command reference
├── PROJECT_CONTEXT.md                 ← Project background
├── PROJECT_SUMMARY.md                 ← Feature summary
├── PROJECT_FILES_GUIDE.md             ← File structure guide
├── README_PUBLICATION.md              ← Publication-ready description
├── PUBLICATION_CHECKLIST.md           ← Publication checklist
├── ROADMAP_ADVANCED.md                ← Future development roadmap
└── METHODOLOGY_OLD.md                 ← Legacy methodology (backup)
```

---

## 🔍 Finding Specific Information

### Looking for Installation Help?
→ [INSTALL.md](INSTALL.md) + [SETUP.md](SETUP.md)

### Want to understand the models?
→ [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md)

### Need to understand multimodal fusion?
→ [MULTIMODAL_METHODOLOGY.md](MULTIMODAL_METHODOLOGY.md)

### Want the complete system flow?
→ [COMPLETE_PROJECT_FLOW.md](COMPLETE_PROJECT_FLOW.md)

### Need API documentation for coding?
→ [COMPONENT_API.md](COMPONENT_API.md)

### Preparing a final report or publication?
→ [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

### Quick command reference?
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

---

## 📝 Additional Resources

### Scripts Documentation

All scripts in `scripts/` directory are documented with:
- Purpose and usage
- Command-line arguments
- Example commands
- Output formats

Key scripts:
- `evaluate_model.py` - Model evaluation
- `visualize_results.py` - Data visualization
- `generate_report.py` - Report generation
- `batch_process.py` - Batch video processing
- `annotate_ground_truth.py` - Ground truth annotation

### Configuration Files

- `config.yaml` - Main configuration (documented in [SETUP.md](SETUP.md))
- `requirements.txt` - Python dependencies
- `requirements_extended.txt` - Extended dependencies for development

---

## 🤝 Contributing

When adding new documentation:
1. Follow existing format and structure
2. Update this index with new document
3. Cross-reference related documents
4. Include code examples where applicable
5. Add to appropriate category

---

## 📧 Support

For questions or issues:
1. Check relevant documentation above
2. Review [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common issues
3. Consult [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for comprehensive details

---

**Last Updated**: January 14, 2026  
**Documentation Version**: 1.0  
**Total Documents**: 19

---

*Navigate to the [Main README](../README.md) to get started with the project.*
