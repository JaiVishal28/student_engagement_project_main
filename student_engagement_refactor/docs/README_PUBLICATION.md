# Student Engagement Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

An automated real-time student engagement detection system using computer vision and deep learning. The system tracks multiple students simultaneously, analyzes visual cues (gaze direction, head pose, eye openness, movement), and computes engagement scores using multi-modal feature fusion.

### Key Features

- **Real-time Multi-Student Tracking**: YOLOv8-based person detection with SORT tracking
- **Visual Feature Extraction**: MediaPipe-based facial landmark detection for gaze, eye openness, mouth state, and head pose
- **Engagement Scoring**: Weighted feature fusion algorithm for engagement level estimation
- **Comprehensive Logging**: CSV-based data logging for analysis and validation
- **Evaluation Tools**: Metrics computation (precision, recall, F1, MAE, RMSE, correlation)
- **Visualization Suite**: Publication-quality plots and statistical analysis
- **Batch Processing**: Process multiple videos/images for dataset analysis

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [System Architecture](#system-architecture)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Publication](#publication)
9. [Contributing](#contributing)
10. [Citation](#citation)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time mode) or video files
- GPU (optional, for faster processing)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd student_engagement_refactor
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Basic requirements
pip install -r requirements.txt

# Extended requirements (for visualization and evaluation)
pip install -r requirements_extended.txt
```

### Step 4: Download Model Weights

Download YOLOv8 weights and place in `models/weights/`:

```bash
mkdir -p models/weights
cd models/weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
cd ../..
```

---

## Quick Start

### Real-time Webcam Mode

```bash
python -m src.main
```

Press `q` to quit, `s` to save screenshot.

### Process Video File

```bash
python -m src.main --source path/to/video.mp4
```

### Process Single Image

```bash
python -m src.main --image path/to/image.jpg
```

### Headless Mode (No Display)

```bash
python -m src.main --no-display --max-frames 1000
```

---

## System Architecture

### Pipeline Overview

```
Input Video/Image
    ↓
Person Detection (YOLOv8)
    ↓
Multi-Object Tracking (SORT)
    ↓
Visual Feature Extraction (MediaPipe)
    ├─ Gaze Direction
    ├─ Eye Openness
    ├─ Mouth State
    ├─ Head Pose
    └─ Movement Tracking
    ↓
Feature Fusion & Engagement Scoring
    ↓
Logging & Visualization
```

### Components

1. **Detection** (`src/detection/`): YOLOv8-based person detection
2. **Tracking** (`src/tracking/`): SORT algorithm for ID consistency
3. **Features** (`src/features/`): MediaPipe-based visual feature extraction
4. **Fusion** (`src/fusion/`): Weighted feature fusion for engagement scoring
5. **Evaluation** (`src/evaluation/`): Metrics computation and validation
6. **Logging** (`src/data_logger.py`): CSV-based data persistence

### Feature Weights

| Feature | Weight | Description |
|---------|--------|-------------|
| Gaze Direction | 0.40 | Forward gaze indicates attention |
| Eye Openness | 0.25 | Measures alertness |
| Head Pose | 0.20 | Upright posture indicates engagement |
| Movement | 0.10 | Excessive movement suggests distraction |
| Mouth State | 0.05 | Detects yawning/talking |

---

## Usage

### Command-Line Interface

```bash
# Full options
python -m src.main \
    --source 0 \              # Video source (0=webcam, or file path)
    --no-display \            # Run without GUI
    --max-frames 5000 \       # Limit processing
    --config custom.yaml      # Custom configuration
```

### Batch Processing

Process multiple videos:

```bash
python scripts/batch_process.py \
    /path/to/videos \
    --output-dir results/batch \
    --type video \
    --max-frames 3000
```

### Ground Truth Annotation

Create manual annotations for validation:

```bash
python scripts/annotate_ground_truth.py \
    video.mp4 \
    --output data/labels/ground_truth.csv
```

Controls:
- **Space**: Pause/resume
- **0-9**: Set engagement level (0=low, 9=high)
- **Q**: Quit and save

---

## Configuration

Edit `config.yaml` to customize system behavior:

```yaml
capture:
  source: 0          # 0 for webcam, or video path
  width: 1280
  height: 720
  fps: 15

detection:
  weights: "models/weights/yolov8s.pt"
  conf: 0.35         # Confidence threshold
  iou: 0.45          # IoU threshold
  device: "cpu"      # "cpu" or "cuda"

tracking:
  max_age: 30        # Frames before track deletion
  min_hits: 3        # Minimum detections for track
  iou_threshold: 0.3

processing:
  process_every_n_frames: 3  # Skip frames for speed
  face_padding: 12
  pose_padding: 12

logging:
  csv_path: "data/labels/engagement_data.csv"
```

---

## Evaluation

### Compute Metrics

```bash
python scripts/evaluate_model.py \
    --predictions data/labels/engagement_data.csv \
    --labels data/labels/ground_truth.csv \
    --output results/evaluation/metrics.csv
```

**Output Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Pearson Correlation
- Precision, Recall, F1-Score
- Accuracy

### Generate Visualizations

```bash
python scripts/visualize_results.py \
    --data data/labels/engagement_data.csv \
    --output-dir results/visualizations \
    --plots timeline distribution correlation summary
```

**Generated Plots:**
- `engagement_timeline.png`: Time-series visualization
- `engagement_distribution.png`: Score distribution histograms
- `feature_correlation.png`: Correlation heatmap
- `summary_statistics.csv`: Statistical summary

---

## Results

### Sample Performance Metrics

| Metric | Value |
|--------|-------|
| Precision | 0.87 |
| Recall | 0.82 |
| F1-Score | 0.84 |
| MAE | 0.12 |
| RMSE | 0.18 |
| Correlation | 0.79 |

*Note: These are example values. Run evaluation on your dataset for actual results.*

### Processing Speed

- **Detection**: ~30 FPS (CPU), ~120 FPS (GPU)
- **Feature Extraction**: ~25 FPS
- **Overall Pipeline**: ~15-20 FPS

---

## Publication

### Methodology

This system implements a multi-modal approach to student engagement detection:

1. **Detection & Tracking**: YOLOv8 for robust person detection, SORT for temporal consistency
2. **Feature Extraction**: MediaPipe Face Mesh (468 landmarks) and Pose (33 keypoints)
3. **Engagement Modeling**: Weighted fusion of visual cues based on educational psychology literature
4. **Validation**: Ground truth comparison with human annotations

### Key Contributions

- Real-time multi-student engagement monitoring
- Non-invasive computer vision approach
- Interpretable feature-based scoring
- Comprehensive evaluation framework
- Open-source implementation for reproducibility

### Recommended Sections for Paper

1. **Introduction**: Importance of engagement monitoring in education
2. **Related Work**: Existing engagement detection systems
3. **Methodology**: Detailed system architecture and algorithms
4. **Experiments**: Dataset description, evaluation protocol
5. **Results**: Quantitative metrics, visualizations, ablation studies
6. **Discussion**: Limitations, ethical considerations, future work
7. **Conclusion**: Summary and impact

### Data Collection Guidelines

For publication-quality results:

1. **Dataset Size**: Minimum 50 hours of annotated video
2. **Diversity**: Multiple classrooms, lighting conditions, student demographics
3. **Annotation**: At least 2 independent annotators with inter-rater reliability > 0.75
4. **Ground Truth**: Frame-level engagement labels (discrete or continuous)
5. **Ethics**: IRB approval, informed consent, anonymization

---

## Project Structure

```
student_engagement_refactor/
├── config.yaml                 # Configuration file
├── requirements.txt            # Core dependencies
├── requirements_extended.txt   # Additional dependencies
├── README.md                   # This file
├── METHODOLOGY.md              # Detailed methodology for paper
├── Dockerfile                  # Docker containerization
├── start.sh                    # Quick start script
│
├── data/
│   ├── raw/                    # Raw videos/images
│   ├── processed/              # Preprocessed data
│   └── labels/                 # CSV annotations
│
├── images/                     # Sample images
│
├── models/
│   ├── weights/                # Model weights (yolov8s.pt)
│   └── README.md               # Model documentation
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_visualization.ipynb
│   └── 03_results_analysis.ipynb
│
├── results/
│   ├── evaluation/             # Evaluation metrics
│   ├── visualizations/         # Generated plots
│   └── batch/                  # Batch processing results
│
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── evaluate_model.py
│   ├── visualize_results.py
│   ├── batch_process.py
│   └── annotate_ground_truth.py
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── capture.py              # Camera interface
│   ├── logging_utils.py        # Logging utilities
│   ├── data_logger.py          # Data persistence
│   │
│   ├── detection/              # Detection module
│   │   ├── __init__.py
│   │   └── yolov_wrapper.py
│   │
│   ├── tracking/               # Tracking module
│   │   ├── __init__.py
│   │   └── sort_tracker.py
│   │
│   ├── features/               # Feature extraction
│   │   ├── __init__.py
│   │   └── visual_features.py
│   │
│   ├── fusion/                 # Feature fusion
│   │   ├── __init__.py
│   │   └── fusion.py
│   │
│   └── evaluation/             # Evaluation module
│       ├── __init__.py
│       └── metrics.py
│
└── tests/                      # Unit tests
    ├── test_detection.py
    ├── test_tracking.py
    ├── test_features.py
    └── test_fusion.py
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features

---

## Citation

If you use this system in your research, please cite:

```bibtex
@article{yourname2024engagement,
  title={Real-Time Student Engagement Detection Using Multi-Modal Computer Vision},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  volume={X},
  pages={XXX-XXX}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- YOLOv8 by Ultralytics
- MediaPipe by Google
- SORT tracking algorithm by Alex Bewley
- Educational psychology research on engagement indicators

---

## Contact

For questions or collaborations:
- **Email**: your.email@institution.edu
- **Project Page**: https://your-project-page.com
- **Issues**: https://github.com/yourusername/repo/issues

---

## Troubleshooting

### Common Issues

**Issue**: "Model weights not found"
- **Solution**: Download yolov8s.pt and place in `models/weights/`

**Issue**: "No module named 'mediapipe'"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Low FPS on CPU
- **Solution**: 
  - Increase `process_every_n_frames` in config.yaml
  - Use GPU by setting `device: cuda` in config.yaml
  - Reduce input resolution

**Issue**: No webcam detected
- **Solution**: Check camera permissions, try different source numbers (0, 1, 2)

---

## Roadmap

### Future Enhancements

- [ ] Deep learning-based engagement classification (CNN/RNN)
- [ ] Audio analysis integration (voice tone, participation)
- [ ] Multi-camera fusion
- [ ] Real-time alerts for instructors
- [ ] Dashboard for engagement analytics
- [ ] Mobile app deployment
- [ ] Privacy-preserving federated learning

---

## Version History

- **v1.0.0** (2024-12-26): Initial release with core functionality
  - YOLOv8 detection
  - SORT tracking
  - MediaPipe features
  - Weighted fusion scoring
  - Evaluation framework
  - Visualization tools

---

**Last Updated**: December 26, 2024
