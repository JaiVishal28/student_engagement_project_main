# Setup and Installation Guide

## Quick Start (5 minutes)

### Windows

```powershell
# 1. Clone repository
git clone <repository-url>
cd student_engagement_refactor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights
mkdir models\weights
cd models\weights
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -o yolov8s.pt
cd ..\..

# 5. Run demo
python -m src.main
```

### Linux/Mac

```bash
# 1. Clone repository
git clone <repository-url>
cd student_engagement_refactor

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights
mkdir -p models/weights
cd models/weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
cd ../..

# 5. Run demo
python -m src.main
```

## Detailed Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Webcam (optional, for real-time demo)
- GPU with CUDA (optional, for faster processing)

### Step-by-Step

#### 1. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

**Windows:**
- Install Python from https://www.python.org/downloads/
- Ensure "Add Python to PATH" is checked during installation

**macOS:**
```bash
brew install python3
```

#### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate (choose based on OS)
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate
```

#### 3. Install Python Packages

```bash
# Core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Extended dependencies (optional, for visualization)
pip install -r requirements_extended.txt
```

#### 4. Download Model Weights

**Option A: Automatic (Linux/Mac/Git Bash on Windows)**
```bash
bash start.sh
```

**Option B: Manual**
1. Download: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
2. Place in: `models/weights/yolov8s.pt`

#### 5. Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Run tests (optional)
pytest tests/ -v
```

## Docker Installation

### Build Docker Image

```bash
docker build -t student-engagement .
```

### Run Docker Container

```bash
# Process video file
docker run -v /path/to/videos:/data student-engagement \
    python -m src.main --source /data/video.mp4 --no-display

# Batch processing
docker run -v /path/to/data:/data student-engagement \
    python scripts/batch_process.py /data/videos --output-dir /data/results
```

## GPU Support (Optional)

### Install CUDA

1. Check GPU compatibility: https://developer.nvidia.com/cuda-gpus
2. Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
3. Install cuDNN: https://developer.nvidia.com/cudnn

### Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Update Config

Edit `config.yaml`:
```yaml
detection:
  device: "cuda"  # Change from "cpu" to "cuda"
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Shows GPU name
```

## Troubleshooting

### Issue: "No module named 'cv2'"

**Solution:**
```bash
pip install opencv-python
```

### Issue: "ImportError: libGL.so.1"

**Solution (Linux):**
```bash
sudo apt-get install libgl1-mesa-glx
```

### Issue: "YOLOv8 weights not found"

**Solution:**
```bash
mkdir -p models/weights
cd models/weights
# Download manually and place here
```

### Issue: Slow FPS on CPU

**Solutions:**
1. Increase `process_every_n_frames` in config.yaml to 5 or 10
2. Reduce input resolution
3. Use GPU (see GPU Support section)

### Issue: Webcam not detected

**Solutions:**
1. Check camera permissions in OS settings
2. Try different source numbers:
```bash
python -m src.main --source 0  # Try 0, 1, 2, etc.
```
3. List available cameras:
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
```

### Issue: "Permission denied" on Linux

**Solution:**
```bash
chmod +x start.sh
sudo usermod -a -G video $USER  # Add user to video group
# Log out and log back in
```

## Development Setup

### Additional Tools

```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 pylint

# Type checking
pip install mypy

# Documentation
pip install sphinx sphinx-rtd-theme
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fusion.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Configuration

### Basic Configuration

Edit `config.yaml`:

```yaml
capture:
  source: 0          # Webcam ID or video path
  width: 1280        # Resolution width
  height: 720        # Resolution height
  fps: 15           # Target FPS

detection:
  weights: "models/weights/yolov8s.pt"
  conf: 0.35         # Lower = more detections, more false positives
  iou: 0.45          # IoU threshold for NMS
  device: "cpu"      # "cpu" or "cuda"

tracking:
  max_age: 30        # Frames before track deletion
  min_hits: 3        # Minimum detections to confirm track
  iou_threshold: 0.3 # IoU for track association

processing:
  process_every_n_frames: 3  # Process every Nth frame (higher = faster, less accurate)

logging:
  csv_path: "data/labels/engagement_data.csv"
```

### Performance Tuning

**For Speed:**
- Increase `process_every_n_frames` to 5-10
- Reduce input resolution (width/height)
- Use GPU (`device: "cuda"`)
- Lower detection confidence (`conf: 0.4`)

**For Accuracy:**
- Process every frame (`process_every_n_frames: 1`)
- Higher resolution
- Lower confidence threshold (`conf: 0.25`)
- Use larger YOLOv8 model (yolov8m.pt or yolov8l.pt)

## Next Steps

1. **Test the System**: Run on sample video or webcam
2. **Collect Data**: Record classroom sessions with consent
3. **Create Ground Truth**: Use annotation tool to label data
4. **Evaluate**: Run evaluation scripts to compute metrics
5. **Visualize**: Generate publication-quality plots
6. **Iterate**: Adjust weights and parameters based on results

## Additional Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **MediaPipe Guide**: https://google.github.io/mediapipe/
- **OpenCV Tutorials**: https://docs.opencv.org/master/d9/df8/tutorial_root.html
- **Project Issues**: https://github.com/yourusername/repo/issues

## Support

For questions or issues:
- Check documentation in `README_PUBLICATION.md` and `METHODOLOGY.md`
- Search existing issues on GitHub
- Create new issue with system info and error logs
- Contact: your.email@institution.edu
