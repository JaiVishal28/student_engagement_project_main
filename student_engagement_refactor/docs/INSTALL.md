# Quick Installation Guide

## For Windows

1. **Clone the repository** (or download and extract)
   ```powershell
   git clone <repository-url>
   cd student_engagement_refactor
   ```

2. **Run the setup script**
   ```powershell
   setup.bat
   ```

3. **Done!** The script will:
   - Create a virtual environment
   - Install all dependencies
   - Download model weights (if possible)
   - Create necessary directories
   - Generate test data

4. **Start using the system**
   ```powershell
   # Activate environment (if not already active)
   venv\Scripts\activate.bat
   
   # Run with webcam
   python -m src.main
   
   # Or process an image
   python -m src.main --image path\to\image.jpg
   ```

---

## For Linux/Mac

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd student_engagement_refactor
   ```

2. **Make setup script executable and run it**
   ```bash
   chmod +x setup.sh
   bash setup.sh
   ```

3. **Done!** The script will handle everything automatically.

4. **Start using the system**
   ```bash
   # Activate environment
   source venv/bin/activate
   
   # Run with webcam
   python -m src.main
   
   # Or process an image
   python -m src.main --image path/to/image.jpg
   ```

---

## Manual Installation (if scripts don't work)

### Step 1: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_extended.txt  # Optional, for visualization
```

### Step 3: Download Model Weights
Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

Save to: `models/weights/yolov8s.pt`

### Step 4: Create Directories
```bash
# Windows
mkdir data\raw data\processed data\labels models\weights results\evaluation results\visualizations results\batch images

# Linux/Mac
mkdir -p data/{raw,processed,labels} models/weights results/{evaluation,visualizations,batch} images
```

### Step 5: Test Installation
```bash
# Run tests
pytest tests/ -v

# Or try a quick demo
python -m src.main --help
```

---

## Troubleshooting

### Issue: "Python not found"
- **Windows**: Install from https://python.org, check "Add to PATH"
- **Linux**: `sudo apt install python3 python3-pip python3-venv`
- **Mac**: `brew install python3`

### Issue: "pip install fails"
- Update pip: `pip install --upgrade pip`
- Try with `--user` flag: `pip install --user -r requirements.txt`
- Check internet connection

### Issue: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Issue: "Permission denied" (Linux/Mac)
```bash
chmod +x setup.sh
# Or run with: bash setup.sh
```

### Issue: Model weights download fails
Download manually and place in `models/weights/yolov8s.pt`:
- Direct link: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

---

## Verify Installation

Run this command to check if everything is installed:
```python
python -c "import cv2, numpy, ultralytics, mediapipe, pandas, yaml; print('All dependencies OK!')"
```

If you see "All dependencies OK!", you're ready to go!

---

## What Gets Installed

### Core Dependencies (requirements.txt)
- opencv-python - Image/video processing
- numpy - Numerical operations
- ultralytics - YOLOv8 detection
- mediapipe - Facial landmark detection
- pandas - Data handling
- pyyaml - Configuration files
- scipy - Scientific computing
- filterpy - Kalman filtering for tracking
- matplotlib - Plotting
- seaborn - Statistical visualization
- tqdm - Progress bars

### Extended Dependencies (requirements_extended.txt)
- scikit-learn - Machine learning utilities
- pytest - Testing framework
- jupyterlab - Interactive notebooks

### Model Weights
- yolov8s.pt (~22 MB) - Pre-trained YOLOv8 small model

---

## Directory Structure After Installation

```
student_engagement_refactor/
├── venv/                    # Virtual environment (created)
├── data/
│   ├── raw/                 # (created - empty)
│   ├── processed/           # (created - empty)
│   └── labels/              # (created - with synthetic data)
├── models/
│   └── weights/
│       └── yolov8s.pt      # (downloaded)
├── results/
│   ├── evaluation/          # (created - empty)
│   ├── visualizations/      # (created - empty)
│   └── batch/               # (created - empty)
└── ... (other project files)
```

---

## Next Steps

1. **Read the documentation**
   - QUICK_REFERENCE.md - Common commands
   - README_PUBLICATION.md - Complete guide
   - SETUP.md - Detailed setup instructions

2. **Test the system**
   ```bash
   # Run with webcam
   python -m src.main
   
   # Process test image
   python scripts/generate_synthetic_data.py
   python -m src.main --image images/sample_class.jpg
   ```

3. **Start your project**
   - Collect video data
   - Annotate ground truth
   - Run experiments
   - Generate results

---

## Quick Commands

```bash
# Activate environment (do this first every time)
# Windows:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Basic usage
python -m src.main                          # Webcam mode
python -m src.main --source video.mp4       # Video mode
python -m src.main --image test.jpg         # Image mode

# Run tests
pytest tests/ -v

# Get help
python -m src.main --help
```

---

## Need Help?

- Check SETUP.md for detailed troubleshooting
- See QUICK_REFERENCE.md for command examples
- Read README_PUBLICATION.md for complete documentation

**Installation should take 5-10 minutes with good internet connection.**
