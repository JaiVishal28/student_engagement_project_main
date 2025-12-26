#!/bin/bash
# Setup script for Student Engagement Detection System (Linux/Mac)
# Run this after cloning the repository: bash setup.sh

set -e  # Exit on error

echo "=========================================="
echo "Student Engagement Detection System"
echo "Linux/Mac Setup Script"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "[OK] Python is installed"
python3 --version
echo ""

# Create virtual environment
echo "[STEP 1/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "[OK] Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "[STEP 2/6] Activating virtual environment..."
source venv/bin/activate
echo "[OK] Virtual environment activated"
echo ""

# Upgrade pip
echo "[STEP 3/6] Upgrading pip..."
pip install --upgrade pip
echo ""

# Install core dependencies
echo "[STEP 4/6] Installing core dependencies..."
pip install -r requirements.txt
echo "[OK] Core dependencies installed"
echo ""

# Install extended dependencies (optional)
echo "[STEP 5/6] Installing extended dependencies for visualization..."
if [ -f "requirements_extended.txt" ]; then
    pip install -r requirements_extended.txt || echo "[WARNING] Some extended dependencies failed"
    echo "[OK] Extended dependencies installed"
else
    echo "[SKIP] requirements_extended.txt not found"
fi
echo ""

# Create necessary directories
echo "[STEP 6/6] Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/labels
mkdir -p models/weights
mkdir -p results/evaluation
mkdir -p results/visualizations
mkdir -p results/batch
mkdir -p images
echo "[OK] Directories created"
echo ""

# Check for model weights
echo "Checking for YOLOv8 model weights..."
if [ -f "models/weights/yolov8s.pt" ]; then
    echo "[OK] Model weights found"
else
    echo "[INFO] Model weights not found"
    echo ""
    echo "Attempting to download YOLOv8 weights..."
    if command -v wget &> /dev/null; then
        wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O models/weights/yolov8s.pt && echo "[OK] Model weights downloaded" || echo "[ERROR] Download failed"
    elif command -v curl &> /dev/null; then
        curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -o models/weights/yolov8s.pt && echo "[OK] Model weights downloaded" || echo "[ERROR] Download failed"
    else
        echo "[ERROR] Neither wget nor curl is available"
        echo "Please download manually:"
        echo "  URL: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        echo "  Save to: models/weights/yolov8s.pt"
    fi
fi
echo ""

# Generate synthetic test data
echo "Generating synthetic test data..."
if [ -f "scripts/generate_synthetic_data.py" ]; then
    python scripts/generate_synthetic_data.py && echo "[OK] Synthetic test data generated" || echo "[WARNING] Failed to generate synthetic data"
else
    echo "[SKIP] generate_synthetic_data.py not found"
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run: python -m src.main"
echo ""
echo "For help:"
echo "  python -m src.main --help"
echo ""
echo "Documentation:"
echo "  - README_PUBLICATION.md  (comprehensive guide)"
echo "  - SETUP.md              (detailed setup)"
echo "  - QUICK_REFERENCE.md    (command cheat sheet)"
echo "=========================================="
echo ""
