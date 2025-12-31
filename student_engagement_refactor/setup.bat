@echo off
REM Setup script for Student Engagement Detection System (Windows)
REM Run this after cloning the repository

echo ==========================================
echo Student Engagement Detection System
echo Windows Setup Script
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version
echo.

REM Create virtual environment
echo [STEP 1/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo [STEP 2/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [STEP 3/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install core dependencies
echo [STEP 4/6] Installing core dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Core dependencies installed
echo.

REM Install extended dependencies (optional)
echo [STEP 5/6] Installing extended dependencies for visualization...
if exist requirements_extended.txt (
    pip install -r requirements_extended.txt
    if errorlevel 1 (
        echo [WARNING] Some extended dependencies failed to install
        echo You can continue, but visualization features may not work
    ) else (
        echo [OK] Extended dependencies installed
    )
) else (
    echo [SKIP] requirements_extended.txt not found
)
echo.

REM Create necessary directories
echo [STEP 6/6] Creating project directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\labels mkdir data\labels
if not exist models\weights mkdir models\weights
if not exist results\evaluation mkdir results\evaluation
if not exist results\visualizations mkdir results\visualizations
if not exist results\batch mkdir results\batch
if not exist images mkdir images
echo [OK] Directories created
echo.

REM Check for model weights
echo Checking for YOLOv8 model weights...
if exist models\weights\yolov8s.pt (
    echo [OK] Model weights found
) else (
    echo [INFO] Model weights not found
    echo.
    echo You need to download YOLOv8 weights manually:
    echo   1. Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
    echo   2. Place in: models\weights\yolov8s.pt
    echo.
    echo Or run this command:
    echo   curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -o models\weights\yolov8s.pt
)
echo.

REM Generate synthetic test data
echo Generating synthetic test data...
if exist scripts\generate_synthetic_data.py (
    python scripts\generate_synthetic_data.py
    if errorlevel 1 (
        echo [WARNING] Failed to generate synthetic data
    ) else (
        echo [OK] Synthetic test data generated
    )
) else (
    echo [SKIP] generate_synthetic_data.py not found
)
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To get started:
echo   1. Keep this terminal open (virtual environment is active)
echo   2. Download model weights if not already done (see above)
echo   3. Run: python -m src.main
echo.
echo For help:
echo   python -m src.main --help
echo.
echo To activate the environment later:
echo   venv\Scripts\activate.bat
echo.
echo Documentation:
echo   - README_PUBLICATION.md  (comprehensive guide)
echo   - SETUP.md              (detailed setup)
echo   - QUICK_REFERENCE.md    (command cheat sheet)
echo ==========================================
echo.
pause
