# =============================================================================
# setup_new_system.ps1
# Full-system bootstrapper for Student Engagement Detection System
#
# USAGE (on a brand-new Windows machine):
#   1. Install Python 3.10 from https://www.python.org  (check "Add to PATH")
#   2. Install Git from https://git-scm.com
#   3. Open PowerShell and run:
#        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#        .\setup_new_system.ps1
#   OR double-click "setup_new_system.bat" — it does step 3 automatically.
# =============================================================================

param(
    [string]$RepoURL    = "https://github.com/JaiVishal28/student_engagement_project_main",
    [string]$InstallDir = ""          # Leave blank to install in current directory
)

$ProjectSubDir  = "student_engagement_refactor"   # sub-folder inside the repo
$VenvName       = ".venv"
$WeightsURL     = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
$WeightsRelPath = "models\weights\yolov8s.pt"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-Step  { param($n,$msg) Write-Host "`n[STEP $n] $msg" -ForegroundColor Cyan }
function Write-OK    { param($msg)    Write-Host "  [OK] $msg"      -ForegroundColor Green }
function Write-Warn  { param($msg)    Write-Host "  [WARN] $msg"    -ForegroundColor Yellow }
function Write-Fail  { param($msg)    Write-Host "  [ERROR] $msg"   -ForegroundColor Red; exit 1 }

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  Student Engagement Detection System  -  System Setup"      -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

# ── STEP 1: Check Git ─────────────────────────────────────────────────────────
Write-Step 1 "Checking Git installation"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail "Git is not installed or not in PATH.`n  Download from: https://git-scm.com/download/win"
}
$gitVer = (git --version)
Write-OK "Git found: $gitVer"

# ── STEP 2: Check Python ──────────────────────────────────────────────────────
Write-Step 2 "Checking Python installation (3.8+ required)"
$pythonCmd = $null
foreach ($cmd in @("python", "python3")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 8) { $pythonCmd = $cmd; break }
        }
    }
}
if (-not $pythonCmd) {
    Write-Fail "Python 3.8+ not found.`n  Download from: https://www.python.org/downloads/`n  IMPORTANT: check 'Add Python to PATH' during install."
}
Write-OK "Python found: $(& $pythonCmd --version)"

# ── STEP 3: Resolve install directory ────────────────────────────────────────
Write-Step 3 "Resolving installation directory"
if ($InstallDir -eq "") { $InstallDir = (Get-Location).Path }
$repoName  = ($RepoURL -split "/")[-1]
$cloneTarget = Join-Path $InstallDir $repoName

Write-Host "  Repo will be cloned to: $cloneTarget" -ForegroundColor White

# ── STEP 4: Clone repository ──────────────────────────────────────────────────
Write-Step 4 "Cloning repository"
if (Test-Path $cloneTarget) {
    Write-Warn "Directory '$cloneTarget' already exists. Pulling latest changes instead."
    Push-Location $cloneTarget
    git pull
    Pop-Location
} else {
    git clone $RepoURL $cloneTarget
    if ($LASTEXITCODE -ne 0) { Write-Fail "git clone failed." }
    Write-OK "Repository cloned successfully."
}

# ── STEP 5: Navigate to the project sub-directory ─────────────────────────────
Write-Step 5 "Navigating to project directory"
$projectPath = Join-Path $cloneTarget $ProjectSubDir
if (-not (Test-Path $projectPath)) {
    Write-Fail "Expected sub-directory '$ProjectSubDir' not found inside cloned repo."
}
Set-Location $projectPath
Write-OK "Working directory: $(Get-Location)"

# ── STEP 6: Create virtual environment ────────────────────────────────────────
Write-Step 6 "Creating virtual environment ($VenvName)"
if (Test-Path $VenvName) {
    Write-Warn "Virtual environment already exists — reusing it."
} else {
    & $pythonCmd -m venv $VenvName
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create virtual environment." }
    Write-OK "Virtual environment created."
}

# ── STEP 7: Upgrade pip ───────────────────────────────────────────────────────
Write-Step 7 "Upgrading pip"
$pipExe = Join-Path $projectPath "$VenvName\Scripts\pip.exe"
& $pipExe install --upgrade pip --quiet
Write-OK "pip upgraded."

# ── STEP 8: Install core dependencies ─────────────────────────────────────────
Write-Step 8 "Installing core dependencies (requirements.txt)"
& $pipExe install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Fail "Core dependency installation failed." }
Write-OK "Core dependencies installed."

# ── STEP 9: Install extended dependencies ────────────────────────────────────
Write-Step 9 "Installing extended dependencies (requirements_extended.txt)"
if (Test-Path "requirements_extended.txt") {
    & $pipExe install -r requirements_extended.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Some extended dependencies failed — visualization features may be limited."
    } else {
        Write-OK "Extended dependencies installed."
    }
} else {
    Write-Warn "requirements_extended.txt not found — skipping."
}

# ── STEP 10: PyAudio Windows fix ─────────────────────────────────────────────
Write-Step 10 "Checking PyAudio (Windows audio)"
$pyaudioCheck = & $pipExe show pyaudio 2>&1
if ($pyaudioCheck -match "Name: PyAudio") {
    Write-OK "PyAudio already installed."
} else {
    Write-Host "  Attempting PyAudio install (may need Microsoft C++ Build Tools)..." -ForegroundColor Yellow
    & $pipExe install pyaudio 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Direct PyAudio install failed. Trying pipwin fallback..."
        & $pipExe install pipwin --quiet
        & $pipExe run pipwin install pyaudio 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "PyAudio could not be installed automatically.`n  Audio features will be disabled.`n  To fix manually: install from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio"
        } else {
            Write-OK "PyAudio installed via pipwin."
        }
    } else {
        Write-OK "PyAudio installed."
    }
}

# ── STEP 11: Create required directories ─────────────────────────────────────
Write-Step 11 "Creating project directories"
$dirs = @(
    "data\raw", "data\processed", "data\labels",
    "models\weights",
    "results\evaluation", "results\visualizations", "results\batch",
    "images"
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}
Write-OK "Directories ready."

# ── STEP 12: Download YOLOv8s weights ────────────────────────────────────────
Write-Step 12 "Checking YOLOv8s model weights"
if (Test-Path $WeightsRelPath) {
    Write-OK "Weights already present at $WeightsRelPath"
} else {
    Write-Host "  Downloading yolov8s.pt from Ultralytics (~22 MB)..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri $WeightsURL -OutFile $WeightsRelPath -UseBasicParsing
        Write-OK "Weights downloaded to $WeightsRelPath"
    } catch {
        Write-Warn "Auto-download failed. Download manually and place at: $WeightsRelPath`n  URL: $WeightsURL"
    }
}

# ── STEP 13: Quick import smoke-test ─────────────────────────────────────────
Write-Step 13 "Running smoke test"
$pythonExe = Join-Path $projectPath "$VenvName\Scripts\python.exe"
$smokeTest = @"
import sys
failures = []
pkgs = ["cv2", "numpy", "ultralytics", "mediapipe", "pandas", "yaml", "scipy", "filterpy", "matplotlib", "torch"]
for p in pkgs:
    try:
        __import__(p)
    except ImportError:
        failures.append(p)
if failures:
    print("MISSING: " + ", ".join(failures))
    sys.exit(1)
else:
    print("ALL OK")
"@
$result = & $pythonExe -c $smokeTest 2>&1
if ($result -match "ALL OK") {
    Write-OK "All core packages import successfully."
} else {
    Write-Warn "Some packages failed to import: $result`n  You can still try running the project — optional packages may be missing."
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  SETUP COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  To run the project:" -ForegroundColor White
Write-Host "    cd `"$projectPath`""  -ForegroundColor Yellow
Write-Host "    .\.venv\Scripts\activate"  -ForegroundColor Yellow
Write-Host "    python src\main.py"  -ForegroundColor Yellow
Write-Host ""
Write-Host "  For the full demo:"  -ForegroundColor White
Write-Host "    python full_engagement_demo.py"  -ForegroundColor Yellow
Write-Host ""
Write-Host "  Config file: config.yaml  (edit camera source, thresholds, etc.)" -ForegroundColor White
Write-Host ""
