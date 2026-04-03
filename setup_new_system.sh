#!/usr/bin/env bash
# =============================================================================
# setup_new_system.sh
# Full-system bootstrapper for Student Engagement Detection System (Linux/Mac)
#
# USAGE (on a brand-new Linux or macOS machine):
#   chmod +x setup_new_system.sh
#   ./setup_new_system.sh
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/JaiVishal28/student_engagement_project_main"
PROJECT_SUBDIR="student_engagement_refactor"
VENV_NAME=".venv"
WEIGHTS_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
WEIGHTS_PATH="models/weights/yolov8s.pt"

# ── Helpers ───────────────────────────────────────────────────────────────────
step()  { echo -e "\n\033[0;36m[STEP $1] $2\033[0m"; }
ok()    { echo -e "  \033[0;32m[OK] $*\033[0m"; }
warn()  { echo -e "  \033[0;33m[WARN] $*\033[0m"; }
fail()  { echo -e "  \033[0;31m[ERROR] $*\033[0m"; exit 1; }

echo ""
echo "============================================================"
echo "  Student Engagement Detection System  -  System Setup"
echo "============================================================"
echo ""

# ── STEP 1: Check Git ─────────────────────────────────────────────────────────
step 1 "Checking Git"
if ! command -v git &>/dev/null; then
    fail "Git not found. Install with:\n  Ubuntu/Debian: sudo apt install git\n  macOS: brew install git"
fi
ok "Git: $(git --version)"

# ── STEP 2: Check Python 3.8+ ────────────────────────────────────────────────
step 2 "Checking Python (3.8+ required)"
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
            PYTHON_CMD="$cmd"; break
        fi
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    fail "Python 3.8+ not found.\n  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip\n  macOS: brew install python@3.11"
fi
ok "Python: $($PYTHON_CMD --version)"

# ── STEP 3: Install system audio libraries (Linux only) ──────────────────────
step 3 "Checking system audio libraries (Linux only)"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! dpkg -l portaudio19-dev &>/dev/null 2>&1; then
        warn "portaudio19-dev not found — attempting install (requires sudo)."
        sudo apt-get install -y portaudio19-dev python3-pyaudio || warn "Could not install portaudio. Audio features may not work."
    else
        ok "portaudio19-dev already installed."
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if ! brew list portaudio &>/dev/null 2>&1; then
        warn "portaudio not found — attempting install via Homebrew."
        brew install portaudio || warn "Could not install portaudio. Audio features may not work."
    else
        ok "portaudio already installed."
    fi
fi

# ── STEP 4: Clone repository ──────────────────────────────────────────────────
step 4 "Cloning repository"
REPO_NAME=$(basename "$REPO_URL")
CLONE_TARGET="./$REPO_NAME"

if [ -d "$CLONE_TARGET" ]; then
    warn "Directory '$CLONE_TARGET' already exists — pulling latest changes."
    git -C "$CLONE_TARGET" pull
else
    git clone "$REPO_URL" "$CLONE_TARGET"
    ok "Repository cloned successfully."
fi

# ── STEP 5: Navigate to project sub-directory ────────────────────────────────
step 5 "Navigating to project directory"
PROJECT_PATH="$CLONE_TARGET/$PROJECT_SUBDIR"
if [ ! -d "$PROJECT_PATH" ]; then
    fail "Expected sub-directory '$PROJECT_SUBDIR' not found inside cloned repo."
fi
cd "$PROJECT_PATH"
ok "Working directory: $(pwd)"

# ── STEP 6: Create virtual environment ────────────────────────────────────────
step 6 "Creating virtual environment ($VENV_NAME)"
if [ -d "$VENV_NAME" ]; then
    warn "Virtual environment already exists — reusing it."
else
    $PYTHON_CMD -m venv "$VENV_NAME"
    ok "Virtual environment created."
fi

# Activate
# shellcheck disable=SC1090
source "$VENV_NAME/bin/activate"

# ── STEP 7: Upgrade pip ───────────────────────────────────────────────────────
step 7 "Upgrading pip"
pip install --upgrade pip --quiet
ok "pip upgraded."

# ── STEP 8: Install core dependencies ─────────────────────────────────────────
step 8 "Installing core dependencies (requirements.txt)"
pip install -r requirements.txt
ok "Core dependencies installed."

# ── STEP 9: Install extended dependencies ────────────────────────────────────
step 9 "Installing extended dependencies (requirements_extended.txt)"
if [ -f "requirements_extended.txt" ]; then
    pip install -r requirements_extended.txt || warn "Some extended dependencies failed — visualization may be limited."
    ok "Extended dependencies installed."
else
    warn "requirements_extended.txt not found — skipping."
fi

# ── STEP 10: Create required directories ──────────────────────────────────────
step 10 "Creating project directories"
mkdir -p data/raw data/processed data/labels \
         models/weights \
         results/evaluation results/visualizations results/batch \
         images
ok "Directories ready."

# ── STEP 11: Download YOLOv8s weights ────────────────────────────────────────
step 11 "Checking YOLOv8s model weights"
if [ -f "$WEIGHTS_PATH" ]; then
    ok "Weights already present at $WEIGHTS_PATH"
else
    echo "  Downloading yolov8s.pt from Ultralytics (~22 MB)..."
    if command -v curl &>/dev/null; then
        curl -L "$WEIGHTS_URL" -o "$WEIGHTS_PATH" && ok "Weights downloaded." || warn "Download failed — place yolov8s.pt at $WEIGHTS_PATH manually.\n  URL: $WEIGHTS_URL"
    elif command -v wget &>/dev/null; then
        wget -q "$WEIGHTS_URL" -O "$WEIGHTS_PATH" && ok "Weights downloaded." || warn "Download failed — see $WEIGHTS_URL"
    else
        warn "Neither curl nor wget found. Download manually:\n  $WEIGHTS_URL -> $WEIGHTS_PATH"
    fi
fi

# ── STEP 12: Smoke test ───────────────────────────────────────────────────────
step 12 "Running smoke test"
python - <<'PYEOF'
import sys
pkgs = ["cv2", "numpy", "ultralytics", "mediapipe", "pandas", "yaml", "scipy", "filterpy", "matplotlib", "torch"]
failures = [p for p in pkgs if not __import__(p, globals(), locals(), [], 0) and False]
# reimport properly
failures = []
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
PYEOF
ok "All core packages import successfully."

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "  \033[0;32mSETUP COMPLETE\033[0m"
echo "============================================================"
echo ""
echo "  To run the project:"
echo "    cd \"$(pwd)\""
echo "    source .venv/bin/activate"
echo "    python src/main.py"
echo ""
echo "  For the full demo:"
echo "    python full_engagement_demo.py"
echo ""
echo "  Config file: config.yaml  (edit camera source, thresholds, etc.)"
echo ""
