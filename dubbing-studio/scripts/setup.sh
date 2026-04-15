#!/usr/bin/env bash
# DubbingStudio Setup Script
# Run this once on your local machine to install all dependencies.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       DubbingStudio Setup                ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "Python: $python_version"
required_major=3
required_minor=10
if python3 -c "import sys; exit(0 if sys.version_info >= ($required_major, $required_minor) else 1)" 2>/dev/null; then
    echo "Python version OK"
else
    echo "ERROR: Python 3.10+ required. Install from https://python.org"
    exit 1
fi

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. CUDA support may not be available."
fi

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg is required but not installed."
    echo "Install: sudo apt install ffmpeg   (Ubuntu/Debian)"
    echo "     or: brew install ffmpeg       (macOS)"
    exit 1
fi
echo "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"

# Check Redis
if ! command -v redis-server &>/dev/null; then
    echo "WARNING: Redis not found. Install it for job queuing."
    echo "Install: sudo apt install redis-server   (Ubuntu/Debian)"
    echo "     or: brew install redis              (macOS)"
fi

echo ""
echo "──────────────────────────────────────────"
echo "Installing Python backend dependencies..."
echo "──────────────────────────────────────────"

cd "$BACKEND_DIR"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment: $BACKEND_DIR/venv"
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install PyTorch with CUDA 12.4 support (adjust CUDA version if needed)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# Install remaining requirements
echo "Installing remaining dependencies..."
pip install -r requirements.txt -q

echo ""
echo "──────────────────────────────────────────"
echo "Installing frontend dependencies..."
echo "──────────────────────────────────────────"

cd "$FRONTEND_DIR"

if command -v npm &>/dev/null; then
    npm install
    echo "Building frontend..."
    npm run build
    echo "Frontend built successfully."
else
    echo "WARNING: npm not found. Install Node.js from https://nodejs.org"
fi

echo ""
echo "──────────────────────────────────────────"
echo "Setting up configuration..."
echo "──────────────────────────────────────────"

cd "$BACKEND_DIR"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo ""
    echo "IMPORTANT: Edit $BACKEND_DIR/.env and set your HF_TOKEN"
    echo "Get a free Hugging Face token at: https://huggingface.co/settings/tokens"
    echo "You must accept the terms for pyannote/speaker-diarization-3.1 at:"
    echo "https://huggingface.co/pyannote/speaker-diarization-3.1"
fi

echo ""
echo "──────────────────────────────────────────"
echo "Setup complete!"
echo "──────────────────────────────────────────"
echo ""
echo "To start DubbingStudio, run:"
echo ""
echo "  ./scripts/start.sh"
echo ""
echo "Or start each component separately:"
echo ""
echo "  Terminal 1: redis-server"
echo "  Terminal 2: cd backend && source venv/bin/activate && celery -A app.core.celery_app worker -Q dubbing --concurrency=1 -l info"
echo "  Terminal 3: cd backend && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "Then open: http://localhost:8000"
