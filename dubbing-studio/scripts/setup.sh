#!/usr/bin/env bash
# DubbingStudio Setup Script
# Run this once on your local machine to install all dependencies.
# Updated for: WhisperX + MADLAD-400 + Fish Speech 1.5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       DubbingStudio Setup                ║"
echo "║  WhisperX + MADLAD-400 + Fish Speech     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "Python: $python_version"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) and sys.version_info < (3, 13) else 1)" 2>/dev/null; then
    echo "Python version OK (3.10-3.12 required)"
else
    echo "ERROR: Python 3.10, 3.11, or 3.12 required."
    echo "Python 3.13 is NOT supported by WhisperX."
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
    echo ""
    echo "WARNING: Redis not found. Install for job queuing:"
    echo "  sudo apt install redis-server   (Ubuntu/Debian)"
    echo "  brew install redis              (macOS)"
    echo ""
fi

echo ""
echo "──────────────────────────────────────────"
echo "Step 1: Python virtual environment"
echo "──────────────────────────────────────────"

cd "$BACKEND_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment: $BACKEND_DIR/venv"
else
    echo "Virtual environment already exists"
fi

source venv/bin/activate
pip install --upgrade pip -q

echo ""
echo "──────────────────────────────────────────"
echo "Step 2: PyTorch with CUDA"
echo "──────────────────────────────────────────"

# Detect CUDA version from nvidia-smi
CUDA_VER=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "Driver version: $CUDA_VER"
fi

echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

echo ""
echo "──────────────────────────────────────────"
echo "Step 3: WhisperX (transcription + diarization)"
echo "──────────────────────────────────────────"

# IMPORTANT: Install WhisperX first, then reinstall PyTorch CUDA
# (WhisperX can overwrite PyTorch with CPU-only version)
pip install whisperx -q

# Reinstall PyTorch with CUDA (WhisperX may have overwritten it)
echo "Re-pinning PyTorch CUDA wheels..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# Fix onnxruntime for GPU diarization
pip uninstall -y onnxruntime 2>/dev/null || true
pip install --force-reinstall --no-cache-dir onnxruntime-gpu -q

echo ""
echo "──────────────────────────────────────────"
echo "Step 4: Fish Speech (voice cloning TTS)"
echo "──────────────────────────────────────────"

pip install fish-speech-lib -q 2>/dev/null || {
    echo "fish-speech-lib not available, trying fish-speech-rs..."
    pip install fish-speech-rs -q 2>/dev/null || {
        echo ""
        echo "NOTE: Fish Speech pip packages not found."
        echo "You may need to install from source:"
        echo "  git clone https://github.com/fishaudio/fish-speech.git"
        echo "  cd fish-speech && pip install -e .[cu124]"
        echo ""
    }
}

echo ""
echo "──────────────────────────────────────────"
echo "Step 5: Remaining backend dependencies"
echo "──────────────────────────────────────────"

pip install -r requirements.txt -q

echo ""
echo "──────────────────────────────────────────"
echo "Step 6: Verify GPU setup"
echo "──────────────────────────────────────────"

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f'  GPU {i}: {name} ({mem:.0f} GB)')
else:
    print('WARNING: CUDA not detected - pipeline will be very slow on CPU')
" || echo "Could not verify PyTorch GPU setup"

echo ""
echo "──────────────────────────────────────────"
echo "Step 7: Frontend"
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
echo "Step 8: Configuration"
echo "──────────────────────────────────────────"

cd "$BACKEND_DIR"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  ACTION REQUIRED: Set your Hugging Face token       ║"
    echo "╠══════════════════════════════════════════════════════╣"
    echo "║  1. Get a free token at:                            ║"
    echo "║     https://huggingface.co/settings/tokens           ║"
    echo "║                                                      ║"
    echo "║  2. Accept model terms at:                           ║"
    echo "║     https://huggingface.co/pyannote/speaker-diarization-3.1  ║"
    echo "║     https://huggingface.co/pyannote/segmentation-3.0 ║"
    echo "║                                                      ║"
    echo "║  3. Edit $BACKEND_DIR/.env                           ║"
    echo "║     Set HF_TOKEN=hf_your_token_here                 ║"
    echo "╚══════════════════════════════════════════════════════╝"
else
    echo ".env already exists (not overwriting)"
fi

echo ""
echo "══════════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════════"
echo ""
echo "  Models used:"
echo "    Transcription:  WhisperX (large-v3-turbo)"
echo "    Diarization:    pyannote community-1 (via WhisperX)"
echo "    Translation:    MADLAD-400 3B (Google, Apache 2.0)"
echo "    Voice cloning:  Fish Speech 1.5"
echo ""
echo "  To start DubbingStudio:"
echo "    ./scripts/start.sh"
echo ""
echo "  Or start manually:"
echo "    Terminal 1: redis-server"
echo "    Terminal 2: cd backend && source venv/bin/activate && celery -A app.core.celery_app worker -Q dubbing --concurrency=1 -l info"
echo "    Terminal 3: cd backend && source venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "  Then open: http://localhost:8000"
echo ""
